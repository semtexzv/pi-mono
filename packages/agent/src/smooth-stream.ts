/**
 * Smooth stream processor — wraps a provider's AssistantMessageEventStream,
 * splitting bursty deltas into small tokens with pacing, and rebuilding
 * incremental partials so consumers can use `event.partial` directly.
 */

import { type AssistantMessage, type AssistantMessageEvent, parseStreamingJson } from "@mariozechner/pi-ai";

type ContentBlock = AssistantMessage["content"][number];
type DeltaType = "text_delta" | "thinking_delta" | "toolcall_delta";

export type ProviderStream = AsyncIterable<AssistantMessageEvent> & { result(): Promise<AssistantMessage> };

const RATE_BASE = 80; // initial chars/sec estimate
const RATE_MIN = 40;
const RATE_MAX = 2000;
const MAX_DELAY_MS = 18;
const BULK_THRESHOLD = 5; // queue entries before we flush at once

function delay(ms: number, signal?: AbortSignal): Promise<void> {
	if (ms <= 0) return Promise.resolve();
	return new Promise<void>((resolve) => {
		const t = setTimeout(done, ms);
		signal?.addEventListener("abort", done, { once: true });
		function done() {
			clearTimeout(t);
			signal?.removeEventListener("abort", done);
			resolve();
		}
	});
}

/** Clear a content block's text/thinking/arguments while keeping its structure. */
function emptyBlock(b: ContentBlock): ContentBlock {
	if (b.type === "text") return { type: "text", text: "" };
	if (b.type === "thinking") return { type: "thinking", thinking: "" };
	if (b.type === "toolCall") return { type: "toolCall", id: b.id, name: b.name, arguments: {} };
	return b;
}

/**
 * Snip one token from the front of `text`.
 * Text/thinking: single char, but markdown delimiters emitted atomically.
 * Toolcall JSON: always single char.
 */
function snipToken(text: string, isToolcall: boolean): [string, string] {
	if (!text) return ["", ""];
	if (isToolcall) {
		const c = Array.from(text);
		return [c[0], c.slice(1).join("")];
	}
	// Atomic markdown elements
	if (text.startsWith("```")) {
		const nl = text.indexOf("\n", 3);
		return nl !== -1 ? [text.slice(0, nl + 1), text.slice(nl + 1)] : ["```", text.slice(3)];
	}
	for (const pair of ["**", "__", "~~"]) {
		if (text.startsWith(pair)) return [pair, text.slice(2)];
	}
	if (text[0] === "\n") {
		let i = 0;
		while (i < text.length && text[i] === "\n") i++;
		return [text.slice(0, i), text.slice(i)];
	}
	for (const re of [/^#{1,6} /, /^\d+\.\s/, /^[-*+] /]) {
		const m = text.match(re);
		if (m) return [m[0], text.slice(m[0].length)];
	}
	if (text.startsWith("> ")) return ["> ", text.slice(2)];
	// Default: one codepoint
	const c = Array.from(text);
	return [c[0], c.slice(1).join("")];
}

/**
 * Wrap a provider event stream with smooth character-level emission.
 * Delta events are queued and drained as small tokens with pacing.
 * Structural/lifecycle events flush the queue first, then pass through.
 * All yielded events carry a correct incremental `partial`.
 */
export function createSmoothStream(source: ProviderStream, signal?: AbortSignal): ProviderStream {
	const queue: { type: DeltaType; contentIndex: number; remaining: string }[] = [];
	const toolcallJson = new Map<number, string>();
	let partial: AssistantMessage | null = null;
	let rate = RATE_BASE;
	let lastMs: number | undefined;

	function trackRate(chars: number): void {
		const now = Date.now();
		if (chars > 0 && lastMs !== undefined) {
			const observed = (chars / Math.max(1, now - lastMs)) * 1000;
			rate = rate * 0.7 + observed * 0.3;
		}
		lastMs = now;
	}

	function mergeNewBlocks(from: AssistantMessage): void {
		if (!partial || from.content.length <= partial.content.length) return;
		const merged = [...partial.content] as ContentBlock[];
		for (let i = merged.length; i < from.content.length; i++) merged.push(emptyBlock(from.content[i]));
		partial = { ...from, content: merged };
	}

	function applyToken(type: DeltaType, ci: number, tok: string): void {
		if (!partial) return;
		const block = partial.content[ci];
		if (!block) return;
		const next = [...partial.content] as ContentBlock[];
		if (type === "text_delta" && block.type === "text") next[ci] = { type: "text", text: block.text + tok };
		else if (type === "thinking_delta" && block.type === "thinking")
			next[ci] = { type: "thinking", thinking: block.thinking + tok };
		else if (type === "toolcall_delta" && block.type === "toolCall") {
			// The provider already parsed arguments in event.partial — but we
			// can't use it because we're emitting sub-tokens of the original
			// delta, so the provider's partial is ahead of what we've emitted.
			// Re-parse from our own accumulator. O(n^2) total but args are small.
			const acc = (toolcallJson.get(ci) ?? "") + tok;
			toolcallJson.set(ci, acc);
			next[ci] = { type: "toolCall", id: block.id, name: block.name, arguments: parseStreamingJson(acc) };
		} else return;
		partial = { ...partial, content: next };
	}

	function emitDelta(type: DeltaType, ci: number, delta: string): AssistantMessageEvent {
		applyToken(type, ci, delta);
		return { type, contentIndex: ci, delta, partial: partial! };
	}

	async function* drain(flush: boolean): AsyncGenerator<AssistantMessageEvent> {
		const bulk = flush || queue.length > BULK_THRESHOLD;
		while (queue.length > 0) {
			const e = queue[0];
			if (!e.remaining) {
				queue.shift();
				continue;
			}
			if (bulk) {
				const d = e.remaining;
				e.remaining = "";
				queue.shift();
				yield emitDelta(e.type, e.contentIndex, d);
				continue;
			}
			const [tok, rest] = snipToken(e.remaining, e.type === "toolcall_delta");
			if (!tok) {
				queue.shift();
				continue;
			}
			e.remaining = rest;
			if (!e.remaining) queue.shift();
			yield emitDelta(e.type, e.contentIndex, tok);
			if (queue.some((q) => q.remaining)) {
				const ms = Math.min(
					MAX_DELAY_MS,
					Math.round((tok.length / Math.max(RATE_MIN, Math.min(RATE_MAX, rate))) * 1000),
				);
				if (ms > 0) await delay(ms, signal);
			}
		}
	}

	async function* generate(): AsyncGenerator<AssistantMessageEvent> {
		for await (const ev of source) {
			switch (ev.type) {
				case "start":
					partial = { ...ev.partial, content: ev.partial.content.map(emptyBlock) };
					yield { type: "start", partial };
					break;
				case "text_delta":
				case "thinking_delta":
				case "toolcall_delta":
					trackRate(ev.delta.length);
					queue.push({ type: ev.type, contentIndex: ev.contentIndex, remaining: ev.delta });
					yield* drain(false);
					break;
				case "text_start":
				case "thinking_start":
				case "toolcall_start":
					yield* drain(true);
					if (ev.type === "toolcall_start") toolcallJson.set(ev.contentIndex, "");
					mergeNewBlocks(ev.partial);
					yield { ...ev, partial: partial! };
					break;
				case "text_end":
				case "thinking_end":
				case "toolcall_end":
					yield* drain(true);
					yield { ...ev, partial: partial! } as AssistantMessageEvent;
					break;
				case "done":
				case "error":
					yield* drain(true);
					yield ev;
					break;
			}
		}
		yield* drain(true);
	}

	const iter = generate();
	return { [Symbol.asyncIterator]: () => iter, result: () => source.result() };
}
