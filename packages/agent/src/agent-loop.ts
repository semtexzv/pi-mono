/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolResult,
	StreamFn,
} from "./types.js";

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	(async () => {
		const newMessages: AgentMessage[] = [...prompts];
		const currentContext: AgentContext = {
			...context,
			messages: [...context.messages, ...prompts],
		};

		stream.push({ type: "agent_start" });
		stream.push({ type: "turn_start" });
		for (const prompt of prompts) {
			stream.push({ type: "message_start", message: prompt });
			stream.push({ type: "message_end", message: prompt });
		}

		await runLoop(currentContext, newMessages, config, signal, stream, streamFn);
	})();

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	(async () => {
		const newMessages: AgentMessage[] = [];
		const currentContext: AgentContext = { ...context };

		stream.push({ type: "agent_start" });
		stream.push({ type: "turn_start" });

		await runLoop(currentContext, newMessages, config, signal, stream, streamFn);
	})();

	return stream;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

const SMOOTH_TEXT_DELTA_MIN_CHARS_PER_EMIT = 1;
const SMOOTH_TEXT_DELTA_MAX_CHARS_PER_EMIT = 2;
const SMOOTH_TEXT_DELTA_BASE_CHARS_PER_SECOND = 90;
const SMOOTH_TEXT_DELTA_MIN_CHARS_PER_SECOND = 40;
const SMOOTH_TEXT_DELTA_MAX_CHARS_PER_SECOND = 2000;
const SMOOTH_TEXT_DELTA_MAX_DELAY_MS = 18;
const SMOOTH_TEXT_DELTA_HIGH_WATERMARK = 160;
const SMOOTH_TEXT_DELTA_CATCHUP_WINDOW_MS = 220;

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

function appendTextDelta(message: AssistantMessage, contentIndex: number, delta: string): AssistantMessage {
	if (!delta) return message;
	const content = message.content[contentIndex];
	if (!content || content.type !== "text") return message;

	const nextContent = [...message.content];
	nextContent[contentIndex] = {
		...content,
		text: content.text + delta,
	};

	return {
		...message,
		content: nextContent,
	};
}

function waitWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
	if (ms <= 0) return Promise.resolve();

	return new Promise<void>((resolve) => {
		const timer = setTimeout(() => {
			cleanup();
			resolve();
		}, ms);

		const onAbort = () => {
			cleanup();
			resolve();
		};

		const cleanup = () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};

		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

async function emitSmoothedTextDeltaUpdates(
	event: Extract<AssistantMessageEvent, { type: "text_delta" }>,
	previousPartial: AssistantMessage,
	stream: EventStream<AgentEvent, AgentMessage[]>,
	targetCharsPerSecond: number,
	signal?: AbortSignal,
): Promise<AssistantMessage> {
	const pendingCodePoints = Array.from(event.delta);
	if (pendingCodePoints.length <= SMOOTH_TEXT_DELTA_MAX_CHARS_PER_EMIT) {
		stream.push({
			type: "message_update",
			assistantMessageEvent: event,
			message: { ...event.partial },
		});
		return event.partial;
	}

	let partial = previousPartial;
	while (pendingCodePoints.length > 0) {
		const backlog = pendingCodePoints.length;
		const emitChars =
			backlog >= SMOOTH_TEXT_DELTA_HIGH_WATERMARK
				? SMOOTH_TEXT_DELTA_MAX_CHARS_PER_EMIT
				: SMOOTH_TEXT_DELTA_MIN_CHARS_PER_EMIT;
		const chunk = pendingCodePoints.splice(0, emitChars).join("");

		partial = appendTextDelta(partial, event.contentIndex, chunk);
		stream.push({
			type: "message_update",
			assistantMessageEvent: {
				type: "text_delta",
				contentIndex: event.contentIndex,
				delta: chunk,
				partial,
			},
			message: { ...partial },
		});

		if (pendingCodePoints.length === 0) {
			break;
		}

		const clampedRate = clamp(
			targetCharsPerSecond,
			SMOOTH_TEXT_DELTA_MIN_CHARS_PER_SECOND,
			SMOOTH_TEXT_DELTA_MAX_CHARS_PER_SECOND,
		);
		const catchUpRate =
			backlog > SMOOTH_TEXT_DELTA_HIGH_WATERMARK
				? (backlog / SMOOTH_TEXT_DELTA_CATCHUP_WINDOW_MS) * 1000
				: clampedRate;
		const effectiveRate = clamp(
			Math.max(clampedRate, catchUpRate),
			SMOOTH_TEXT_DELTA_MIN_CHARS_PER_SECOND,
			SMOOTH_TEXT_DELTA_MAX_CHARS_PER_SECOND,
		);

		const delayMs =
			backlog > SMOOTH_TEXT_DELTA_HIGH_WATERMARK * 2
				? 0
				: Math.min(SMOOTH_TEXT_DELTA_MAX_DELAY_MS, Math.round((emitChars / effectiveRate) * 1000));
		await waitWithAbort(delayMs, signal);
	}

	return event.partial;
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
	streamFn?: StreamFn,
): Promise<void> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;
		let steeringAfterTools: AgentMessage[] | null = null;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				stream.push({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					stream.push({ type: "message_start", message });
					stream.push({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, stream, streamFn);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				stream.push({ type: "turn_end", message, toolResults: [] });
				stream.push({ type: "agent_end", messages: newMessages });
				stream.end(newMessages);
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");
			hasMoreToolCalls = toolCalls.length > 0;

			const toolResults: ToolResultMessage[] = [];
			if (hasMoreToolCalls) {
				const toolExecution = await executeToolCalls(
					currentContext.tools,
					message,
					signal,
					stream,
					config.getSteeringMessages,
				);
				toolResults.push(...toolExecution.toolResults);
				steeringAfterTools = toolExecution.steeringMessages ?? null;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			stream.push({ type: "turn_end", message, toolResults });

			// Get steering messages after turn completes
			if (steeringAfterTools && steeringAfterTools.length > 0) {
				pendingMessages = steeringAfterTools;
				steeringAfterTools = null;
			} else {
				pendingMessages = (await config.getSteeringMessages?.()) || [];
			}
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	stream.push({ type: "agent_end", messages: newMessages });
	stream.end(newMessages);
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;
	let estimatedIngressCharsPerSecond = SMOOTH_TEXT_DELTA_BASE_CHARS_PER_SECOND;
	let lastTextDeltaTimestampMs: number | undefined;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				stream.push({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_delta":
				if (partialMessage) {
					const deltaChars = Array.from(event.delta).length;
					const now = Date.now();
					if (deltaChars > 0 && lastTextDeltaTimestampMs !== undefined) {
						const elapsedMs = Math.max(1, now - lastTextDeltaTimestampMs);
						const observedCharsPerSecond = (deltaChars / elapsedMs) * 1000;
						estimatedIngressCharsPerSecond = estimatedIngressCharsPerSecond * 0.7 + observedCharsPerSecond * 0.3;
					}
					lastTextDeltaTimestampMs = now;

					partialMessage = await emitSmoothedTextDeltaUpdates(
						event,
						partialMessage,
						stream,
						estimatedIngressCharsPerSecond,
						signal,
					);
					context.messages[context.messages.length - 1] = partialMessage;
				}
				break;

			case "text_start":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					stream.push({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					stream.push({ type: "message_start", message: { ...finalMessage } });
				}
				stream.push({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	return await response.result();
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	tools: AgentTool<any>[] | undefined,
	assistantMessage: AssistantMessage,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
	getSteeringMessages?: AgentLoopConfig["getSteeringMessages"],
): Promise<{ toolResults: ToolResultMessage[]; steeringMessages?: AgentMessage[] }> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const results: ToolResultMessage[] = [];
	let steeringMessages: AgentMessage[] | undefined;

	for (let index = 0; index < toolCalls.length; index++) {
		const toolCall = toolCalls[index];
		const tool = tools?.find((t) => t.name === toolCall.name);

		stream.push({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		let result: AgentToolResult<any>;
		let isError = false;

		try {
			if (!tool) throw new Error(`Tool ${toolCall.name} not found`);

			const validatedArgs = validateToolArguments(tool, toolCall);

			result = await tool.execute(toolCall.id, validatedArgs, signal, (partialResult) => {
				stream.push({
					type: "tool_execution_update",
					toolCallId: toolCall.id,
					toolName: toolCall.name,
					args: toolCall.arguments,
					partialResult,
				});
			});
		} catch (e) {
			result = {
				content: [{ type: "text", text: e instanceof Error ? e.message : String(e) }],
				details: {},
			};
			isError = true;
		}

		stream.push({
			type: "tool_execution_end",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			result,
			isError,
		});

		const toolResultMessage: ToolResultMessage = {
			role: "toolResult",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			content: result.content,
			details: result.details,
			isError,
			timestamp: Date.now(),
		};

		results.push(toolResultMessage);
		stream.push({ type: "message_start", message: toolResultMessage });
		stream.push({ type: "message_end", message: toolResultMessage });

		// Check for steering messages - skip remaining tools if user interrupted
		if (getSteeringMessages) {
			const steering = await getSteeringMessages();
			if (steering.length > 0) {
				steeringMessages = steering;
				const remainingCalls = toolCalls.slice(index + 1);
				for (const skipped of remainingCalls) {
					results.push(skipToolCall(skipped, stream));
				}
				break;
			}
		}
	}

	return { toolResults: results, steeringMessages };
}

function skipToolCall(
	toolCall: Extract<AssistantMessage["content"][number], { type: "toolCall" }>,
	stream: EventStream<AgentEvent, AgentMessage[]>,
): ToolResultMessage {
	const result: AgentToolResult<any> = {
		content: [{ type: "text", text: "Skipped due to queued user message." }],
		details: {},
	};

	stream.push({
		type: "tool_execution_start",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		args: toolCall.arguments,
	});
	stream.push({
		type: "tool_execution_end",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		result,
		isError: true,
	});

	const toolResultMessage: ToolResultMessage = {
		role: "toolResult",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		content: result.content,
		details: {},
		isError: true,
		timestamp: Date.now(),
	};

	stream.push({ type: "message_start", message: toolResultMessage });
	stream.push({ type: "message_end", message: toolResultMessage });

	return toolResultMessage;
}
