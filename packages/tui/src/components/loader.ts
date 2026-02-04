import type { TUI } from "../tui.js";
import { Text } from "./text.js";

/**
 * Loader component that updates every 80ms with a full-width KITT scanning bar animation
 */
export class Loader extends Text {
	private position = 0;
	private direction = 1;
	private intervalId: NodeJS.Timeout | null = null;
	private ui: TUI | null = null;

	constructor(
		ui: TUI,
		private spinnerColorFn: (str: string) => string,
		private messageColorFn: (str: string) => string,
		private message: string = "Loading...",
	) {
		super("", 1, 0);
		this.ui = ui;
		this.start();
	}

	render(width: number): string[] {
		const barWidth = Math.max(1, width - 2);
		const bar = this.buildBar(barWidth);
		const barLine = ` ${bar}${" ".repeat(Math.max(0, width - barWidth - 1))}`;
		return ["", barLine, ...super.render(width)];
	}

	start() {
		this.updateDisplay();
		this.intervalId = setInterval(() => {
			const barWidth = Math.max(1, (this.ui?.terminal.columns ?? 80) - 2);
			const maxPos = barWidth - 1;
			this.position = Math.min(this.position, maxPos);
			this.position += this.direction;
			if (this.position >= maxPos) {
				this.position = maxPos;
				this.direction = -1;
			} else if (this.position <= 0) {
				this.position = 0;
				this.direction = 1;
			}
			this.updateDisplay();
		}, 80);
	}

	stop() {
		if (this.intervalId) {
			clearInterval(this.intervalId);
			this.intervalId = null;
		}
	}

	setMessage(message: string) {
		this.message = message;
		this.updateDisplay();
	}

	private updateDisplay() {
		this.setText(this.messageColorFn(this.message));
		if (this.ui) {
			this.ui.requestRender();
		}
	}

	private buildBar(barWidth: number): string {
		const chars: string[] = [];
		for (let i = 0; i < barWidth; i++) {
			const dist = Math.abs(i - this.position);
			if (dist === 0) {
				chars.push(this.spinnerColorFn("━"));
			} else if (dist === 1) {
				chars.push(this.spinnerColorFn("─"));
			} else {
				chars.push(" ");
			}
		}
		return chars.join("");
	}
}
