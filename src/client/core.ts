import type { CancellationToken } from 'vscode';
import { safeStringify } from '../json';
import { logger } from '../logger';
import type {
    DeepSeekRequest,
    DeepSeekStreamChunk,
    DeepSeekToolCall,
    DeepSeekUsage,
    StreamCallbacks,
} from '../types';
import {
    createHttpError,
    DeepSeekRequestError,
    formatRequestError,
    normalizeRequestError,
} from './error';

const REQUEST_TIMEOUT_MS = 180_000;
const MAX_CONNECT_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 2000, 4000];

/**
 * Lightweight SSE-streaming DeepSeek API client.
 * No external dependencies — uses Node's built-in fetch.
 */
export class DeepSeekClient {
	constructor(
		private readonly baseUrl: string,
		private readonly apiKey: string,
	) {}

	/**
	 * Stream a chat completion from the DeepSeek API.
	 * Automatically retries on transient connection errors (up to 3 retries
	 * with 1s / 2s / 4s exponential backoff). HTTP errors and cancellations
	 * are not retried.
	 */
	async streamChatCompletion(
		request: DeepSeekRequest,
		callbacks: StreamCallbacks,
		cancellationToken?: CancellationToken,
	): Promise<void> {
		for (let attempt = 0; attempt <= MAX_CONNECT_RETRIES; attempt++) {
			if (cancellationToken?.isCancellationRequested) {
				return;
			}

			if (attempt > 0) {
				const delay = RETRY_DELAYS_MS[attempt - 1];
				logger.warn(
					`Retrying request after connection error (attempt ${attempt + 1}/${MAX_CONNECT_RETRIES + 1}, waiting ${delay}ms)`,
				);
				await new Promise((resolve) => setTimeout(resolve, delay));
			}

			try {
				await this._performStreamRequest(request, callbacks, cancellationToken);
				return;
			} catch (error) {
				if (cancellationToken?.isCancellationRequested) {
					return;
				}

				if (!isRetryableError(error) || attempt >= MAX_CONNECT_RETRIES) {
					const normalizedError = normalizeRequestError(error, {
						baseUrl: this.baseUrl,
						request,
					});
					logger.error(
						'DeepSeek request failed:',
						formatRequestError(normalizedError),
					);
					callbacks.onError(normalizedError);
					return;
				}

				// Connection error — will retry on next iteration
				const brief = error instanceof Error ? error.message : String(error);
				logger.warn(
					`Connect error (attempt ${attempt + 1}/${MAX_CONNECT_RETRIES + 1}): ${brief}`,
				);
			}
		}
	}

	/**
	 * Execute a single streaming request attempt.
	 * Throws on failure — retry logic is handled by the caller.
	 */
	private async _performStreamRequest(
		request: DeepSeekRequest,
		callbacks: StreamCallbacks,
		cancellationToken?: CancellationToken,
	): Promise<void> {
		const controller = new AbortController();
		const cancelListener = cancellationToken?.onCancellationRequested(() => {
			controller.abort();
		});
		if (cancellationToken?.isCancellationRequested) {
			controller.abort();
		}

		const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

		try {
			// Request usage stats in streaming responses so we can calibrate token counting.
			const requestBody = {
				...request,
				stream_options: { include_usage: true },
			};

			const response = await fetch(`${this.baseUrl}/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${this.apiKey}`,
				},
				body: safeStringify(requestBody),
				signal: controller.signal,
			});

			if (!response.ok) {
				throw await createHttpError(response, { baseUrl: this.baseUrl, request });
			}

			if (!response.body) {
				throw new Error('No response body received');
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			let latestUsage: DeepSeekUsage | undefined;

			// Accumulate tool call deltas by index, then emit on finish_reason=stop/tool_calls
			const pendingToolCalls = new Map<number, DeepSeekToolCall>();

			while (true) {
				if (cancellationToken?.isCancellationRequested) {
					controller.abort();
					return;
				}

				const { done, value } = await reader.read();
				if (done) {
					// Flush any remaining tool calls before the stream ends without [DONE].
					for (const tc of pendingToolCalls.values()) {
						callbacks.onToolCall(tc);
					}
					pendingToolCalls.clear();
					break;
				}

				buffer += decoder.decode(value, { stream: true });

				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					const trimmed = line.trim();

					if (!trimmed || trimmed.startsWith(':')) {
						continue;
					}

					if (trimmed === 'data: [DONE]') {
						// Flush any remaining tool calls
						for (const tc of pendingToolCalls.values()) {
							callbacks.onToolCall(tc);
						}
						pendingToolCalls.clear();
						reportFinalUsage(callbacks, latestUsage);
						callbacks.onDone();
						return;
					}

					if (!trimmed.startsWith('data: ')) {
						continue;
					}

					const jsonStr = trimmed.slice(6);
					try {
						const chunk: DeepSeekStreamChunk = JSON.parse(jsonStr);
						const choice = chunk.choices?.[0];

						// Some OpenAI-compatible providers emit usage on every streaming chunk.
						// Keep only the latest value and report it once when the stream completes.
						if (chunk.usage) {
							latestUsage = chunk.usage;
						}

						if (!choice) {
							continue;
						}

						// Thinking content → report with correct field name so VS Code renders collapsible blocks
						const reasoning = choice.delta.reasoning_content;
						if (reasoning) {
							callbacks.onThinking(reasoning);
						}

						// Regular content
						if (choice.delta.content) {
							callbacks.onContent(choice.delta.content);
						}

						// Tool calls — accumulate deltas by index
						if (choice.delta.tool_calls) {
							for (const tc of choice.delta.tool_calls) {
								let pending = pendingToolCalls.get(tc.index);
								if (!pending) {
									if (!tc.id) {
										logger.warn(
											`Received tool call delta without id for new tool call index ${tc.index}; skipping delta.`,
										);
										continue;
									}
									pending = {
										id: tc.id,
										type: 'function',
										function: { name: '', arguments: '' },
									};
									pendingToolCalls.set(tc.index, pending);
								}
								if (tc.function?.name) {
									pending.function.name += tc.function.name;
								}
								if (tc.function?.arguments) {
									pending.function.arguments += tc.function.arguments;
								}
							}
						}

						// Flush pending tool calls on finish
						if (choice.finish_reason === 'tool_calls' || choice.finish_reason === 'stop') {
							for (const tc of pendingToolCalls.values()) {
								callbacks.onToolCall(tc);
							}
							pendingToolCalls.clear();
						}
					} catch (e) {
						// JSON parse failure means the stream is corrupted — abort.
					// Discard all pending tool calls without emitting them:
					// at this point we cannot distinguish complete tool calls
					// (accumulated across chunks but not yet flushed) from
					// partially-received ones. Emitting incomplete tool calls
					// would cause downstream agent failures. The retry wrapper
					// will re-request and deliver complete data.
					pendingToolCalls.clear();
						const message = e instanceof Error ? e.message : String(e);
						throw new Error(
							`Failed to parse SSE chunk: ${jsonStr.slice(0, 200)} — ${message}`,
						);
					}
				}
			}

			reportFinalUsage(callbacks, latestUsage);
			callbacks.onDone();
		} catch (error) {
			if (isAbortError(error) && cancellationToken?.isCancellationRequested) {
				return;
			}
			// Re-throw to let the retry wrapper decide
			throw error;
		} finally {
			clearTimeout(timeout);
			cancelListener?.dispose();
		}
	}
}

function reportFinalUsage(callbacks: StreamCallbacks, usage: DeepSeekUsage | undefined): void {
	if (!usage || !callbacks.onUsage) {
		return;
	}
	callbacks.onUsage(usage);
}

function isAbortError(error: unknown): boolean {
	return error instanceof Error && error.name === 'AbortError';
}

/**
 * Determine whether a fetch error is retryable.
 * HTTP errors (wrapped in DeepSeekRequestError) and cancellations are NOT retryable.
 * Only transient network-level errors (connection refused, DNS failure, reset, timeout)
 * are retryable. Application-level errors (JSON parse, serialization, etc.) are not.
 */
function isRetryableError(error: unknown): boolean {
	if (isAbortError(error)) {
		return false;
	}
	if (error instanceof DeepSeekRequestError) {
		return false;
	}
	// Only retry transient network errors from fetch / Node.js internals.
	// Non-network TypeErrors (e.g. JSON parse, safeStringify) must not be retried.
	if (error instanceof TypeError) {
		return isNetworkError(error);
	}
	// Other Error subclasses (SyntaxError, RangeError, etc.) are application bugs.
	if (error instanceof Error) {
		return false;
	}
	// Unknown thrown primitives — do not retry.
	return false;
}

/** Network-related error codes / messages that indicate a transient failure. */
const RETRYABLE_NETWORK_PATTERNS = [
	'fetch failed',
	'ECONNREFUSED',
	'ECONNRESET',
	'ETIMEDOUT',
	'ENOTFOUND',
	'EAI_AGAIN',
	'EPIPE',
	'ERR_INTERNET_DISCONNECTED',
	'ERR_PROXY_CONNECTION_FAILED',
	'ERR_CONNECTION_RESET',
	'ERR_CONNECTION_REFUSED',
	'ERR_CONNECTION_TIMED_OUT',
	'UND_ERR_CONNECT_TIMEOUT',
	'UND_ERR_HEADERS_TIMEOUT',
	'UND_ERR_SOCKET',
];

function isNetworkError(error: TypeError): boolean {
	const message = error.message.toLowerCase();
	if (
		RETRYABLE_NETWORK_PATTERNS.some((pattern) =>
			message.includes(pattern.toLowerCase()),
		)
	) {
		return true;
	}
	// undici wraps low-level socket errors (ECONNRESET, EPIPE, etc.) in a
	// TypeError("terminated") where the real error code lives in cause.code.
	// Walk the cause chain to detect transient network failures that are not
	// visible in the top-level message.
	return hasNetworkErrorCause(error);
}

/**
 * Recursively check the error cause chain for known network error codes.
 * Handles patterns like:
 *   TypeError("terminated") → cause: { code: "ECONNRESET", message: "read ECONNRESET" }
 *   TypeError("fetch failed") → cause: { code: "UND_ERR_SOCKET", … }
 */
function hasNetworkErrorCause(error: unknown): boolean {
	if (!(error instanceof Error)) {
		return false;
	}
	const cause = (error as Error & { cause?: unknown }).cause;
	if (!cause) {
		return false;
	}

	const causeObj = typeof cause === 'object' && cause !== null
		? (cause as Record<string, unknown>)
		: null;

	// Check cause.code (e.g., "ECONNRESET", "UND_ERR_SOCKET")
	if (causeObj) {
		const code = causeObj.code;
		if (typeof code === 'string') {
			const codeLower = code.toLowerCase();
			if (
				RETRYABLE_NETWORK_PATTERNS.some(
					(p) => p.toLowerCase() === codeLower,
				)
			) {
				return true;
			}
		}
		// Check cause.message (e.g., "read ECONNRESET")
		const causeMsg = causeObj.message;
		if (typeof causeMsg === 'string') {
			const msgLower = causeMsg.toLowerCase();
			if (
				RETRYABLE_NETWORK_PATTERNS.some((p) =>
					msgLower.includes(p.toLowerCase()),
				)
			) {
				return true;
			}
		}
	}

	// Recurse into nested Error causes
	if (cause instanceof Error) {
		return hasNetworkErrorCause(cause);
	}
	return false;
}
