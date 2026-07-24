import vscode from 'vscode';
import { createUserFacingError } from '../client';
import { logger } from '../logger';
import type { DeepSeekToolCall, DeepSeekUsage } from '../types';
import {
	observeCancellationToken,
	type CacheDiagnosticsRun,
	type ReplayMarkerReportTrigger,
} from './debug';
import {
	createReplayMarkerPart,
	hasReplayMarkerMetadata,
	type ReplayMarkerMetadata,
} from './replay';
import type { PreparedChatRequest } from './request';
import { formatRequestLogLine, type RequestKind } from './routing';

interface ResponseStreamState {
	accumulatedReasoning: string;
	emittedToolCallIds: string[];
	initialResponseNoticeReported: boolean;
	replayMarkerReported: boolean;
}

const COPILOT_USAGE_DATA_PART_MIME = 'usage';

/** Number of consecutive replay marker failures before warning the user. */
const REPLAY_MARKER_FAILURE_WARN_THRESHOLD = 3;

/** Track consecutive replay marker failures per endpoint (by baseUrl). */
const replayFailureCount = new Map<string, number>();

export interface StreamChatCompletionOptions {
	prepared: PreparedChatRequest;
	progress: vscode.Progress<vscode.LanguageModelResponsePart>;
	token: vscode.CancellationToken;
	initialResponseNotice?: string;
	getCharsPerToken: () => number;
	setCharsPerToken: (charsPerToken: number) => void;
}

export function streamChatCompletion({
	prepared,
	progress,
	token,
	initialResponseNotice,
	getCharsPerToken,
	setCharsPerToken,
}: StreamChatCompletionOptions): Promise<void> {
	const state: ResponseStreamState = {
		accumulatedReasoning: '',
		emittedToolCallIds: [],
		initialResponseNoticeReported: false,
		replayMarkerReported: false,
	};
	const cancelListener = observeCancellationToken(token, prepared.cacheDiagnostics);

	return prepared.client
		.streamChatCompletion(
			prepared.request,
			{
				onContent: (content: string) => {
					reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
					progress.report(new vscode.LanguageModelTextPart(content));
				},

				onThinking: (text: string) => {
					reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
					handleThinking(text, state, progress);
				},

				onToolCall: (toolCall: DeepSeekToolCall) => {
					reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
					handleToolCall(toolCall, state, progress);
				},

				onError: (error: Error) => {
					throw createUserFacingError(error);
				},

				onDone: () => {
					reportReplayMarkerOnce(prepared, progress, state, 'done');
					finalizeReplayDiagnostics(
						prepared.trailingToolResultIds,
						state,
						prepared.cacheDiagnostics,
					);
				},

				onUsage: (usage) => {
					const charsPerToken = updateCharsPerToken(
						prepared.totalRequestChars,
						usage,
						getCharsPerToken(),
					);
					setCharsPerToken(charsPerToken);
					prepared.cacheDiagnostics.onUsage(usage, charsPerToken);
					reportCopilotContextUsage(progress, usage, prepared.requestKind);
				},
			},
			token,
		)
		.then(undefined, (error) => {
			reportSkippedReplayMarkerIfNeeded(
				prepared,
				state,
				token.isCancellationRequested ? 'cancelled' : 'stream-error',
				error,
			);
			throw error;
		})
		.then(() => {
			if (token.isCancellationRequested) {
				reportSkippedReplayMarkerIfNeeded(prepared, state, 'cancelled');
			}
		})
		.finally(() => {
			cancelListener.dispose();
		});
}

function reportInitialResponseNoticeOnce(
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	initialResponseNotice: string | undefined,
): void {
	if (!initialResponseNotice || state.initialResponseNoticeReported) {
		return;
	}
	state.initialResponseNoticeReported = true;
	progress.report(new vscode.LanguageModelTextPart(initialResponseNotice));
}

function reportReplayMarkerOnce(
	prepared: PreparedChatRequest,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	trigger: ReplayMarkerReportTrigger,
): void {
	if (state.replayMarkerReported) {
		return;
	}
	state.replayMarkerReported = true;
	reportReplayMarker(prepared, progress, state, trigger);
}

function reportSkippedReplayMarkerIfNeeded(
	prepared: PreparedChatRequest,
	state: ResponseStreamState,
	reason: 'cancelled' | 'stream-error',
	error?: unknown,
): void {
	if (state.replayMarkerReported) {
		return;
	}
	state.replayMarkerReported = true;
	prepared.cacheDiagnostics.onReplayMarkerReport({
		status: 'skipped',
		reason,
		visionTextChars: prepared.visionMarkerTextChars,
		reasoningTextChars: state.accumulatedReasoning.length || undefined,
		error,
	});
}

function reportReplayMarker(
	prepared: PreparedChatRequest,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	trigger: ReplayMarkerReportTrigger,
): void {
	const metadata = getReplayMarkerMetadata(prepared, state);
	if (!hasReplayMarkerMetadata(metadata)) {
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'skipped',
			trigger,
			reason: 'no-replay-data',
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
		});
		return;
	}

	try {
		const markerPart = createReplayMarkerPart(metadata);
		progress.report(markerPart);
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'reported',
			trigger,
			markerBytes: markerPart.data.byteLength,
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
		});
		// Reset failure counter on success.
		replayFailureCount.delete(prepared.client.endpointBaseUrl);
	} catch (error) {
		const prev = replayFailureCount.get(prepared.client.endpointBaseUrl) ?? 0;
		const current = prev + 1;
		replayFailureCount.set(prepared.client.endpointBaseUrl, current);
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'failed',
			trigger,
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
			error,
		});
		logger.warn(
			formatRequestLogLine(prepared.requestKind, 'Failed to report replay marker'),
			error,
		);

		if (current >= REPLAY_MARKER_FAILURE_WARN_THRESHOLD) {
			// Show a one-time warning when replay markers consistently fail.
			// The prompt cache hit rate depends on replay markers carrying
			// reasoning_content across turns. Without them, DeepSeek's prefix
			// cache is invalidated and costs increase ~50x.
			const message = current === REPLAY_MARKER_FAILURE_WARN_THRESHOLD
				? vscode.window.showWarningMessage(
					'DeepSeek Copilot: Replay markers are failing. Prompt cache hit rate may degrade, leading to higher API costs. Check the "DeepSeek" output channel for details.',
					'Show Logs',
				)
				: undefined;

			if (message) {
				void message.then((choice) => {
					if (choice === 'Show Logs') {
						logger.show();
					}
				});
			}
		}
	}
}

function getReplayMarkerMetadata(
	prepared: PreparedChatRequest,
	state: ResponseStreamState,
): ReplayMarkerMetadata {
	return {
		...prepared.replayMarkerMetadata,
		reasoningText: state.accumulatedReasoning || undefined,
	};
}

function handleThinking(
	text: string,
	state: ResponseStreamState,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
): void {
	state.accumulatedReasoning += text;

	// LanguageModelThinkingPart is a proposed API; the project root augmentation provides types.
	progress.report(
		new vscode.LanguageModelThinkingPart(text) as unknown as vscode.LanguageModelResponsePart,
	);
}

function handleToolCall(
	toolCall: DeepSeekToolCall,
	state: ResponseStreamState,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
): void {
	state.emittedToolCallIds.push(toolCall.id);

	try {
		const args = JSON.parse(toolCall.function.arguments);
		progress.report(
			new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.function.name, args),
		);
	} catch (error) {
		logger.warn(
			'Failed to parse tool call arguments, falling back to empty object. toolCallId=',
			toolCall.id,
			error,
		);
		// Inject a notice before the tool call so the user sees that the
		// arguments were corrupted, rather than wondering why the tool failed
		// with no explanation.
		const reason = error instanceof Error ? error.message : String(error);
		const notice = `\n\n> ⚠️ Tool call arguments for \`${toolCall.function.name}\` could not be parsed (${reason}). The tool was invoked with empty input.\n\n`;
		progress.report(new vscode.LanguageModelTextPart(notice));
		progress.report(
			new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.function.name, {}),
		);
	}
}

function finalizeReplayDiagnostics(
	trailingToolResultIds: readonly string[],
	state: ResponseStreamState,
	cacheDiagnostics: CacheDiagnosticsRun,
): void {
	cacheDiagnostics.onDone({
		reasoningTextChars: state.accumulatedReasoning.length,
		emittedToolCalls: state.emittedToolCallIds.length,
		trailingToolResults: trailingToolResultIds.length,
	});
}

function updateCharsPerToken(
	totalRequestChars: number,
	usage: DeepSeekUsage,
	charsPerToken: number,
): number {
	if (totalRequestChars > 0 && usage.prompt_tokens > 0) {
		const observedRatio = totalRequestChars / usage.prompt_tokens;
		return charsPerToken * 0.7 + observedRatio * 0.3;
	}
	return charsPerToken;
}

function reportCopilotContextUsage(
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	usage: DeepSeekUsage,
	requestKind: RequestKind,
): void {
	const data = {
		prompt_tokens: usage.prompt_tokens,
		completion_tokens: usage.completion_tokens,
		total_tokens: usage.total_tokens,
		prompt_tokens_details: {
			cached_tokens: usage.prompt_cache_hit_tokens ?? 0,
		},
	};

	try {
		progress.report(
			new vscode.LanguageModelDataPart(
				new TextEncoder().encode(JSON.stringify(data)),
				COPILOT_USAGE_DATA_PART_MIME,
			),
		);
	} catch (error) {
		logger.warn(formatRequestLogLine(requestKind, 'Failed to report usage data'), error);
	}
}
