import vscode from 'vscode';
import { t } from '../../i18n';
import { logToolFlowDiagnostics } from '../debug';
import type { RequestKind } from '../routing';
import { ACTIVATE_TOOL_PREFIX, MAX_PREFLIGHT_ROUNDS_PER_USER_REQUEST, PREFLIGHT_ACTIVATE_CALL_ID_PREFIX, TOOL_DRIFT_NOTICE_START, VISION_PROXY_NOTICE_START } from './consts';
import { createToolDriftNotice, filterProviderNotices } from './notices';
import {
    createPreflightToolCallId,
    filterPreflightControlFlow,
    inspectActivatePreflight,
} from './preflight';

interface ToolFlowOptions {
	stabilizeToolList: boolean;
	messages: readonly vscode.LanguageModelChatRequestMessage[];
	tools: readonly vscode.LanguageModelChatTool[] | undefined;
	progress: vscode.Progress<vscode.LanguageModelResponsePart>;
	requestKind: RequestKind;
}

interface ToolFlowResult {
	preflightHandled: boolean;
	messages: readonly vscode.LanguageModelChatRequestMessage[];
	initialResponseNotice?: string;
}

export function processToolFlow({
	stabilizeToolList,
	messages,
	tools,
	progress,
	requestKind,
}: ToolFlowOptions): ToolFlowResult {
	// Fast path: in the common case there are no preflight callIds and no
	// provider-notice markers. Skip both filtering passes entirely.
	const needsFiltering = hasFilterableContent(messages);
	const filteredMessages = needsFiltering
		? filterProviderNotices(filterPreflightControlFlow(messages))
		: messages;
	const messagesFiltered = filteredMessages !== messages;

	if (!stabilizeToolList) {
		logToolFlowDiagnostics({
			requestKind,
			tools,
			messagesFiltered,
			preflight: 'skipped',
		});
		return {
			preflightHandled: false,
			messages: filteredMessages,
		};
	}

	const activatePreflight = inspectActivatePreflight(messages, tools);
	if (activatePreflight.remainingActivatorNames.length > 0) {
		if (activatePreflight.rounds >= MAX_PREFLIGHT_ROUNDS_PER_USER_REQUEST) {
			logToolFlowDiagnostics({
				requestKind,
				tools,
				messagesFiltered,
				preflight: 'round-limit',
				activatePreflight,
			});
			throw new Error(
				t('request.preflightRoundLimitExceeded', MAX_PREFLIGHT_ROUNDS_PER_USER_REQUEST),
			);
		}

		const nextRound = activatePreflight.rounds + 1;
		logToolFlowDiagnostics({
			requestKind,
			tools,
			messagesFiltered,
			preflight: 'handled',
			activatePreflight,
			nextRound,
		});
		for (const toolName of activatePreflight.remainingActivatorNames) {
			progress.report(
				new vscode.LanguageModelToolCallPart(
					createPreflightToolCallId(nextRound, toolName),
					toolName,
					{},
				),
			);
		}

		return { preflightHandled: true, messages };
	}

	const hasUnexpandedActivateTools =
		activatePreflight.rounds > 0 &&
		tools?.some((tool) => tool.name.startsWith(ACTIVATE_TOOL_PREFIX));
	logToolFlowDiagnostics({
		requestKind,
		tools,
		messagesFiltered,
		preflight: 'ready',
		activatePreflight,
		initialResponseNotice: hasUnexpandedActivateTools,
	});

	return {
		preflightHandled: false,
		messages: filteredMessages,
		initialResponseNotice: hasUnexpandedActivateTools ? createToolDriftNotice() : undefined,
	};
}

/**
 * Quick pre-scan to determine whether the message list contains any
 * content that needs filtering (preflight callIds or provider-notice
 * markers). Returns false for the common case so we can skip the
 * two full-pass filters entirely.
 */
function hasFilterableContent(
	messages: readonly vscode.LanguageModelChatRequestMessage[],
): boolean {
	for (const message of messages) {
		for (const part of message.content) {
			// Preflight: check tool-call / tool-result parts for the preflight prefix.
			if (
				(part instanceof vscode.LanguageModelToolCallPart ||
					part instanceof vscode.LanguageModelToolResultPart) &&
				part.callId.startsWith(PREFLIGHT_ACTIVATE_CALL_ID_PREFIX)
			) {
				return true;
			}
			// Provider notices: check text parts in assistant messages for marker blocks.
			if (
				message.role === vscode.LanguageModelChatMessageRole.Assistant &&
				part instanceof vscode.LanguageModelTextPart
			) {
				const text = part.value;
				if (
					text.includes(TOOL_DRIFT_NOTICE_START) ||
					text.includes(VISION_PROXY_NOTICE_START)
				) {
					return true;
				}
			}
		}
	}
	return false;
}
