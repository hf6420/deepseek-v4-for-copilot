import vscode from 'vscode';
import { getBaseUrl } from '../config';
import { safeStringify } from '../json';
import { logger } from '../logger';
import type { DeepSeekMessage, DeepSeekTool, DeepSeekToolCall } from '../types';
import { getEndpointCompatibility } from './compat';
import { parseFirstReplayMarker } from './replay';

/**
 * Convert VS Code chat messages to DeepSeek format.
 * Injects marker-replayed reasoning_content for assistant messages
 * only when the endpoint supports it.
 */
export function convertMessages(
	messages: readonly vscode.LanguageModelChatRequestMessage[],
	isThinkingModel: boolean,
): DeepSeekMessage[] {
	const compat = getEndpointCompatibility(getBaseUrl());
	const shouldInjectReasoning = isThinkingModel && compat.sendReasoningContent;
	const result: DeepSeekMessage[] = [];

	for (const message of messages) {
		const role = mapRole(message.role);

		let content = '';
		let thinkingContent = '';
		const toolCalls: DeepSeekToolCall[] = [];
		const toolResults: Array<{ callId: string; content: string }> = [];

		for (const part of message.content) {
			if (part instanceof vscode.LanguageModelTextPart) {
				content += part.value;
			} else if (isLanguageModelThinkingPart(part)) {
				thinkingContent += normalizeThinkingPartText(part.value);
			} else if (part instanceof vscode.LanguageModelToolCallPart) {
				toolCalls.push({
					id: part.callId,
					type: 'function',
					function: {
						name: part.name,
						arguments: safeStringify(part.input),
					},
				});
			} else if (part instanceof vscode.LanguageModelToolResultPart) {
				let toolContent = '';
				for (const item of part.content) {
					if (item instanceof vscode.LanguageModelTextPart) {
						toolContent += item.value;
					}
				}
				toolResults.push({
					callId: part.callId,
					content: toolContent || safeStringify(part.content),
				});
			} else if (part instanceof vscode.LanguageModelDataPart) {
				// Data parts carry binary metadata (e.g. replay markers, usage info).
				// They are never sent as content to DeepSeek — skip silently.
			} else {
				// Unknown part type — try duck-type extraction before falling through.
				// VS Code Insiders / minified builds may obfuscate constructor names
				// (e.g. LanguageModelPromptTsxPart → "i"), so instanceof checks above
				// can miss real types.
				const duckText = tryExtractUnknownPartText(part);
				if (duckText !== undefined) {
					content += duckText;
				} else {
					// Truly unknown — log once per session at debug level to avoid
					// spamming the output channel. Do not throw; future VS Code API
					// additions may introduce new part types.
					logger.debug(
						'Unknown LanguageModelChatRequestMessage part type encountered during conversion:',
						part?.constructor?.name ?? part,
					);
				}
			}
		}

		if (role === 'assistant') {
			if (content || toolCalls.length > 0) {
				const replayMarker = shouldInjectReasoning ? parseFirstReplayMarker(message) : undefined;
				const msg: DeepSeekMessage = {
					role: 'assistant' as const,
					content: content || (toolCalls.length > 0 ? null : ''),
				};

				if (toolCalls.length > 0) {
					msg.tool_calls = toolCalls;
				}

				if (shouldInjectReasoning) {
					const reasoning = getReasoningContent(replayMarker, thinkingContent);
					if (reasoning) {
						msg.reasoning_content = reasoning;
					}
					// When reasoning is empty (e.g. thinking was disabled for this turn),
					// omit the field entirely so the JSON prefix matches the original
					// DeepSeek response, preserving prompt cache hit rate.
				}

				result.push(msg);
			}
		} else {
			if (content) {
				result.push({
					role: role,
					content: content,
				});
			}
		}

		// Tool result messages follow their associated assistant message
		for (const tr of toolResults) {
			result.push({
				role: 'tool',
				content: tr.content,
				tool_call_id: tr.callId,
			});
		}
	}

	return result;
}

function getReasoningContent(
	replayMarker: ReturnType<typeof parseFirstReplayMarker>,
	thinkingContent: string,
): string {
	if (replayMarker?.valid && replayMarker.reasoningText) {
		return replayMarker.reasoningText;
	}
	return thinkingContent;
}

function isLanguageModelThinkingPart(part: unknown): part is vscode.LanguageModelThinkingPart {
	return (
		typeof vscode.LanguageModelThinkingPart === 'function' &&
		part instanceof vscode.LanguageModelThinkingPart
	);
}

function normalizeThinkingPartText(value: string | string[]): string {
	return Array.isArray(value) ? value.join('') : value;
}

/**
 * Try to extract text from an unrecognized content part via duck-typing.
 *
 * VS Code Insiders / minified builds may obfuscate constructor names (e.g.
 * {@link vscode.LanguageModelPromptTsxPart} → `"i"`), so `instanceof` checks
 * can miss known types. This fallback inspects the object shape instead.
 *
 * @returns The extracted text string, or `undefined` if the part shape is
 *          not recognized as text-bearing.
 */
function tryExtractUnknownPartText(part: unknown): string | undefined {
	if (!part || typeof part !== 'object') {
		return undefined;
	}

	// LanguageModelPromptTsxPart (proposed API) — carries TSX-shaped prompt text.
	// In minified builds constructor.name is obfuscated, but the `value` property
	// is always present and is either a string or a structured object.
	if ('value' in part) {
		const value = (part as { value: unknown }).value;
		if (typeof value === 'string') {
			return value;
		}
		// value could be a structured object (e.g. TSX AST) — stringify it
		if (value !== undefined && value !== null) {
			try {
				return JSON.stringify(value);
			} catch {
				return String(value);
			}
		}
	}

	return undefined;
}

function mapRole(role: vscode.LanguageModelChatMessageRole): 'system' | 'user' | 'assistant' {
	switch (role) {
		case vscode.LanguageModelChatMessageRole.User:
			return 'user';
		case vscode.LanguageModelChatMessageRole.Assistant:
			return 'assistant';
		default:
			return 'system';
	}
}

/**
 * Convert VS Code tool definitions to DeepSeek format.
 */
export function convertTools(
	tools: readonly vscode.LanguageModelChatTool[] | undefined,
): DeepSeekTool[] | undefined {
	if (!tools || tools.length === 0) {
		return undefined;
	}

	return tools.map((tool) => ({
		type: 'function' as const,
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.inputSchema as Record<string, unknown> | undefined,
		},
	}));
}

/**
 * Count total characters across all messages to calibrate chars-per-token ratio.
 */
export function countMessageChars(messages: DeepSeekMessage[]): number {
	let total = 0;
	for (const msg of messages) {
		total += msg.content?.length ?? 0;
		total += msg.reasoning_content?.length ?? 0;
		if (msg.tool_calls) {
			for (const tc of msg.tool_calls) {
				total += tc.function?.name?.length ?? 0;
				total += tc.function?.arguments?.length ?? 0;
			}
		}
	}
	return total;
}
