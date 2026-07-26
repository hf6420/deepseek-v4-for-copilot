import vscode from 'vscode';
import { AuthManager } from '../auth';
import { DeepSeekClient } from '../client';
import { getApiModelId, getBaseUrl } from '../config';
import { MODELS } from '../consts';
import { t } from '../i18n';
import type { DeepSeekRequest, ModelDefinition } from '../types';
import { getEndpointCompatibility } from './compat';
import { convertMessages, countMessageChars } from './convert';
import {
    dumpDeepSeekRequest,
    type CacheDiagnosticsRecorder,
    type CacheDiagnosticsRun,
} from './debug';
import { getConfiguredThinkingEffort, type ModelConfigurationOptions } from './models';
import type { ReplayMarkerMetadata } from './replay';
import { classifyDeepSeekRequest, shouldForceThinkingNone, type RequestKind } from './routing';
import type { ConversationSegment } from './segment';
import { collectTrailingToolResultIds, prepareRequestTools } from './tools/request';
import { resolveImageMessages, type VisionDescriber } from './vision';

export interface PreparedChatRequest {
	client: DeepSeekClient;
	request: DeepSeekRequest;
	/** Extra body fields merged into the API request body (from vendor model config). */
	extraBody?: Record<string, unknown>;
	/** Whether to send `stream_options: { include_usage: true }`. Pre-computed
	 * from compat (including per-model overrides) so the client doesn't need
	 * to re-resolve compat. */
	sendStreamOptions: boolean;
	isThinkingModel: boolean;
	totalRequestChars: number;
	trailingToolResultIds: string[];
	cacheDiagnostics: CacheDiagnosticsRun;
	requestKind: RequestKind;
	segment: ConversationSegment;
	replayMarkerMetadata: ReplayMarkerMetadata;
	visionMarkerTextChars?: number;
	initialResponseNotice?: string;
}

export interface PrepareChatRequestOptions {
	authManager: AuthManager;
	globalStorageUri: vscode.Uri;
	modelInfo: vscode.LanguageModelChatInformation;
	/** Optional model definition from vendor configuration (JSON array). */
	modelDefOverride?: ModelDefinition;
	/** Per-model base URL override (from vendor config). */
	effectiveBaseUrl?: string;
	/** Per-model API key override (from vendor config). */
	effectiveApiKey?: string;
	/** Per-model extra body fields to merge into the API request (from vendor config). */
	effectiveExtraBody?: Record<string, unknown>;
	/** Per-model toolChoice override (from vendor config). Takes priority over the
	 * global `deepseek-copilot.compat.toolChoice` setting. */
	effectiveToolChoice?: boolean;
	/** Per-model compat overrides (from `models[].compat`). Takes priority over
	 * global `deepseek-copilot.compat.*` settings for this model's endpoint. */
	effectiveModelCompat?: { thinkingParam?: import('../types').CompatMode; streamOptions?: import('../types').CompatMode; toolChoice?: import('../types').CompatMode };
	segment: ConversationSegment;
	messages: readonly vscode.LanguageModelChatRequestMessage[];
	options: vscode.ProvideLanguageModelChatResponseOptions;
	token: vscode.CancellationToken;
	cacheDiagnostics: CacheDiagnosticsRecorder;
	getVisionDescriber: () => Promise<VisionDescriber | undefined>;
}

export async function prepareChatRequest({
	authManager,
	globalStorageUri,
	modelInfo,
	modelDefOverride,
	effectiveBaseUrl,
	effectiveApiKey,
	effectiveExtraBody,
	effectiveToolChoice,
	effectiveModelCompat,
	segment,
	messages,
	options,
	token,
	cacheDiagnostics,
	getVisionDescriber,
}: PrepareChatRequestOptions): Promise<PreparedChatRequest> {
	// Resolve effective API key, base URL, and model ID.
	// Per-model config takes priority, falling back to global settings.
	const resolvedApiKey = effectiveApiKey || await authManager.getApiKey();
	if (!resolvedApiKey) {
		throw new Error(t('auth.notConfigured'));
	}

	const resolvedBaseUrl = effectiveBaseUrl || getBaseUrl();
	const resolvedApiModelId = getApiModelId(modelInfo.id);

	const client = new DeepSeekClient(resolvedBaseUrl, resolvedApiKey);

	// Resolve model definition: vendor config → built-in
	const modelDef: ModelDefinition | undefined =
		modelDefOverride ?? MODELS.find((m) => m.id === modelInfo.id);

	const isThinkingModel = modelDef?.capabilities.thinking ?? false;
	const supportsVision = modelDef?.capabilities.imageInput ?? false;
	const maxTokens = (modelDef?.max_tokens && modelDef.max_tokens > 0)
		? modelDef.max_tokens
		: undefined;

	const visionResolution = supportsVision
		? await resolveImageMessages(messages, token, getVisionDescriber)
		: { messages, replayMarkerMetadata: {}, stats: { inputImageParts: 0, inputImageMessages: 0, currentImageMessages: 0, generatedImageMessages: 0, replayedImageMessages: 0, omittedImageMessages: 0, unavailableImageMessages: 0, failedImageMessages: 0, droppedImageParts: 0, markerVisionTextChars: 0, invalidMarkerVisionMetadata: 0 } };
	const resolvedMessages = visionResolution.messages;
	const deepseekMessages = convertMessages(resolvedMessages, isThinkingModel, resolvedBaseUrl, effectiveModelCompat);
	const tools = prepareRequestTools(modelDef?.capabilities.toolCalling, options);

	const compat = getEndpointCompatibility(resolvedBaseUrl, effectiveModelCompat);
	const totalRequestChars = countMessageChars(deepseekMessages);
	const sendToolChoice = effectiveToolChoice ?? compat.sendToolChoice;
	const baseRequest: DeepSeekRequest = {
		model: resolvedApiModelId,
		messages: deepseekMessages,
		stream: true,
		tools,
		tool_choice: sendToolChoice && tools && tools.length > 0 ? ('auto' as const) : undefined,
		max_tokens: maxTokens,
		...(compat.temperature ? { temperature: compat.temperature } : {}),
		...(compat.topP ? { top_p: compat.topP } : {}),
	};
	const requestKind = classifyDeepSeekRequest({
		request: baseRequest,
		inputMessages: messages,
	});
	const configuredThinkingEffort = getConfiguredThinkingEffort(
		options as ModelConfigurationOptions,
	);
	// Force helper requests (non-main-agent) into disabled thinking for
	// all endpoints so simple queries skip the reasoning overhead.
	const forceNoneThinking = shouldForceThinkingNone(requestKind);
	const thinkingEffort = forceNoneThinking ? 'none' : configuredThinkingEffort;
	const request: DeepSeekRequest = {
		...baseRequest,
		...(compat.sendThinkingParam && isThinkingModel
			? {
					thinking: {
						type: thinkingEffort === 'none' ? ('disabled' as const) : ('enabled' as const),
					},
					...(compat.sendReasoningEffort && thinkingEffort !== 'none'
						? { reasoning_effort: thinkingEffort }
						: {}),
				}
			: {}),
	};
	dumpDeepSeekRequest(request, {
		globalStorageUri,
		segment,
		requestKind,
		vscodeModelId: modelInfo.id,
		isThinkingModel,
		thinkingEffort,
		maxTokens,
		inputMessages: messages,
		resolvedMessages,
		requestOptions: options,
		visionModelId: visionResolution.visionModelId,
		visionProxySource: visionResolution.visionProxySource,
		visionStats: visionResolution.stats,
	});

	const diagnosticsRun = cacheDiagnostics.beginRequest({
		request,
		segment,
		requestKind,
		vscodeModelId: modelInfo.id,
		isThinkingModel,
		thinkingEffort,
		maxTokens,
		inputMessages: messages,
		resolvedMessages,
		visionModelId: visionResolution.visionModelId,
		visionProxySource: visionResolution.visionProxySource,
		visionStats: visionResolution.stats,
	});

	return {
		client,
		request,
		extraBody: effectiveExtraBody,
		sendStreamOptions: compat.sendStreamOptions,
		isThinkingModel,
		totalRequestChars,
		trailingToolResultIds: collectTrailingToolResultIds(deepseekMessages),
		cacheDiagnostics: diagnosticsRun,
		requestKind,
		segment,
		replayMarkerMetadata: visionResolution.replayMarkerMetadata,
		visionMarkerTextChars: visionResolution.stats.markerVisionTextChars || undefined,
		initialResponseNotice: visionResolution.initialResponseNotice,
	};
}
