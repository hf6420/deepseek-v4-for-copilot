import { getStreamOptionsMode, getTemperature, getThinkingParamMode, getToolChoiceMode, getTopP } from '../config';
import { isOfficialDeepSeekBaseUrl } from '../endpoint';
import { logger } from '../logger';
import type { CompatMode, EndpointCompatibility } from '../types';

/**
 * Resolve endpoint compatibility settings by combining user-configuration
 * with endpoint-awareness (official DeepSeek vs third-party).
 *
 * ── Auto mode behavior ──
 * - Official DeepSeek (api.deepseek.com): all features enabled.
 * - Any other endpoint: DeepSeek-specific features (thinking, reasoning_effort,
 *   reasoning_content) disabled. OpenAI-common features (stream_options,
 *   tool_choice) enabled by default, with automatic fallback via learnFromError.
 *
 * Users can force-enable/disable any feature via `deepseek-copilot.compat.*`.
 */
export function resolveEndpointCompatibility(baseUrl: string): EndpointCompatibility {
	const isOfficial = isOfficialDeepSeekBaseUrl(baseUrl);

	const thinkingMode = getThinkingParamMode();
	const streamOptionsMode = getStreamOptionsMode();
	const toolChoiceMode = getToolChoiceMode();
	const temperature = getTemperature();
	const topP = getTopP();

	// thinking / reasoning_effort / reasoning_content are DeepSeek-specific.
	// They are resolved together from the same CompatMode: in practice all
	// three are tied to the DeepSeek reasoning feature, and splitting the
	// user-facing setting would add complexity without clear benefit. The
	// fields are kept separate in EndpointCompatibility so learnFromError can
	// downgrade them independently if an endpoint supports some but not all.
	const sendThinkingParam = resolveCompatMode(thinkingMode, isOfficial);
	const sendReasoningEffort = resolveCompatMode(thinkingMode, isOfficial);
	const sendReasoningContent = resolveCompatMode(thinkingMode, isOfficial);

	// stream_options is widely supported across OpenAI-compatible endpoints.
	// Keep it enabled by default; learnFromError handles the rare cases where
	// a third-party proxy rejects it.
	const sendStreamOptions = streamOptionsMode !== 'never';

	// tool_choice: "auto" is standard but some third-party proxies reject it.
	// In auto mode, only send to the official DeepSeek API. Users can
	// force-enable for third-party endpoints via 'always'.
	const sendToolChoice = toolChoiceMode === 'always' || (toolChoiceMode === 'auto' && isOfficial);

	return {
		sendThinkingParam,
		sendReasoningEffort,
		sendReasoningContent,
		sendStreamOptions,
		sendToolChoice,
		temperature: temperature > 0 ? temperature : undefined,
		topP: topP > 0 ? topP : undefined,
		providerName: isOfficial ? 'DeepSeek (Official)' : 'Third-Party',
	};
}

function resolveCompatMode(mode: CompatMode, isOfficial: boolean): boolean {
	return mode === 'always' || (mode === 'auto' && isOfficial);
}

// ---- Adaptive capability cache (error-driven learning) ----

const CACHE_DURATION_MS = 10 * 60 * 1000; // 10 minutes

/**
 * Number of consecutive successful requests required before a previously
 * downgraded capability is re-evaluated from user configuration. This lets
 * third-party endpoints that were temporarily rejecting a field (e.g. due to
 * a transient server-side issue) recover automatically without requiring a
 * VS Code restart or cache expiry.
 */
const RECOVERY_SUCCESS_THRESHOLD = 5;

interface CachedCapabilities {
	baseUrl: string;
	cachedAt: number;
	compatibility: EndpointCompatibility;
	/** Consecutive successful requests since the last learnFromError downgrade. */
	successCount: number;
}

/**
 * In-memory cache of endpoint capabilities. Evolving through learnFromError.
 */
const capabilityCache = new Map<string, CachedCapabilities>();

/**
 * Get cached capabilities for a baseUrl if not expired, or compute fresh ones.
 */
export function getEndpointCompatibility(baseUrl: string): EndpointCompatibility {
	const cached = capabilityCache.get(baseUrl);
	if (cached && Date.now() - cached.cachedAt < CACHE_DURATION_MS) {
		return cached.compatibility;
	}

	const compatibility = resolveEndpointCompatibility(baseUrl);
	capabilityCache.set(baseUrl, {
		baseUrl,
		cachedAt: Date.now(),
		compatibility,
		successCount: 0,
	});
	return compatibility;
}

/**
 * Record a successful request for an endpoint. After enough consecutive
 * successes, previously downgraded capabilities are re-evaluated from user
 * configuration, allowing recovery when a third-party endpoint starts
 * supporting a field it previously rejected.
 */
export function recordRequestSuccess(baseUrl: string): void {
	const cached = capabilityCache.get(baseUrl);
	if (!cached) {
		return;
	}

	cached.successCount += 1;
	if (cached.successCount < RECOVERY_SUCCESS_THRESHOLD) {
		return;
	}

	// Re-resolve from configuration. If the fresh resolution enables a field
	// that was previously downgraded by learnFromError, restore it.
	const fresh = resolveEndpointCompatibility(baseUrl);
	const current = cached.compatibility;
	let recovered = false;

	if (!current.sendThinkingParam && fresh.sendThinkingParam) {
		current.sendThinkingParam = true;
		recovered = true;
	}
	if (!current.sendReasoningEffort && fresh.sendReasoningEffort) {
		current.sendReasoningEffort = true;
		recovered = true;
	}
	if (!current.sendReasoningContent && fresh.sendReasoningContent) {
		current.sendReasoningContent = true;
		recovered = true;
	}
	if (!current.sendStreamOptions && fresh.sendStreamOptions) {
		current.sendStreamOptions = true;
		recovered = true;
	}
	if (!current.sendToolChoice && fresh.sendToolChoice) {
		current.sendToolChoice = true;
		recovered = true;
	}

	cached.successCount = 0;
	if (recovered) {
		logger.info(
			`[compat] Restored downgraded capabilities for ${current.providerName} (${baseUrl}) after ${RECOVERY_SUCCESS_THRESHOLD} consecutive successes`,
		);
	}
}

/**
 * Evolve adaptive capabilities when a request fails with a 400 error,
 * inferring which field caused the error from the response message.
 * Only disables non-essential fields — never escalates.
 *
 * @returns true if the compatibility was mutated (so callers can retry).
 */
export function learnFromError(baseUrl: string, errorMessage: string): boolean {
	const lowered = errorMessage.toLowerCase();
	const cached = capabilityCache.get(baseUrl);
	if (!cached) {
		return false;
	}

	let changed = false;
	const compat = { ...cached.compatibility };

	// Heuristic: if the error mentions "thinking", disable the thinking object.
	if (compat.sendThinkingParam && hasFieldError(lowered, 'thinking')) {
		compat.sendThinkingParam = false;
		changed = true;
		logger.info(
			`[compat] Disabling thinking param for ${compat.providerName} (${baseUrl}) due to API rejection`,
		);
	}

	// Heuristic: if the error mentions "reasoning_effort", disable it independently.
	if (compat.sendReasoningEffort && hasFieldError(lowered, 'reasoning_effort')) {
		compat.sendReasoningEffort = false;
		changed = true;
		logger.info(
			`[compat] Disabling reasoning_effort for ${compat.providerName} (${baseUrl}) due to API rejection`,
		);
	}

	// Heuristic: if the error mentions "reasoning_content", disable it independently.
	if (compat.sendReasoningContent && hasFieldError(lowered, 'reasoning_content')) {
		compat.sendReasoningContent = false;
		changed = true;
		logger.info(
			`[compat] Disabling reasoning_content for ${compat.providerName} (${baseUrl}) due to API rejection`,
		);
	}

	// Heuristic: if the error mentions "stream_options", disable it.
	if (compat.sendStreamOptions && hasFieldError(lowered, 'stream_options')) {
		compat.sendStreamOptions = false;
		changed = true;
		logger.info(
			`[compat] Disabling stream_options for ${compat.providerName} (${baseUrl}) due to API rejection`,
		);
	}

	// Heuristic: if the error mentions "tool_choice", disable it.
	if (compat.sendToolChoice && hasFieldError(lowered, 'tool_choice')) {
		compat.sendToolChoice = false;
		changed = true;
		logger.info(
			`[compat] Disabling tool_choice for ${compat.providerName} (${baseUrl}) due to API rejection`,
		);
	}

	// Fallback: when the error is a generic validation failure that doesn't
	// name a specific field (common in vLLM/FastAPI: "Extra inputs are not
	// permitted"), conservatively disable all DeepSeek-specific fields at
	// once. OpenAI-common fields (stream_options, tool_choice) are kept
	// because they're widely supported.
	if (!changed && isGenericFieldError(lowered)) {
		if (compat.sendThinkingParam) { compat.sendThinkingParam = false; changed = true; }
		if (compat.sendReasoningEffort) { compat.sendReasoningEffort = false; changed = true; }
		if (compat.sendReasoningContent) { compat.sendReasoningContent = false; changed = true; }
		if (changed) {
			logger.info(
				`[compat] Disabling all DeepSeek-specific fields for ${compat.providerName} (${baseUrl}) due to generic validation error`,
			);
		}
	}

	if (changed) {
		capabilityCache.set(baseUrl, {
			baseUrl,
			cachedAt: Date.now(),
			compatibility: compat,
			successCount: 0,
		});
	}

	return changed;
}

function hasFieldError(errorMessage: string, fieldName: string): boolean {
	// Match against both English and Chinese (zh-cn) error patterns so
	// third-party proxies that localise their error messages are covered.
	return (
		hasFieldErrorPattern(errorMessage) ||
		hasFieldErrorPatternChinese(errorMessage)
	) && errorMessage.includes(fieldName);
}

function hasFieldErrorPattern(message: string): boolean {
	return (
		message.includes(`unknown field`) ||
		message.includes(`unrecognized field`) ||
		message.includes(`unexpected field`) ||
		message.includes(`invalid field`) ||
		message.includes(`unknown parameter`) ||
		message.includes(`unrecognized parameter`) ||
		message.includes(`not supported`) ||
		// vLLM / FastAPI-style validation errors
		message.includes(`extra inputs are not permitted`) ||
		message.includes(`extra fields not permitted`) ||
		message.includes(`additional properties are not allowed`) ||
		message.includes(`is not permitted`) ||
		// Generic OpenAI-compatible error patterns
		message.includes(`unrecognized request argument`) ||
		message.includes(`invalid_request_error`) ||
		message.includes(`does not support`) ||
		// LiteLLM proxy error patterns
		message.includes(`is not a valid parameter`) ||
		message.includes(`bad request`) && message.includes(`param`)
	);
}

function hasFieldErrorPatternChinese(message: string): boolean {
	return (
		message.includes(`未知字段`) ||
		message.includes(`无法识别`) ||
		message.includes(`不支持的字段`) ||
		message.includes(`不支持的参数`) ||
		message.includes(`不识别的字段`) ||
		message.includes(`不识别的参数`) ||
		message.includes(`未预期的字段`) ||
		message.includes(`无效的字段`) ||
		message.includes(`无效的参数`) ||
		message.includes(`不被支持`) ||
		message.includes(`不支持`)
	);
}

/**
 * Detect generic 400/422 validation errors that don't name a specific field.
 * vLLM/FastAPI return "Extra inputs are not permitted" without nesting the
 * offending field name, making field-by-field inference impossible.
 */
function isGenericFieldError(message: string): boolean {
	return (
		message.includes(`extra inputs are not permitted`) ||
		message.includes(`extra fields not permitted`) ||
		message.includes(`additional properties are not allowed`)
	);
}
