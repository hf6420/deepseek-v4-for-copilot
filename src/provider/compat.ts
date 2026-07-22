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
	const sendThinkingParam = resolveCompatMode(thinkingMode, isOfficial);
	const sendReasoningContent = resolveCompatMode(thinkingMode, isOfficial);

	// stream_options & tool_choice are OpenAI-common, widely supported.
	const sendStreamOptions = streamOptionsMode !== 'never';
	const sendToolChoice = toolChoiceMode !== 'never';

	return {
		sendThinkingParam,
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

interface CachedCapabilities {
	baseUrl: string;
	cachedAt: number;
	compatibility: EndpointCompatibility;
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
	});
	return compatibility;
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

	// Heuristic: if the error mentions "thinking", disable DeepSeek-specific fields.
	if (compat.sendThinkingParam && hasFieldError(lowered, 'thinking')) {
		compat.sendThinkingParam = false;
		compat.sendReasoningContent = false;
		changed = true;
		logger.info(
			`[compat] Disabling thinking param for ${compat.providerName} (${baseUrl}) due to API rejection`,
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

	if (changed) {
		capabilityCache.set(baseUrl, {
			baseUrl,
			cachedAt: Date.now(),
			compatibility: compat,
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
		message.includes(`not supported`)
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
