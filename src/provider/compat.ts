import { getStreamOptionsMode, getTemperature, getThinkingParamMode, getToolChoiceMode, getTopP } from '../config';
import type { CompatMode, EndpointCompatibility } from '../types';

/**
 * Resolve endpoint compatibility settings purely from user configuration.
 *
 * ── Mode behavior ──
 * - `auto` (default): only OpenAI-standard parameters are sent.
 *   DeepSeek-specific fields (thinking, reasoning_effort, reasoning_content)
 *   are disabled — safe for any OpenAI-compatible endpoint.
 * - `always`: DeepSeek-specific fields are enabled. Use for endpoints that
 *   support the full DeepSeek API surface.
 * - `never`: feature is disabled regardless of other settings.
 *
 * Disable/enable features via `deepseek-copilot.compat.*` globally
 * or per-model via `models[].compat`.
 */
export function resolveEndpointCompatibility(
	baseUrl: string,
	modelCompat?: { thinkingParam?: CompatMode; streamOptions?: CompatMode; toolChoice?: CompatMode },
): EndpointCompatibility {
	const thinkingMode = modelCompat?.thinkingParam ?? getThinkingParamMode();
	const streamOptionsMode = modelCompat?.streamOptions ?? getStreamOptionsMode();
	const toolChoiceMode = modelCompat?.toolChoice ?? getToolChoiceMode();
	const temperature = getTemperature();
	const topP = getTopP();

	// thinking / reasoning_effort / reasoning_content are DeepSeek-specific.
	// Disabled by default (auto = off). Enable explicitly via compat settings
	// for endpoints that support the full DeepSeek API surface.
	const sendThinkingParam = resolveCompatMode(thinkingMode);
	const sendReasoningEffort = resolveCompatMode(thinkingMode);
	const sendReasoningContent = resolveCompatMode(thinkingMode);

	// stream_options is standard OpenAI (include_usage in SSE stream).
	const sendStreamOptions = streamOptionsMode !== 'never';

	// tool_choice: "auto" is standard OpenAI. Enabled by default for all
	// endpoints; users can disable via `deepseek-copilot.compat.toolChoice: never`.
	const sendToolChoice = toolChoiceMode !== 'never';

	return {
		sendThinkingParam,
		sendReasoningEffort,
		sendReasoningContent,
		sendStreamOptions,
		sendToolChoice,
		temperature: temperature > 0 ? temperature : undefined,
		topP: topP > 0 ? topP : undefined,
		providerName: 'DeepSeek Compatible',
	};
}

function resolveCompatMode(mode: CompatMode): boolean {
	return mode === 'always';
}

/**
 * Get endpoint compatibility — stateless, no caching.
 * Users configure custom fields via `extraBody` in model config.
 * Per-model overrides (from `models[].compat`) take priority over global settings.
 */
export function getEndpointCompatibility(
	baseUrl: string,
	modelCompat?: { thinkingParam?: CompatMode; streamOptions?: CompatMode; toolChoice?: CompatMode },
): EndpointCompatibility {
	return resolveEndpointCompatibility(baseUrl, modelCompat);
}
