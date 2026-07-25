import { getStreamOptionsMode, getTemperature, getThinkingParamMode, getToolChoiceMode, getTopP } from '../config';
import { isOfficialDeepSeekBaseUrl } from '../endpoint';
import type { CompatMode, EndpointCompatibility } from '../types';

/**
 * Resolve endpoint compatibility settings by combining user-configuration
 * with endpoint-awareness (official DeepSeek vs third-party).
 *
 * ── Auto mode behavior ──
 * - Official DeepSeek (api.deepseek.com): all features enabled.
 * - Any other endpoint: only OpenAI-standard parameters are sent.
 *   DeepSeek-specific fields (thinking, reasoning_effort, reasoning_content)
 *   are disabled. Use `extraBody` in model config to add custom fields.
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
	// Disabled for third-party endpoints to stay compatible with standard
	// OpenAI-compatible APIs. Users who need them on a third-party endpoint
	// can force-enable via `deepseek-copilot.compat.thinkingParam: always`.
	const sendThinkingParam = resolveCompatMode(thinkingMode, isOfficial);
	const sendReasoningEffort = resolveCompatMode(thinkingMode, isOfficial);
	const sendReasoningContent = resolveCompatMode(thinkingMode, isOfficial);

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
		providerName: isOfficial ? 'DeepSeek (Official)' : 'Third-Party',
	};
}

function resolveCompatMode(mode: CompatMode, isOfficial: boolean): boolean {
	return mode === 'always' || (mode === 'auto' && isOfficial);
}

/**
 * Get endpoint compatibility — stateless, no caching.
 * Users configure custom fields via `extraBody` in model config.
 */
export function getEndpointCompatibility(baseUrl: string): EndpointCompatibility {
	return resolveEndpointCompatibility(baseUrl);
}
