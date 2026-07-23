/**
 * Shared types for the DeepSeek Copilot extension.
 */

// ---- API request/response types ----

export interface DeepSeekMessage {
	role: 'system' | 'user' | 'assistant' | 'tool';
	content: string | null;
	tool_call_id?: string;
	tool_calls?: DeepSeekToolCall[];
	reasoning_content?: string;
}

export interface DeepSeekToolCall {
	id: string;
	type: 'function';
	function: {
		name: string;
		arguments: string;
	};
}

export interface DeepSeekTool {
	type: 'function';
	function: {
		name: string;
		description?: string;
		parameters?: Record<string, unknown>;
	};
}

export interface DeepSeekUsage {
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
	prompt_cache_hit_tokens?: number;
	prompt_cache_miss_tokens?: number;
}

export interface DeepSeekRequest {
	model: string;
	messages: DeepSeekMessage[];
	stream: boolean;
	temperature?: number;
	top_p?: number;
	max_tokens?: number;
	tools?: DeepSeekTool[];
	tool_choice?: 'none' | 'auto' | 'required';
	thinking?: { type: 'enabled' | 'disabled' };
	reasoning_effort?: 'high' | 'max';

}

export interface DeepSeekStreamChunk {
	id: string;
	object: string;
	created: number;
	model: string;
	choices: Array<{
		index: number;
		delta: {
			role?: string;
			content?: string;
			reasoning_content?: string;
			tool_calls?: Array<{
				index: number;
				id?: string;
				type?: string;
				function?: {
					name?: string;
					arguments?: string;
				};
			}>;
		};
		finish_reason: string | null;
	}>;
	usage?: DeepSeekUsage;
}

// ---- Stream callbacks ----

export interface StreamCallbacks {
	onContent: (content: string) => void;
	onThinking: (text: string) => void;
	onToolCall: (toolCall: DeepSeekToolCall) => void;
	onError: (error: Error) => void;
	onDone: () => void;
	onUsage?: (usage: DeepSeekUsage) => void;
}

// ---- Model definitions ----

export type PricingCurrency = 'USD' | 'CNY';

export type PriceCategory = 'low' | 'medium' | 'high' | 'very_high';

export interface ModelPricing {
	cacheHitInput: number;
	cacheMissInput: number;
	output: number;
}

export interface ModelDefinition {
	id: string;
	name: string;
	family: string;
	version: string;
	detail: string;
	maxInputTokens: number;
	maxOutputTokens: number;
	capabilities: {
		toolCalling: boolean | number;
		imageInput: boolean;
		thinking: boolean;
	};
	requiresThinkingParam: boolean;
	pricing?: Readonly<Record<PricingCurrency, ModelPricing>>;
	priceCategory?: PriceCategory;
}

// ---- Endpoint compatibility ----

/**
 * User-controlled compatibility mode for individual features.
 * - 'auto': enable for official DeepSeek API, disable for third-party endpoints.
 * - 'always': force-enable regardless of endpoint.
 * - 'never': force-disable regardless of endpoint.
 */
export type CompatMode = 'auto' | 'always' | 'never';

/**
 * Resolved compatibility flags for a specific endpoint.
 * These are computed once and cached; consumed by request builders.
 */
export interface EndpointCompatibility {
	/** Whether to send `thinking` / `reasoning_effort` params (DeepSeek-specific). */
	sendThinkingParam: boolean;
	/** Whether to inject `reasoning_content` into assistant messages (DeepSeek-specific). */
	sendReasoningContent: boolean;
	/** Whether to send `stream_options: { include_usage: true }` (OpenAI extension). */
	sendStreamOptions: boolean;
	/** Whether to send `tool_choice: "auto"` (OpenAI standard). */
	sendToolChoice: boolean;
	/** Temperature parameter (0 = unset). */
	temperature: number | undefined;
	/** Top-p parameter (0 = unset). */
	topP: number | undefined;
	/** Human-readable provider name for diagnostics. */
	providerName: string;
}
