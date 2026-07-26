import vscode from 'vscode';
import { AuthManager } from '../auth';
import { t } from '../i18n';
import type { CompatMode, ModelDefinition } from '../types';
import { DeepSeekChatProvider } from './index';
import { toChatInfo } from './models';

interface VendorModelConfig {
	id: string;
	name: string;
	detail?: string;
	url?: string;
	apiKey?: string;
	maxInputTokens?: number;
	maxOutputTokens?: number;
	toolCalling?: boolean;
	vision?: boolean;
	thinking?: boolean;
	extraBody?: Record<string, unknown>;
	toolChoice?: boolean;
	/** Per-model compat overrides. Takes priority over global `deepseek-copilot.compat.*` settings. */
	compat?: {
		thinkingParam?: import('../types').CompatMode;
		streamOptions?: import('../types').CompatMode;
		toolChoice?: import('../types').CompatMode;
	};
}

/**
 * HF Chat Provider — the single model provider registered as vendor "hf".
 * Models are defined through VS Code's configuration JSON array (`models`).
 */
export class HFChatProvider implements vscode.LanguageModelChatProvider {
	private readonly authManager: AuthManager;
	private readonly engine: DeepSeekChatProvider;
	private readonly onDidChangeEmitter = new vscode.EventEmitter<void>();
	private isActive = true;

	readonly onDidChangeLanguageModelChatInformation = this.onDidChangeEmitter.event;

	constructor(context: vscode.ExtensionContext) {
		this.authManager = new AuthManager(context);
		// Share the AuthManager with the engine to avoid duplicate
		// SecretStorage listeners and cache invalidation.
		this.engine = new DeepSeekChatProvider(context, this.authManager);

		context.subscriptions.push(
			this.onDidChangeEmitter,
			vscode.workspace.onDidChangeConfiguration((e) => {
				if (
					e.affectsConfiguration('deepseek-copilot.apiKey') ||
					e.affectsConfiguration('deepseek-copilot.baseUrl')
				) {
					if (e.affectsConfiguration('deepseek-copilot.apiKey')) {
						const config = vscode.workspace.getConfiguration('deepseek-copilot');
						if (config.get<string>('apiKey')?.trim()) {
							void this.authManager.deleteApiKey();
						}
					}
					this.refresh();
				}
			}),
			context.secrets.onDidChange((e) => {
				if (e.key === 'deepseek-copilot.apiKey') {
					this.refresh();
				}
			}),
		);
	}

	// ---- Public lifecycle ----

	refresh(): void {
		this.onDidChangeEmitter.fire();
	}

	async prepareForDeactivate(): Promise<void> {
		this.isActive = false;
		this.onDidChangeEmitter.fire();
		await this.engine.prepareForDeactivate();
	}

	// ---- Commands ----

	async configureApiKey(): Promise<void> {
		const saved = await this.authManager.promptForApiKey();
		if (saved) { this.refresh(); }
	}

	async clearApiKey(): Promise<void> {
		await this.authManager.deleteApiKey();
		this.refresh();
		vscode.window.showInformationMessage(t('auth.removed'));
	}

	async setVisionModel(): Promise<void> {
		await this.engine.setVisionModel();
	}

	async hasApiKey(): Promise<boolean> {
		return this.authManager.hasApiKey();
	}

	// ---- LanguageModelChatProvider ----

	async provideTokenCount(
		model: vscode.LanguageModelChatInformation,
		text: string | vscode.LanguageModelChatRequestMessage,
		token: vscode.CancellationToken,
	): Promise<number> {
		return this.engine.provideTokenCount(model, text, token);
	}

	async provideLanguageModelChatInformation(
		_options: vscode.PrepareLanguageModelChatModelOptions,
		_token: vscode.CancellationToken,
	): Promise<vscode.LanguageModelChatInformation[]> {
		if (!this.isActive) { return []; }

		const opts = _options as vscode.PrepareLanguageModelChatModelOptions & {
			configuration?: Record<string, unknown>;
		};
		const cfg = opts.configuration;

		// Persist the FIRST configured API key and base URL as global
		// defaults. Per-model configs take priority at request time.
		const configuredApiKey = typeof cfg?.apiKey === 'string' ? (cfg.apiKey as string).trim() : '';
		const vendorBaseUrl = typeof cfg?.baseUrl === 'string' ? (cfg.baseUrl as string).trim() : '';

		// Global defaults: only set when not already configured.
		if (configuredApiKey) {
			try {
				const hasExisting = await this.authManager.hasApiKey();
				if (!hasExisting) { await this.authManager.setApiKey(configuredApiKey); }
			} catch { /* ok */ }
		}
		if (vendorBaseUrl) {
			try {
				const dsCfg = vscode.workspace.getConfiguration('deepseek-copilot');
				const existing = dsCfg.get<string>('baseUrl');
				if (!existing || existing === 'https://api.deepseek.com') {
					await dsCfg.update('baseUrl', vendorBaseUrl, vscode.ConfigurationTarget.Global);
				}
			} catch { /* ok */ }
		}

		// Parse `models` array from configuration.
		const rawModels: unknown[] = Array.isArray(cfg?.models) ? (cfg!.models as unknown[]) : [];
		const configs: VendorModelConfig[] = [];
		for (const item of rawModels) {
			if (item && typeof item === 'object') {
				const m = item as Record<string, unknown>;
				if (typeof m.id === 'string' && typeof m.name === 'string') {
					configs.push({
						id: m.id as string,
						name: m.name as string,
						detail: typeof m.detail === 'string' ? m.detail as string : undefined,
						// per-model url wins; fall back to vendor-level baseUrl.
						url: typeof m.url === 'string' ? m.url as string : (vendorBaseUrl || undefined),
						// per-model apiKey wins; fall back to vendor-level apiKey.
						apiKey: typeof m.apiKey === 'string' ? m.apiKey as string : (configuredApiKey || undefined),
						maxInputTokens: typeof m.maxInputTokens === 'number' ? m.maxInputTokens as number : undefined,
						maxOutputTokens: typeof m.maxOutputTokens === 'number' ? m.maxOutputTokens as number : undefined,
						toolCalling: typeof m.toolCalling === 'boolean' ? m.toolCalling as boolean : undefined,
						vision: typeof m.vision === 'boolean' ? m.vision as boolean : undefined,
						thinking: typeof m.thinking === 'boolean' ? m.thinking as boolean : undefined,

						extraBody: typeof m.extraBody === 'object' && m.extraBody !== null && !Array.isArray(m.extraBody) ? m.extraBody as Record<string, unknown> : undefined,
						toolChoice: typeof m.toolChoice === 'boolean' ? m.toolChoice as boolean : undefined,
						compat: parseModelCompat(m.compat),
					});
				}
			}
		}

		const hasGlobalKey = await this.authManager.hasApiKey();
		const hasAnyModelKey = configs.some((c) => c.apiKey?.trim());
		const hasKey = hasGlobalKey || hasAnyModelKey;

		// --- Attach each vendor config directly to the returned model info ---
		// Previously a shared Map<string, VendorModelConfig> was used, but
		// when multiple groups define the same model ID the last write wins,
		// causing a model selected in group A to silently route through
		// group B's URL / API key / extraBody.  Storing the config on the
		// object itself guarantees that request-time lookup returns the
		// exact config that was registered for *this* model – regardless of
		// how many groups share the same ID.
		const pricingCurrency = this.engine.getBalanceCurrency();
		return configs.map((c) => {
			const def = this.toDefinition(c);
			const info = toChatInfo(def, hasKey, pricingCurrency);
			(info as unknown as Record<string, unknown>).__vendorConfig = c;
			return info;
		});
	}

	async provideLanguageModelChatResponse(
		modelInfo: vscode.LanguageModelChatInformation,
		messages: readonly vscode.LanguageModelChatRequestMessage[],
		options: vscode.ProvideLanguageModelChatResponseOptions,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
	): Promise<void> {
		// Read the vendor config that was attached during model listing.
		// Using the object itself avoids cross-group collisions when
		// multiple groups share the same model ID.
		const cfg = (modelInfo as unknown as Record<string, unknown>).__vendorConfig as VendorModelConfig | undefined;
		const modelDef = cfg ? this.toDefinition(cfg) : undefined;

		return this.engine.provideLanguageModelChatResponseWithDef(
			modelInfo, messages, options, progress, token,
			modelDef,
			cfg?.url?.trim() || undefined,
			cfg?.apiKey?.trim() || undefined,
			cfg?.extraBody,
			cfg?.toolChoice,
			cfg?.compat,
		);
	}

	// ---- Helpers ----

	private toDefinition(c: VendorModelConfig): ModelDefinition {
		return {
			id: c.id,
			name: c.name,
			family: 'hf',
			version: '',
			detail: c.detail ?? '',
			maxInputTokens: c.maxInputTokens ?? 131072,
			maxOutputTokens: c.maxOutputTokens ?? 16384,
			capabilities: {
				toolCalling: c.toolCalling ?? true,
				imageInput: c.vision ?? false,
				thinking: c.thinking ?? false,
			},
		};
	}
}

/**
 * Parse per-model compat override from raw JSON config.
 * Only accepts valid CompatMode values; ignores invalid entries.
 */
function parseModelCompat(raw: unknown): VendorModelConfig['compat'] {
	if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
		return undefined;
	}
	const obj = raw as Record<string, unknown>;
	const result: VendorModelConfig['compat'] = {};

	const thinkingParam = obj.thinkingParam;
	if (thinkingParam === 'auto' || thinkingParam === 'always' || thinkingParam === 'never') {
		result.thinkingParam = thinkingParam as CompatMode;
	}
	const streamOptions = obj.streamOptions;
	if (streamOptions === 'auto' || streamOptions === 'always' || streamOptions === 'never') {
		result.streamOptions = streamOptions as CompatMode;
	}
	const toolChoice = obj.toolChoice;
	if (toolChoice === 'auto' || toolChoice === 'always' || toolChoice === 'never') {
		result.toolChoice = toolChoice as CompatMode;
	}

	return Object.keys(result).length > 0 ? result : undefined;
}
