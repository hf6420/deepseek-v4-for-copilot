import vscode from 'vscode';
import { AuthManager } from '../auth';
import { t } from '../i18n';
import type { ModelDefinition } from '../types';
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
	requiresThinkingParam?: boolean;
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

	/** Cache of vendor model configs keyed by model ID. */
	private vendorConfigs = new Map<string, VendorModelConfig>();

	readonly onDidChangeLanguageModelChatInformation = this.onDidChangeEmitter.event;

	constructor(context: vscode.ExtensionContext) {
		this.authManager = new AuthManager(context);
		this.engine = new DeepSeekChatProvider(context);

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
						requiresThinkingParam: typeof m.requiresThinkingParam === 'boolean' ? m.requiresThinkingParam as boolean : undefined,
					});
				}
			}
		}

		const hasGlobalKey = await this.authManager.hasApiKey();
		const hasAnyModelKey = configs.some((c) => c.apiKey?.trim());
		const hasKey = hasGlobalKey || hasAnyModelKey;

		// Build ModelDefinitions and merge into cache for request-time lookup.
		// Never clear — each group contributes its own models.
		const defs: ModelDefinition[] = [];
		for (const c of configs) {
			const def = this.toDefinition(c);
			defs.push(def);
			this.vendorConfigs.set(c.id, c);
		}

		const pricingCurrency = this.engine.getBalanceCurrency();
		return defs.map((def) => toChatInfo(def, hasKey, pricingCurrency));
	}

	async provideLanguageModelChatResponse(
		modelInfo: vscode.LanguageModelChatInformation,
		messages: readonly vscode.LanguageModelChatRequestMessage[],
		options: vscode.ProvideLanguageModelChatResponseOptions,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
	): Promise<void> {
		// Look up the vendor config to pass per-model URL / API key overrides.
		const cfg = this.vendorConfigs.get(modelInfo.id);
		const modelDef = cfg ? this.toDefinition(cfg) : undefined;

		return this.engine.provideLanguageModelChatResponseWithDef(
			modelInfo, messages, options, progress, token,
			modelDef,
			cfg?.url?.trim() || undefined,
			cfg?.apiKey?.trim() || undefined,
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
			requiresThinkingParam: c.requiresThinkingParam ?? false,
		};
	}
}

