import vscode from 'vscode';
import { AuthManager } from '../auth';
import { getStabilizeToolListEnabled } from '../config';
import { DEFAULT_CHARS_PER_TOKEN, MODELS } from '../consts';
import { t } from '../i18n';
import { logger } from '../logger';
import { createCacheDiagnosticsRecorder, dumpProviderInput } from './debug';
import { toChatInfo } from './models';
import { BalanceCurrencyResolver } from './pricing/currency';
import { prepareChatRequest } from './request';
import { classifyProviderRequest } from './routing';
import { resolveConversationSegment } from './segment';
import { streamChatCompletion } from './stream';
import { estimateTokenCount } from './tokens';
import { processToolFlow } from './tools/flow';
import { createVisionService } from './vision';

/**
 * DeepSeek Chat Provider — internal engine that implements the full chat
 * pipeline (vision proxy, tool flow, streaming). It is NOT directly registered
 * as a vendor; the HFChatProvider (hf.ts) wraps it as the public API surface.
 */
export class DeepSeekChatProvider implements vscode.LanguageModelChatProvider {
	private readonly authManager: AuthManager;
	private readonly context: vscode.ExtensionContext;
	private readonly globalStorageUri: vscode.Uri;
	private readonly onDidChangeLanguageModelChatInformationEmitter = new vscode.EventEmitter<void>();
	private isActive = true;

	readonly onDidChangeLanguageModelChatInformation =
		this.onDidChangeLanguageModelChatInformationEmitter.event;

	private readonly cacheDiagnostics = createCacheDiagnosticsRecorder();

	/**
	 * Shared chars-per-token ratio calibrated across requests via EMA.
	 * Seeded from the default and updated after each streaming response
	 * receives actual usage data. Used by both provideTokenCount and
	 * streamChatCompletion so token estimates stay accurate for CJK text
	 * where the default 4.0 chars/token ratio is a significant overestimate.
	 */
	private charsPerToken = DEFAULT_CHARS_PER_TOKEN;

	/** Vision proxy: internal bridge + VS Code LM fallback. */
	private readonly vision: ReturnType<typeof createVisionService>;
	private readonly balanceCurrencyResolver: BalanceCurrencyResolver;

	/**
	 * @param context VS Code extension context.
	 * @param sharedAuthManager Optional shared AuthManager from the wrapping
	 *   HFChatProvider. When provided, avoids duplicate SecretStorage
	 *   listeners and cache instances.
	 */
	constructor(context: vscode.ExtensionContext, sharedAuthManager?: AuthManager) {
		this.context = context;
		this.authManager = sharedAuthManager ?? new AuthManager(context);
		this.globalStorageUri = context.globalStorageUri;
		this.vision = createVisionService(context);
		this.balanceCurrencyResolver = new BalanceCurrencyResolver(context, this.authManager, () =>
			this.onDidChangeLanguageModelChatInformationEmitter.fire(),
		);

		context.subscriptions.push(
			this.onDidChangeLanguageModelChatInformationEmitter,
			// Settings-based fallback API key + base URL changes.
			vscode.workspace.onDidChangeConfiguration((e) => {
				if (e.affectsConfiguration('deepseek-copilot.apiKey')) {
					// When the user sets an API key in Settings UI, clear
					// SecretStorage so the settings value takes effect.
					const config = vscode.workspace.getConfiguration('deepseek-copilot');
					if (config.get<string>('apiKey')?.trim()) {
						void this.authManager.deleteApiKey();
					}
					this.invalidateCurrencyAndRefreshModels();
				}
				if (e.affectsConfiguration('deepseek-copilot.baseUrl')) {
					this.invalidateCurrencyAndRefreshModels();
				}
			}),
		);

		// Multi-window: SecretStorage changes don't fire onDidChangeConfiguration.
		// When another window sets/clears the API key, refresh this window's
		// model picker and currency cache so the warning/price state stays in sync.
		// Always registered — HFChatProvider only handles model picker refresh,
		// not currency invalidation.
		context.subscriptions.push(
			context.secrets.onDidChange((e) => {
				if (e.key === 'deepseek-copilot.apiKey') {
					this.invalidateCurrencyAndRefreshModels();
				}
			}),
		);
	}

	// ---- Public commands ----

	async configureApiKey(): Promise<void> {
		const saved = await this.authManager.promptForApiKey();
		if (saved) {
			this.invalidateCurrencyAndRefreshModels();
		}
	}

	async clearApiKey(): Promise<void> {
		await this.authManager.deleteApiKey();
		this.invalidateCurrencyAndRefreshModels();
		vscode.window.showInformationMessage(t('auth.removed'));
	}

	async hasApiKey(): Promise<boolean> {
		return this.authManager.hasApiKey();
	}

	/** Force Copilot Chat to re-query model information (including configurationSchema). */
	refreshModelPicker(): void {
		this.onDidChangeLanguageModelChatInformationEmitter.fire();
	}

	private invalidateCurrencyAndRefreshModels(): void {
		void this.balanceCurrencyResolver
			.invalidate()
			.catch((error) => logger.warn('Failed to invalidate DeepSeek balance currency', error))
			.finally(() => {
				this.onDidChangeLanguageModelChatInformationEmitter.fire();
			});
	}

	async prepareForDeactivate(): Promise<void> {
		this.isActive = false;
	}

	async setVisionModel(): Promise<void> {
		await this.vision.openConfiguration();
	}

	// ---- LanguageModelChatProvider ----

	async provideLanguageModelChatInformation(
		_options: vscode.PrepareLanguageModelChatModelOptions,
		_token: vscode.CancellationToken,
	): Promise<vscode.LanguageModelChatInformation[]> {
		if (!this.isActive) {
			return [];
		}

		const hasKey = await this.authManager.hasApiKey();
		const pricingCurrency = this.balanceCurrencyResolver.getDisplayCurrency();
		if (hasKey) {
			this.balanceCurrencyResolver.refreshInBackground();
		}

		// This method is retained for backward compatibility but is
		// no longer called directly — the HF provider (hf.ts) handles
		// model listing. It still serves as an internal fallback.
		return MODELS.map((model) => toChatInfo(model, hasKey, pricingCurrency));
	}

	/** Expose balance currency for the HF provider's pricing display. */
	getBalanceCurrency(): import('../types').PricingCurrency | undefined {
		return this.balanceCurrencyResolver.getDisplayCurrency();
	}

	async provideLanguageModelChatResponse(
		modelInfo: vscode.LanguageModelChatInformation,
		messages: readonly vscode.LanguageModelChatRequestMessage[],
		options: vscode.ProvideLanguageModelChatResponseOptions,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
	): Promise<void> {
		return this.provideLanguageModelChatResponseWithDef(
			modelInfo, messages, options, progress, token, undefined,
		);
	}

	async provideLanguageModelChatResponseWithDef(
		modelInfo: vscode.LanguageModelChatInformation,
		messages: readonly vscode.LanguageModelChatRequestMessage[],
		options: vscode.ProvideLanguageModelChatResponseOptions,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
		modelDefOverride?: import('../types').ModelDefinition,
		effectiveBaseUrl?: string,
		effectiveApiKey?: string,
	): Promise<void> {
		const segment = resolveConversationSegment(messages);
		const requestKind = classifyProviderRequest({
			messages,
			tools: options.tools,
		});

		dumpProviderInput({
			globalStorageUri: this.globalStorageUri,
			segment,
			modelInfo,
			messages,
			requestOptions: options,
			requestKind,
		});

		const toolFlow = processToolFlow({
			stabilizeToolList: getStabilizeToolListEnabled(),
			messages,
			tools: options.tools,
			progress,
			requestKind,
		});
		if (toolFlow.preflightHandled) {
			return;
		}

		const prepared = await prepareChatRequest({
			authManager: this.authManager,
			globalStorageUri: this.globalStorageUri,
			modelInfo,
			modelDefOverride,
			effectiveBaseUrl,
			effectiveApiKey,
			segment,
			messages: toolFlow.messages,
			options,
			token,
			cacheDiagnostics: this.cacheDiagnostics,
			getVisionDescriber: () => this.vision.get(),
		});

		// Seed from the shared calibrated value so EMA learning persists
		// across requests. The per-request closure still isolates concurrent
		// requests from each other's intermediate updates.
		const charsPerToken = { value: this.charsPerToken };
		return streamChatCompletion({
			prepared,
			progress,
			token,
			initialResponseNotice: joinInitialResponseNotices(
				toolFlow.initialResponseNotice,
				prepared.initialResponseNotice,
			),
			getCharsPerToken: () => charsPerToken.value,
			setCharsPerToken: (value) => {
				charsPerToken.value = value;
				this.charsPerToken = value;
			},
		});
	}

	async provideTokenCount(
		_modelInfo: vscode.LanguageModelChatInformation,
		text: string | vscode.LanguageModelChatRequestMessage,
		_token: vscode.CancellationToken,
	): Promise<number> {
		return estimateTokenCount(text, this.charsPerToken);
	}
}

function joinInitialResponseNotices(...notices: (string | undefined)[]): string | undefined {
	const joined = notices.filter((notice) => notice && notice.trim().length > 0).join('\n');
	return joined || undefined;
}
