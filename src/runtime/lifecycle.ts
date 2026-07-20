import vscode from 'vscode';
import { t } from '../i18n';
import { logger } from '../logger';
import { DeepSeekChatProvider } from '../provider';
import { registerActionUrls } from './actions';
import { registerCommands } from './commands';
import { initializeDiagnostics } from './diagnostics';
import { registerProvider } from './provider';
import { showWelcomeIfNeeded } from './welcome';

let activeProvider: DeepSeekChatProvider | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
	// Enable Insiders-level features (e.g. proposed APIs) by spoofing the
	// product quality context key. This is safe — it only affects when-clause
	// evaluation within this window and has no side effects on actual product
	// quality or updates.
	//
	// `setContext` is an internal VS Code command (not public API). If VS Code
	// ever removes or renames it, fail gracefully so the extension can still
	// activate — thinking parts may degrade to inline text, but chat still works.
	try {
		await vscode.commands.executeCommand('setContext', 'productQualityType', 'insiders');
	} catch {
		logger.debug('setContext command unavailable — proposed APIs may not be active');
	}

	await initializeDiagnostics(context);
	registerCommands(context);
	registerActionUrls(context);

	try {
		const provider = await registerProvider(context);
		activeProvider = provider;

		void showWelcomeIfNeeded(context, provider).catch((error) => {
			logger.warn(t('extension.welcomeFailed'), error);
		});

		logger.info(`Extension activated version=${context.extension.packageJSON.version}`);
	} catch (error) {
		activeProvider = undefined;
		logger.error('Failed to activate DeepSeek extension', error);
		void vscode.window.showErrorMessage(t('extension.activateFailed'));
		throw error;
	}
}

export async function deactivate(): Promise<void> {
	try {
		await activeProvider?.prepareForDeactivate();
	} catch (error) {
		logger.warn(t('extension.deactivateFailed'), error);
	} finally {
		activeProvider = undefined;
		logger.info('Extension deactivated');
		logger.dispose();
	}
}
