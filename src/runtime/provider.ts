import vscode from 'vscode';
import { logger } from '../logger';
import { HFChatProvider } from '../provider/hf';

export async function registerProvider(
	context: vscode.ExtensionContext,
): Promise<HFChatProvider> {
	const hfProvider = new HFChatProvider(context);

	context.subscriptions.push(
		vscode.commands.registerCommand('deepseek-copilot.setApiKey', () =>
			hfProvider.configureApiKey(),
		),
		vscode.commands.registerCommand('deepseek-copilot.clearApiKey', () =>
			hfProvider.clearApiKey(),
		),
		vscode.commands.registerCommand('deepseek-copilot.setVisionModel', () =>
			hfProvider.setVisionModel(),
		),
		vscode.lm.registerLanguageModelChatProvider('hf', hfProvider),
	);

	await activateCopilotChat();
	hfProvider.refresh();

	return hfProvider;
}

async function activateCopilotChat(): Promise<void> {
	try {
		await vscode.extensions.getExtension('github.copilot-chat')?.activate();
	} catch (error) {
		logger.warn('Copilot Chat activation unavailable; model picker refresh may be delayed', error);
	}
}
