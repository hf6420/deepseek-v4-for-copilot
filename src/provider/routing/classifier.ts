import vscode from 'vscode';
import { logger } from '../../logger';
import type { DeepSeekRequest, DeepSeekTool } from '../../types';

export type RequestKind =
	| 'main-agent'
	| 'terminal-steering'
	| 'todo-tracker'
	| 'settings-resolver'
	| 'prompt-categorizer'
	| 'chat-title'
	| 'inline-progress-message'
	| 'git-branch-name'
	| 'git-commit-message'
	| 'rename-suggestions'
	| 'background'
	| 'unknown';

// ── Prefix-based classifiers (brittle — Copilot may reword these at any time) ──

const TODO_TRACKER_PREFIX = 'You are a background task tracker';
const PROMPT_CATEGORIZER_PREFIX = 'You are an expert classifier for AI coding assistant prompts';
const SETTINGS_RESOLVER_PREFIX =
	'You are a Visual Studio Code assistant. Your job is to assist users in using Visual Studio Code by returning settings';
const CHAT_TITLE_PREFIXES = [
	'You are an expert in crafting ultra-compact titles',
	'You are an expert in crafting pithy titles',
] as const;
const INLINE_PROGRESS_MESSAGE_PREFIX =
	'You are an expert in writing short, catchy, and encouraging progress messages';
const GIT_BRANCH_NAME_PREFIX = 'You are an expert in crafting pithy branch names';
const GIT_COMMIT_MESSAGE_PREFIX =
	'You are an AI programming assistant, helping a software developer to come with the best git commit message';
const RENAME_SUGGESTIONS_PREFIX = 'You are a distinguished software engineer';
const MAIN_AGENT_PREFIX = 'You are an expert AI programming assistant';
const TERMINAL_NOTIFICATION_PATTERN = /^\[Terminal\s+\S+\s+notification:/;

// ── Keyword-based fallback matching (more resilient to prompt rewording) ──
// These are matched against the full first system message text (substring, not prefix).
// Ordered from most specific to least specific to avoid false positives.

const MAIN_AGENT_KEYWORDS = [
	'expert AI programming assistant',
	'<instructions>',       // Copilot agent instruction blocks
	'<skills>',              // Skill listing in agent prompts
	'<agents>',              // Sub-agent listing
	'<toolUseInstructions>', // Tool use instructions block
	'<memoryInstructions>',  // Memory instructions block
];

const TODO_TRACKER_KEYWORDS = ['background task tracker', 'track your progress'];
const CHAT_TITLE_KEYWORDS = ['ultra-compact title', 'pithy title', 'crafting titles'];
const GIT_KEYWORDS = ['git commit message', 'branch name', 'crafting pithy branch'];

// ── Regex-based fuzzy matching (tolerates minor prompt rewording) ──
// When Copilot tweaks wording (e.g. "expert" → "highly skilled", added articles,
// punctuation shifts), exact prefix matching fails but these regex patterns still
// catch the semantic core of each system prompt.
// Ordered from most specific to least specific to avoid false positives.

const FUZZY_CLASSIFIERS: ReadonlyArray<readonly [RegExp, RequestKind]> = [
	[/^You\s+are\s+an?\s+(?:expert\s+|highly\s+skilled\s+)?AI\s+programming\s+assistant/i, 'main-agent'],
	[/^You\s+are\s+a\s+background\s+task\s+tracker/i, 'todo-tracker'],
	[/expert\s+classifier\s+for\s+AI\s+coding\s+assistant\s+prompts/i, 'prompt-categorizer'],
	[/Visual\s+Studio\s+Code\s+assistant.*(?:returning|using)\s+settings/i, 'settings-resolver'],
	[/(?:ultra-compact|pithy)\s+titles?\b/i, 'chat-title'],
	[/short.*catchy.*encouraging.*progress\s+messages?/i, 'inline-progress-message'],
	[/pithy\s+branch\s+names?/i, 'git-branch-name'],
	[/best\s+git\s+commit\s+message/i, 'git-commit-message'],
	[/distinguished\s+software\s+engineer/i, 'rename-suggestions'],
];

// ── Tool-name-based classification (most robust — tool names rarely change) ──
// These serve as the primary fallback when prefix/keyword matching is ambiguous.

const TOOL_NAME_CLASSIFIERS: Readonly<Record<string, RequestKind>> = {
	manage_todo_list: 'todo-tracker',
	categorize_prompt: 'prompt-categorizer',
	setChatTitle: 'chat-title',
	getChatTitle: 'chat-title',
	generateGitBranchName: 'git-branch-name',
	generateGitCommitMessage: 'git-commit-message',
	generateRenameSuggestions: 'rename-suggestions',
};

// ── Requests where thinking is forced off to save costs ──
// NOTE: 'background' is intentionally excluded. It is a catch-all for unrecognized
// request patterns (e.g. when Copilot rewords prompts). Forcing thinking off for an
// unrecognized request that might actually be a main-agent call is worse than the
// small cost of leaving thinking enabled on a genuinely trivial request.
const REQUEST_KINDS_WITH_FORCED_NONE_THINKING = new Set<RequestKind>([
	'todo-tracker',
	'prompt-categorizer',
	'settings-resolver',
	'chat-title',
	'inline-progress-message',
	'git-branch-name',
	'git-commit-message',
	'rename-suggestions',
	'unknown',
]);

export function formatModelFields(vscodeModelId: string, apiModelId?: string): string {
	const apiField = apiModelId && apiModelId !== vscodeModelId ? ` apiModel=${apiModelId}` : '';
	return `model=${vscodeModelId}${apiField}`;
}

export function formatRequestLogLine(requestKind: RequestKind, message: string): string {
	return `[${requestKind}] ${message}`;
}

export function shouldForceThinkingNone(requestKind: RequestKind): boolean {
	return REQUEST_KINDS_WITH_FORCED_NONE_THINKING.has(requestKind);
}

export function classifyProviderRequest(input: {
	messages: readonly vscode.LanguageModelChatRequestMessage[];
	tools?: readonly vscode.LanguageModelChatTool[];
}): RequestKind {
	return classifyRequest({
		firstText: getFirstVscodeText(input.messages),
		latestUserText: getLatestVscodeUserText(input.messages),
		toolNames: input.tools?.map((tool) => tool.name) ?? [],
	});
}

export function classifyDeepSeekRequest(input: {
	request: DeepSeekRequest;
	inputMessages?: readonly vscode.LanguageModelChatRequestMessage[];
}): RequestKind {
	return classifyRequest({
		firstText:
			input.request.messages[0]?.content ??
			(input.inputMessages ? getFirstVscodeText(input.inputMessages) : ''),
		latestUserText:
			(input.inputMessages ? getLatestVscodeUserText(input.inputMessages) : '') ||
			getLatestDeepSeekUserText(input.request),
		toolNames: input.request.tools?.map(getDeepSeekToolName) ?? [],
	});
}

function classifyRequest(input: {
	firstText: string;
	latestUserText: string;
	toolNames: readonly string[];
}): RequestKind {
	const firstText = input.firstText.trimStart();
	const latestUserText = input.latestUserText.trimStart();
	const firstTextLower = firstText.toLowerCase();

	// ── Layer 1: Terminal notification (most specific, highest priority) ──
	if (TERMINAL_NOTIFICATION_PATTERN.test(latestUserText)) {
		return 'terminal-steering';
	}

	// ── Layer 2: Tool-name-based classification (robust — tool names rarely change) ──
	if (input.toolNames.length === 1) {
		const toolKind = TOOL_NAME_CLASSIFIERS[input.toolNames[0]];
		if (toolKind) {
			return toolKind;
		}
	}
	// Multi-tool check: if all tools belong to one known category, classify accordingly.
	if (input.toolNames.length > 1) {
		const allKinds = input.toolNames.map((n) => TOOL_NAME_CLASSIFIERS[n]).filter(Boolean);
		if (allKinds.length === input.toolNames.length && new Set(allKinds).size === 1) {
			return allKinds[0];
		}
	}

	// ── Layer 3: Prefix classification (exact, fast, but brittle) ──
	const PREFIX_CLASSIFIERS: ReadonlyArray<readonly [string | readonly string[], RequestKind]> = [
		[TODO_TRACKER_PREFIX, 'todo-tracker'],
		[PROMPT_CATEGORIZER_PREFIX, 'prompt-categorizer'],
		[SETTINGS_RESOLVER_PREFIX, 'settings-resolver'],
		[CHAT_TITLE_PREFIXES, 'chat-title'],
		[INLINE_PROGRESS_MESSAGE_PREFIX, 'inline-progress-message'],
		[GIT_BRANCH_NAME_PREFIX, 'git-branch-name'],
		[GIT_COMMIT_MESSAGE_PREFIX, 'git-commit-message'],
		[RENAME_SUGGESTIONS_PREFIX, 'rename-suggestions'],
	] as const;

	for (const [prefix, kind] of PREFIX_CLASSIFIERS) {
		if (typeof prefix === 'string') {
			if (firstText.startsWith(prefix)) {
				return kind;
			}
		} else if (startsWithAny(firstText, prefix)) {
			return kind;
		}
	}

	// ── Layer 3.5: Regex fuzzy matching (tolerates minor prompt rewording) ──
	// When Copilot tweaks wording (e.g. "expert" → "highly skilled"),
	// exact prefix matching fails but regex patterns still catch the semantic core.
	for (const [pattern, kind] of FUZZY_CLASSIFIERS) {
		if (pattern.test(firstText)) {
			return kind;
		}
	}

	// ── Layer 4: Main agent detection (prefix + keyword fallback) ──
	if (firstText.startsWith(MAIN_AGENT_PREFIX)) {
		return 'main-agent';
	}
	// Keyword-based fallback: if the prompt was reworded but still contains
	// characteristic agent markers, treat it as a main-agent request.
	if (matchesAnyKeyword(firstTextLower, MAIN_AGENT_KEYWORDS)) {
		// Only classify as main-agent via keyword if no other tool-signature was found.
		// (Layer 2 already handled single-tool cases above.)
		return 'main-agent';
	}

	// ── Layer 5: Keyword-based matching for non-agent categories ──
	if (matchesAnyKeyword(firstTextLower, TODO_TRACKER_KEYWORDS)) {
		return 'todo-tracker';
	}
	if (matchesAnyKeyword(firstTextLower, CHAT_TITLE_KEYWORDS)) {
		return 'chat-title';
	}
	if (matchesAnyKeyword(firstTextLower, GIT_KEYWORDS)) {
		// Narrower check: don't match 'branch name' in unrelated contexts
		if (firstTextLower.includes('commit message') || firstTextLower.includes('git commit')) {
			return 'git-commit-message';
		}
		if (firstTextLower.includes('branch name')) {
			return 'git-branch-name';
		}
	}

	// ── Layer 6: Catch-all fallback with diagnostic logging ──
	// 'background' is the safe catch-all when classification is uncertain.
	// It does NOT disable thinking (see REQUEST_KINDS_WITH_FORCED_NONE_THINKING).
	if (input.toolNames.length > 0 || firstText.length > 0) {
		logUncertainClassification('background', input);
		return 'background';
	}
	logUncertainClassification('unknown', input);
	return 'unknown';
}

/**
 * Log when a request falls through to a catch-all classification.
 * This helps detect when Copilot's internal prompts have changed and
 * the classifier needs updating.
 */
function logUncertainClassification(
	kind: 'background' | 'unknown',
	input: { firstText: string; latestUserText: string; toolNames: readonly string[] },
): void {
	// Log only a fingerprint to avoid leaking user content.
	const fingerprint = [
		`kind=${kind}`,
		`firstTextLen=${input.firstText.length}`,
		`firstTextHead=${safeHead(input.firstText, 80)}`,
		`tools=${input.toolNames.join(',') || '(none)'}`,
	].join(' ');
	logger.debug('[classifier] uncertain classification:', fingerprint);
}

function safeHead(text: string, maxLen: number): string {
	const head = text.slice(0, maxLen);
	return head.length < text.length ? head + '…' : head;
}

function matchesAnyKeyword(text: string, keywords: readonly string[]): boolean {
	return keywords.some((kw) => text.includes(kw));
}

function startsWithAny(text: string, prefixes: readonly string[]): boolean {
	return prefixes.some((prefix) => text.startsWith(prefix));
}

function getDeepSeekToolName(tool: DeepSeekTool): string {
	return tool.function.name;
}

function getFirstVscodeText(messages: readonly vscode.LanguageModelChatRequestMessage[]): string {
	const firstMessage = messages[0];
	if (!firstMessage) {
		return '';
	}

	return getVscodeMessageText(firstMessage);
}

function getLatestVscodeUserText(
	messages: readonly vscode.LanguageModelChatRequestMessage[],
): string {
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		const message = messages[index];
		if (message.role === vscode.LanguageModelChatMessageRole.User) {
			return getVscodeMessageText(message);
		}
	}
	return '';
}

function getVscodeMessageText(message: vscode.LanguageModelChatRequestMessage): string {
	let text = '';
	for (const part of message.content) {
		if (part instanceof vscode.LanguageModelTextPart) {
			text += part.value;
		}
	}
	return text;
}

function getLatestDeepSeekUserText(request: DeepSeekRequest): string {
	for (let index = request.messages.length - 1; index >= 0; index -= 1) {
		const message = request.messages[index];
		if (message.role === 'user') {
			return message.content;
		}
	}
	return '';
}
