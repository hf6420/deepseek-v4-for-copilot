export const OFFICIAL_DEEPSEEK_API_HOST = 'api.deepseek.com';

/**
 * Check whether a base URL points to the official DeepSeek API.
 * Only returns true for the exact hostname `api.deepseek.com`.
 */
export function isOfficialDeepSeekBaseUrl(baseUrl: string): boolean {
	try {
		return new URL(baseUrl).hostname.toLowerCase() === OFFICIAL_DEEPSEEK_API_HOST;
	} catch {
		return false;
	}
}

export function normalizeBaseUrl(baseUrl: string): string {
	return baseUrl.trim().replace(/\/+$/u, '');
}
