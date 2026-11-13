import type { Model } from "@earendil-works/pi-ai";
import { streamSimple as openAiCompletionsStreamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import type { StreamFn } from "@earendil-works/pi-agent-core";

/**
 * Resolved LLM settings for one session: per-request LlmConfig from
 * CreateSession wins, otherwise the process env (LLM_API_KEY / LLM_BASE_URL /
 * LLM_MODEL) supplies defaults.
 */
export interface SessionLlmConfig {
  apiKey?: string;
  baseUrl: string;
  model: string;
}

/**
 * Build a pi-ai Model pointing at any OpenAI-compatible endpoint. This mirrors
 * how pi-coding-agent constructs custom models (provider-composer.ts
 * modelFromJson): api "openai-completions" + an explicit baseUrl.
 */
export function createModel(config: SessionLlmConfig): Model<"openai-completions"> {
  return {
    id: config.model,
    name: config.model,
    api: "openai-completions",
    provider: "openai-compatible",
    baseUrl: config.baseUrl,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 16384,
    // Google's OpenAI-compatible endpoint (and possibly others) rejects the
    // `store` parameter with a bare 400. pi-ai auto-detection doesn't know
    // these providers, so disable `store` explicitly.
    compat: { supportsStore: false },
  };
}

/**
 * Stream function for the pi Agent loop: delegates to pi-ai's OpenAI chat
 * completions streaming, injecting the session's API key on every call.
 */
export function createStreamFn(config: SessionLlmConfig): StreamFn {
  const disableDeepSeekThinking =
    URL.canParse(config.baseUrl) &&
    new URL(config.baseUrl).hostname === "api.deepseek.com" &&
    config.model === "deepseek-flash";
  return (model, context, options) =>
    openAiCompletionsStreamSimple(model as Model<"openai-completions">, context, {
      ...options,
      apiKey: config.apiKey,
      // DeepSeek Flash defaults to high-effort thinking. Our Chat contract
      // exposes only answer text, so explicitly disable it for this endpoint.
      // Preserve a caller-supplied payload hook and an explicit thinking value.
      onPayload: disableDeepSeekThinking
        ? async (payload, selectedModel) => {
            const customized = (await options?.onPayload?.(payload, selectedModel)) ?? payload;
            if (typeof customized !== "object" || customized === null || "thinking" in customized) return customized;
            return { ...customized, thinking: { type: "disabled" } };
          }
        : options?.onPayload,
    });
}
