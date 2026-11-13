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
  thinkingMode?: string;
}

export type ThinkingMode = "off" | "low" | "high" | "max" | "provider_default";

function isDeepSeekFlash(config: SessionLlmConfig): boolean {
  return URL.canParse(config.baseUrl) &&
    new URL(config.baseUrl).hostname === "api.deepseek.com" &&
    config.model === "deepseek-flash";
}

/** Resolve legacy requests and reject explicit modes unsupported by this adapter. */
export function resolveThinkingMode(config: SessionLlmConfig): ThinkingMode {
  const mode = config.thinkingMode || (isDeepSeekFlash(config) ? "off" : "provider_default");
  if (mode !== "off" && mode !== "low" && mode !== "high" && mode !== "max" && mode !== "provider_default") {
    throw new Error(`invalid thinking_mode: ${mode}`);
  }
  if (mode !== "provider_default" && !isDeepSeekFlash(config)) {
    throw new Error(`thinking_mode ${mode} is not supported for model ${config.model}`);
  }
  return mode;
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
export function createStreamFn(config: SessionLlmConfig, turnOverride?: () => "off" | undefined): StreamFn {
  return (model, context, options) => {
    // Resolve on every provider call so tool follow-ups share the active
    // turn override. Clearing the override restores the frozen session mode.
    const thinkingMode = resolveThinkingMode({ ...config, thinkingMode: turnOverride?.() ?? config.thinkingMode });
    return openAiCompletionsStreamSimple(model as Model<"openai-completions">, context, {
      ...options,
      apiKey: config.apiKey,
      // provider_default leaves the provider payload untouched.
      onPayload: thinkingMode !== "provider_default"
        ? async (payload, selectedModel) => {
            const customized = (await options?.onPayload?.(payload, selectedModel)) ?? payload;
            if (typeof customized !== "object" || customized === null || "thinking" in customized) return customized;
            return thinkingMode === "off"
              ? { ...customized, thinking: { type: "disabled" } }
              : { ...customized, thinking: { type: "enabled" }, reasoning_effort: thinkingMode };
          }
        : options?.onPayload,
    });
  };
}
