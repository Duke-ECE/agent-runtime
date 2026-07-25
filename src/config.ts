export interface LlmEnvConfig {
  apiKey?: string;
  baseUrl: string;
  model: string;
}

export interface ServiceConfig {
  port: number;
  maxSessions: number;
  sessionTtlMinutes: number;
  llm: LlmEnvConfig;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ServiceConfig {
  return {
    port: Number(env.PORT ?? 50052),
    maxSessions: Number(env.MAX_SESSIONS ?? 20),
    sessionTtlMinutes: Number(env.SESSION_TTL_MINUTES ?? 30),
    llm: {
      apiKey: env.LLM_API_KEY || undefined,
      baseUrl: env.LLM_BASE_URL || "https://api.openai.com/v1",
      model: env.LLM_MODEL || "gpt-4o-mini",
    },
  };
}
