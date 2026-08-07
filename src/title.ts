import { createModel, createStreamFn, type SessionLlmConfig } from "./llm.js";

/**
 * One-shot session title generation. Called with the session's own LLM config
 * and the first user message; resolves to the raw (unsanitized) title text.
 * This is the injectable seam: tests replace it via createRuntime options.
 */
export type TitleGenerator = (llm: SessionLlmConfig, firstUserMessage: string) => Promise<string>;

const TITLE_SYSTEM_PROMPT =
  "You write very short conversation titles. Title this conversation in at most 8 words, " +
  "no quotes, no punctuation at the end. Reply with the title only.";

/** Titles are cosmetic; keep the completion tiny. */
const TITLE_MAX_TOKENS = 32;
const TITLE_MAX_CHARS = 80;

/**
 * Normalize a raw LLM title: keep the first line, trim, strip surrounding
 * quotes, cap at ~80 chars. Returns "" when nothing usable remains.
 */
export function sanitizeTitle(raw: string): string {
  const firstLine = (raw.split("\n", 1)[0] ?? "").trim();
  const unquoted = firstLine.replace(/^["'“”‘’]+|["'“”‘’]+$/g, "").trim();
  return unquoted.slice(0, TITLE_MAX_CHARS).trimEnd();
}

/**
 * Default title generator: a single direct model stream (no Agent run) seeded
 * with the user's first message, using the session's own model and endpoint.
 */
export const generateTitle: TitleGenerator = async (llm, firstUserMessage) => {
  const stream = await createStreamFn(llm)(
    createModel(llm),
    {
      systemPrompt: TITLE_SYSTEM_PROMPT,
      messages: [{ role: "user", content: firstUserMessage, timestamp: Date.now() }],
    },
    { maxTokens: TITLE_MAX_TOKENS },
  );
  const message = await stream.result();
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    throw new Error(message.errorMessage ?? "title generation failed");
  }
  return message.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
};
