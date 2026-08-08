import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import type { TranscriptTurn } from "./session-client.js";

/**
 * Zero token usage stamped on hydrated assistant messages. Assistant
 * content_json may carry an optional persisted `usage` object
 * (`{"input_tokens": N, "output_tokens": M}`), but hydration ignores it —
 * pi-ai requires a Usage object, so hydrated messages get zero usage.
 */
export const ZERO_USAGE: Usage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

/** Identity fields hydrated assistant messages inherit from the session model. */
export interface HydrateOptions {
  provider: string;
  model: string;
  api: string;
}

/**
 * The session's frozen LLM triple as recorded in transcript "config" turns
 * (`content_json` `{"llm": {...}}`). All-string fields; a missing/absent
 * field surfaces as "" (a key-less session records api_key "").
 */
export interface TranscriptLlm {
  api_key: string;
  base_url: string;
  model: string;
}

/** Outcome of hydrating a durable transcript. */
export interface HydrationResult {
  messages: AgentMessage[];
  /**
   * Content of the transcript's LAST system turn (the session's persisted
   * system prompt), or "" when the transcript records none. System turns
   * never become pi conversation messages.
   */
  systemPrompt: string;
  /**
   * The transcript's LAST config turn (the session's frozen LLM triple), or
   * all-"" fields when the transcript records none. Config turns never
   * become pi conversation messages.
   */
  llm: TranscriptLlm;
}

function toTimestamp(createdAt: string | undefined): number {
  const parsed = createdAt ? Date.parse(createdAt) : Number.NaN;
  return Number.isNaN(parsed) ? Date.now() : parsed;
}

/** Parse a turn's content_json; returns undefined (after warning) on malformed input. */
function parsePayload(turn: TranscriptTurn): Record<string, unknown> | undefined {
  try {
    const payload: unknown = JSON.parse(turn.contentJson);
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      return payload as Record<string, unknown>;
    }
  } catch {
    // fall through to the warning
  }
  console.warn(`hydration: skipping turn seq=${turn.seq} role=${turn.role}: malformed content_json`);
  return undefined;
}

function textPayload(turn: TranscriptTurn): string | undefined {
  const payload = parsePayload(turn);
  if (!payload) return undefined;
  if (typeof payload.content !== "string") {
    console.warn(`hydration: skipping turn seq=${turn.seq} role=${turn.role}: missing "content" string`);
    return undefined;
  }
  return payload.content;
}

/**
 * Parse a "config" turn's frozen LLM triple; returns undefined (after warning)
 * on malformed input. Non-string fields surface as "".
 */
function parseConfigTurn(turn: TranscriptTurn): TranscriptLlm | undefined {
  const payload = parsePayload(turn);
  if (!payload) return undefined;
  const llm = payload.llm;
  if (!llm || typeof llm !== "object" || Array.isArray(llm)) {
    console.warn(`hydration: skipping turn seq=${turn.seq} role=${turn.role}: missing "llm" object`);
    return undefined;
  }
  const record = llm as Record<string, unknown>;
  return {
    api_key: typeof record.api_key === "string" ? record.api_key : "",
    base_url: typeof record.base_url === "string" ? record.base_url : "",
    model: typeof record.model === "string" ? record.model : "",
  };
}

/**
 * Extract the session's frozen LLM triple from raw transcript turns — the
 * LAST valid "config" turn wins; undefined when the transcript records none
 * (pre-config-turn transcripts and fresh sessions). CreateSession uses this
 * before building the model: the frozen triple takes precedence over the
 * request llm (snapshot semantics — a session's model never switches).
 */
export function extractTranscriptLlm(turns: TranscriptTurn[]): TranscriptLlm | undefined {
  let llm: TranscriptLlm | undefined;
  for (const turn of turns) {
    if (turn.role !== "config") continue;
    const parsed = parseConfigTurn(turn);
    if (parsed) llm = parsed;
  }
  return llm;
}

/**
 * Extract the session's persisted system prompt from raw transcript turns —
 * the LAST valid "system" turn wins; "" when the transcript records none.
 * Used alongside extractTranscriptLlm for marker recovery when a windowed
 * hydration skipped the transcript's head (where these markers usually sit).
 */
export function extractTranscriptSystemPrompt(turns: TranscriptTurn[]): string {
  let prompt = "";
  for (const turn of turns) {
    if (turn.role !== "system") continue;
    const text = textPayload(turn);
    if (text !== undefined) prompt = text;
  }
  return prompt;
}

function parseArguments(raw: unknown, seq: number, name: string): Record<string, unknown> {
  if (typeof raw === "string" && raw.length > 0) {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // fall through to the warning
    }
    console.warn(`hydration: tool_call seq=${seq} (${name}): bad arguments_json, using {}`);
  }
  return {};
}

/**
 * Convert a durable session-manager transcript back into pi agent messages for
 * `AgentOptions.initialState.messages`. The transcript records one aggregated
 * assistant text per turn plus tool_call/tool_result pairs (see the AppendTurn
 * call site in server.ts); tool pairs are rebuilt into an assistant
 * toolCall message followed by its toolResult message, linked by a synthetic
 * `hydrated-<seq>` id. System turns (role "system", same `{"content": ...}`
 * shape as user turns) are collected into the result's `systemPrompt` — last
 * one wins — and never become pi messages. Config turns (role "config",
 * `{"llm": {api_key, base_url, model}}`) are collected into the result's
 * `llm` — last one wins — and never become pi messages either. Unknown roles
 * and malformed entries are skipped with a warning — this function never
 * throws. Payload keys
 * beyond `content` (e.g. the optional `usage` object on assistant turns) are
 * ignored, so a windowed transcript that starts mid-tool-pair hydrates
 * cleanly too (the orphan tool_result is skipped like any other).
 */
export function transcriptToMessages(turns: TranscriptTurn[], opts: HydrateOptions): HydrationResult {
  const messages: AgentMessage[] = [];
  let systemPrompt = "";
  let llm: TranscriptLlm = { api_key: "", base_url: "", model: "" };
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    const timestamp = toTimestamp(turn.createdAt);
    switch (turn.role) {
      case "system": {
        const text = textPayload(turn);
        if (text === undefined) break;
        systemPrompt = text; // last system turn wins
        break;
      }
      case "config": {
        const parsed = parseConfigTurn(turn);
        if (parsed) llm = parsed; // last config turn wins
        break;
      }
      case "user": {
        const text = textPayload(turn);
        if (text === undefined) break;
        const message: UserMessage = { role: "user", content: [{ type: "text", text }], timestamp };
        messages.push(message);
        break;
      }
      case "assistant": {
        const text = textPayload(turn);
        if (text === undefined) break;
        const message: AssistantMessage = {
          role: "assistant",
          content: [{ type: "text", text }],
          api: opts.api,
          provider: opts.provider,
          model: opts.model,
          usage: ZERO_USAGE,
          stopReason: "stop",
          timestamp,
        };
        messages.push(message);
        break;
      }
      case "tool_call": {
        const payload = parsePayload(turn);
        if (!payload) break;
        const name = payload.tool;
        if (typeof name !== "string" || !name) {
          console.warn(`hydration: skipping tool_call seq=${turn.seq}: missing "tool" name`);
          break;
        }
        const next = turns[i + 1];
        const resultPayload = next && next.role === "tool_result" ? parsePayload(next) : undefined;
        if (!resultPayload) {
          console.warn(`hydration: skipping tool_call seq=${turn.seq} (${name}): no valid tool_result follows`);
          // A malformed tool_result can never pair with anything; consume it.
          if (next?.role === "tool_result") i++;
          break;
        }
        if (resultPayload.tool !== name) {
          console.warn(
            `hydration: skipping tool_call seq=${turn.seq} (${name}): following tool_result is for ${String(resultPayload.tool)}`,
          );
          break;
        }
        i++; // consume the paired tool_result
        const ok = resultPayload.ok === true;
        const output = typeof resultPayload.output === "string" ? resultPayload.output : "";
        const error = typeof resultPayload.error === "string" ? resultPayload.error : "";
        const toolCallId = `hydrated-${turn.seq}`;
        const assistantMessage: AssistantMessage = {
          role: "assistant",
          content: [{ type: "toolCall", id: toolCallId, name, arguments: parseArguments(payload.arguments_json, turn.seq, name) }],
          api: opts.api,
          provider: opts.provider,
          model: opts.model,
          usage: ZERO_USAGE,
          stopReason: "toolUse",
          timestamp,
        };
        const resultMessage: ToolResultMessage = {
          role: "toolResult",
          toolCallId,
          toolName: name,
          content: [{ type: "text", text: ok ? output : error || output }],
          isError: !ok,
          timestamp: toTimestamp(next.createdAt),
        };
        messages.push(assistantMessage, resultMessage);
        break;
      }
      case "tool_result":
        console.warn(`hydration: skipping orphan tool_result seq=${turn.seq}`);
        break;
      default:
        console.warn(`hydration: skipping unknown role ${turn.role} (seq=${turn.seq})`);
        break;
    }
  }
  return { messages, systemPrompt, llm };
}
