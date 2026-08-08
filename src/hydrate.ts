import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import type { TranscriptTurn } from "./session-client.js";

/**
 * Zero token usage stamped on hydrated assistant messages: real usage is not
 * persisted in the durable transcript, and pi-ai requires a Usage object.
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

/** Outcome of hydrating a durable transcript. */
export interface HydrationResult {
  messages: AgentMessage[];
  /**
   * Content of the transcript's LAST system turn (the session's persisted
   * system prompt), or "" when the transcript records none. System turns
   * never become pi conversation messages.
   */
  systemPrompt: string;
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
 * one wins — and never become pi messages. Unknown roles and malformed entries
 * are skipped with a warning — this function never throws.
 */
export function transcriptToMessages(turns: TranscriptTurn[], opts: HydrateOptions): HydrationResult {
  const messages: AgentMessage[] = [];
  let systemPrompt = "";
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
  return { messages, systemPrompt };
}
