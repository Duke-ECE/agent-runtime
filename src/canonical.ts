import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ToolResultMessage, UserMessage, Usage } from "@earendil-works/pi-ai";
import { ZERO_USAGE } from "./hydrate.js";

/**
 * The canonical message model of session.v2, as stored in agent_messages
 * (format_version 1). Field names match the proto so the wire, storage, and
 * domain shapes stay one vocabulary: ordered blocks, request identity, message
 * completeness, and per-model-call usage.
 */

export type CanonicalRole = "user" | "assistant" | "tool";
export type CanonicalMessageStatus = "complete" | "partial" | "interrupted";
export type ToolResultStatus = "success" | "error" | "cancelled";

export interface TextBlock {
  type: "text";
  text: string;
}

export interface ToolCallBlock {
  type: "tool_call";
  id: string;
  name: string;
  arguments_json: string;
}

export interface ToolResultBlock {
  type: "tool_result";
  tool_call_id: string;
  status: ToolResultStatus;
  content: CanonicalBlock[];
  error_code?: string;
}

export interface OpaqueBlock {
  type: "opaque";
  provider: string;
  opaque_type: string;
  data_json: Uint8Array;
}

export type CanonicalBlock = TextBlock | ToolCallBlock | ToolResultBlock | OpaqueBlock;

/** Provider usage normalized to one shape per model call. */
export interface CanonicalUsage {
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
}

export interface RequestExecution {
  client_request_id: string;
  request_hash: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
  error_code?: string;
  cancellation_requested: boolean;
}

export interface CanonicalMessage {
  id: string;
  session_id: string;
  seq: number;
  request_message_id: string;
  role: CanonicalRole;
  content: CanonicalBlock[];
  format_version: number;
  status: CanonicalMessageStatus;
  reply_to_message_id?: string;
  usage?: CanonicalUsage;
  provider_metadata?: Record<string, unknown>;
  request?: RequestExecution;
  created_at?: string;
}

export const FORMAT_VERSION = 1;

export function text(text: string): TextBlock {
  return { type: "text", text };
}

export function toolCall(id: string, name: string, args: unknown): ToolCallBlock {
  return { type: "tool_call", id, name, arguments_json: JSON.stringify(args ?? {}) };
}

export function toolResult(
  toolCallId: string,
  status: ToolResultStatus,
  content: CanonicalBlock[],
  errorCode?: string,
): ToolResultBlock {
  const block: ToolResultBlock = { type: "tool_result", tool_call_id: toolCallId, status, content };
  if (errorCode) block.error_code = errorCode;
  return block;
}

/** Explain why a block list is not valid, or null when it is. */
export function validateBlocks(blocks: CanonicalBlock[]): string | null {
  if (blocks.length === 0) return "content must not be empty";
  for (let i = 0; i < blocks.length; i++) {
    const problem = validateBlock(blocks[i], i);
    if (problem) return problem;
  }
  return null;
}

function validateBlock(block: CanonicalBlock, index: number): string | null {
  switch (block.type) {
    case "text":
      return typeof block.text === "string" ? null : `block ${index}: text must be a string`;
    case "tool_call": {
      if (!block.id || !block.name) return `block ${index}: tool_call needs id and name`;
      try {
        JSON.parse(block.arguments_json);
      } catch {
        return `block ${index}: tool_call arguments_json is not valid JSON`;
      }
      return null;
    }
    case "tool_result": {
      if (!block.tool_call_id) return `block ${index}: tool_result needs tool_call_id`;
      if (block.status !== "success" && block.status !== "error" && block.status !== "cancelled") {
        return `block ${index}: tool_result status ${String(block.status)} is invalid`;
      }
      for (let i = 0; i < block.content.length; i++) {
        const problem = validateBlock(block.content[i], i);
        if (problem) return problem;
      }
      return null;
    }
    case "opaque":
      if (!block.provider || !block.opaque_type) return `block ${index}: opaque needs provider and opaque_type`;
      return null;
    default:
      return `block ${index}: unknown type ${String((block as { type?: string }).type)}`;
  }
}

/** Every tool-call id in a block list, including nested result content. */
export function toolCallIds(blocks: CanonicalBlock[]): string[] {
  const out: string[] = [];
  for (const block of blocks) {
    if (block.type === "tool_call") out.push(block.id);
    if (block.type === "tool_result") out.push(...toolCallIds(block.content));
  }
  return out;
}

/** Every tool-call id a result block refers to. */
export function toolResultRefs(blocks: CanonicalBlock[]): string[] {
  const out: string[] = [];
  for (const block of blocks) {
    if (block.type === "tool_result") {
      out.push(block.tool_call_id);
      out.push(...toolResultRefs(block.content));
    }
  }
  return out;
}

/** Tool-call blocks without a paired result in the same list. */
export function unresolvedToolCalls(blocks: CanonicalBlock[]): string[] {
  const calls = toolCallIds(blocks);
  const resolved = new Set(toolResultRefs(blocks));
  return calls.filter((id) => !resolved.has(id));
}

/** Duplicate tool-call ids across a message list, which make links ambiguous. */
export function duplicateToolCallIds(messages: CanonicalMessage[]): string[] {
  const seen = new Map<string, number>();
  for (const message of messages) {
    for (const id of toolCallIds(message.content)) {
      seen.set(id, (seen.get(id) ?? 0) + 1);
    }
  }
  return [...seen.entries()].filter(([, count]) => count > 1).map(([id]) => id).sort();
}

// ------------------------------------------------------------- pi conversion

export interface PiIdentity {
  api: string;
  provider: string;
  model: string;
}

/** Convert a canonical usage record into pi's Usage shape. */
export function usageToPi(usage: CanonicalUsage | undefined): Usage {
  if (!usage) return ZERO_USAGE;
  return {
    input: usage.input_tokens,
    output: usage.output_tokens,
    cacheRead: usage.cached_input_tokens,
    cacheWrite: 0,
    totalTokens: usage.total_tokens,
    cost: ZERO_USAGE.cost,
  };
}

/** Normalize pi's usage into the canonical record for one model call. */
export function usageFromPi(usage: Usage | undefined): CanonicalUsage | undefined {
  if (!usage) return undefined;
  return {
    input_tokens: usage.input ?? 0,
    cached_input_tokens: usage.cacheRead ?? 0,
    output_tokens: usage.output ?? 0,
    reasoning_tokens: 0,
    total_tokens: usage.totalTokens ?? 0,
  };
}

function timestampOf(createdAt: string | undefined, fallback: number): number {
  const parsed = createdAt ? Date.parse(createdAt) : Number.NaN;
  return Number.isNaN(parsed) ? fallback : parsed;
}

/**
 * Convert one canonical message into the pi messages it replays as. An
 * assistant message with text and/or tool calls becomes one AssistantMessage;
 * a tool message becomes one ToolResultMessage per result block, linked by the
 * original tool-call id (never by adjacency).
 */
export function canonicalToPi(message: CanonicalMessage, identity: PiIdentity, fallbackTimestamp = Date.now()): AgentMessage[] {
  const timestamp = timestampOf(message.created_at, fallbackTimestamp);
  switch (message.role) {
    case "user": {
      const user: UserMessage = { role: "user", content: piTextContent(message.content), timestamp };
      return [user];
    }
    case "assistant": {
      const assistant: AssistantMessage = {
        role: "assistant",
        content: piAssistantContent(message.content) as AssistantMessage["content"],
        api: identity.api,
        provider: identity.provider,
        model: identity.model,
        usage: usageToPi(message.usage),
        stopReason: message.content.some((b) => b.type === "tool_call") ? "toolUse" : "stop",
        timestamp,
      };
      return [assistant];
    }
    case "tool": {
      const out: AgentMessage[] = [];
      for (const block of message.content) {
        if (block.type !== "tool_result") continue;
        const result: ToolResultMessage = {
          role: "toolResult",
          toolCallId: block.tool_call_id,
          toolName: providerToolName(message, block.tool_call_id),
          content: piTextContent(block.content),
          isError: block.status === "error",
          timestamp,
        };
        out.push(result);
      }
      return out;
    }
    default:
      return [];
  }
}

/** The tool name a result refers to, from the provider metadata the runtime recorded. */
function providerToolName(message: CanonicalMessage, toolCallId: string): string {
  const names = message.provider_metadata?.tool_names;
  if (names && typeof names === "object" && !Array.isArray(names)) {
    const name = (names as Record<string, unknown>)[toolCallId];
    if (typeof name === "string") return name;
  }
  return "tool";
}

/** Text-only content for user and tool-result messages. */
function piTextContent(blocks: CanonicalBlock[]): Array<{ type: "text"; text: string }> {
  const out: Array<{ type: "text"; text: string }> = [];
  for (const block of blocks) {
    if (block.type === "text") out.push({ type: "text", text: block.text });
  }
  return out;
}

/** Assistant content: text plus tool calls, in canonical order. */
function piAssistantContent(
  blocks: CanonicalBlock[],
): Array<{ type: "text"; text: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }> {
  const out: Array<
    { type: "text"; text: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
  > = [];
  for (const block of blocks) {
    if (block.type === "text") {
      out.push({ type: "text", text: block.text });
    } else if (block.type === "tool_call") {
      out.push({ type: "toolCall", id: block.id, name: block.name, arguments: parseArguments(block.arguments_json) });
    }
    // opaque provider blocks are replayed through provider-specific adapters;
    // the portable text/tool model drops them here.
  }
  return out;
}

function parseArguments(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // fall through
  }
  return {};
}

/**
 * Convert an ordered canonical history into pi's message list. Partial and
 * interrupted assistant output is excluded from model context by default (the
 * design's recovery rule): the caller inserts an explicit recovery note instead
 * of replaying unfinished model output.
 */
export function historyToPi(
  messages: CanonicalMessage[],
  identity: PiIdentity,
  policy: TruncationPolicy = DEFAULT_TRUNCATION,
): AgentMessage[] {
  const out: AgentMessage[] = [];
  for (const message of messages) {
    if (message.role === "assistant" && message.status !== "complete") continue;
    // Tool results are capped for the model only; the canonical message keeps
    // the full output, and the cut names where to read it. A call whose result
    // never landed (a crash) and a result whose call is gone (compaction) are
    // both dropped: neither can be replayed to a provider safely.
    let content: CanonicalBlock[];
    if (message.role === "tool") {
      const calls = new Set<string>();
      for (const other of messages) for (const id of toolCallIds(other.content)) calls.add(id);
      content = pairedResultContent(calls, truncateForModel(message.content, policy, message.id).blocks);
      if (content.length === 0) continue;
    } else if (message.role === "assistant") {
      content = pairedBlocks(messages, message.content);
    } else {
      content = message.content;
    }
    out.push(...canonicalToPi({ ...message, content }, identity));
  }
  return out;
}

// ------------------------------------------------- model-facing truncation

/**
 * How much of a tool result may reach the model. The stored canonical message is
 * never altered: this only bounds the copy assembled into a provider request, so
 * a huge tool output cannot blow the context window while the original stays
 * readable in the transcript.
 */
export interface TruncationPolicy {
  /** Maximum characters of tool-result text exposed to the model. */
  maxToolResultChars: number;
}

export const DEFAULT_TRUNCATION: TruncationPolicy = { maxToolResultChars: 20_000 };

export interface TruncatedContent {
  blocks: CanonicalBlock[];
  /** True when any block was shortened. */
  truncated: boolean;
}

/**
 * Shorten a tool result for the model, leaving the original untouched.
 *
 * The cut carries an explicit marker naming how much was removed and, when the
 * caller knows it, the canonical message the full result lives in — an
 * authorized range read can then fetch it. Truncation is never silent: a model
 * that saw a cut result must be able to tell that it was cut.
 */
export function truncateForModel(
  blocks: CanonicalBlock[],
  policy: TruncationPolicy = DEFAULT_TRUNCATION,
  originalMessageId?: string,
): TruncatedContent {
  let truncated = false;

  const walk = (input: CanonicalBlock[]): CanonicalBlock[] =>
    input.map((block) => {
      if (block.type === "text") {
        if (block.text.length <= policy.maxToolResultChars) return block;
        truncated = true;
        const omitted = block.text.length - policy.maxToolResultChars;
        const where = originalMessageId ? ` in message ${originalMessageId}` : "";
        return {
          type: "text",
          text:
            block.text.slice(0, policy.maxToolResultChars) +
            `\n[truncated: ${omitted} of ${block.text.length} characters omitted; the full tool result is preserved${where}]`,
        };
      }
      if (block.type === "tool_result") {
        const nested = walk(block.content);
        return nested === block.content ? block : { ...block, content: nested };
      }
      return block;
    });

  const out = walk(blocks);
  return { blocks: truncated ? out : blocks, truncated };
}

/**
 * Tool calls in the history that have no persisted result, and the assistant
 * message that issued each. This is the crash case: a runtime can die after
 * persisting the call but before persisting the observed result, so the
 * outcome is genuinely unknown.
 */
export interface UnresolvedToolCalls {
  assistantMessageId: string;
  toolCallIds: string[];
}

/**
 * Find tool calls recorded without their result anywhere in the given history.
 * The caller must not replay these: an unknown outcome is not a success, and a
 * provider protocol is never satisfied by inventing a result for one.
 */
export function unresolvedToolCallsInHistory(messages: CanonicalMessage[]): UnresolvedToolCalls[] {
  const resolved = new Set<string>();
  for (const message of messages) {
    for (const ref of toolResultRefs(message.content)) resolved.add(ref);
  }
  const out: UnresolvedToolCalls[] = [];
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const dangling = toolCallIds(message.content).filter((id) => !resolved.has(id));
    if (dangling.length > 0) out.push({ assistantMessageId: message.id, toolCallIds: dangling });
  }
  return out;
}

/**
 * Keep only tool calls whose result is present, and only results whose call is
 * present. A dangling call would be rejected by the provider (or worse, invite
 * a fabricated result), and an orphan result refers to a summarized-away call.
 * Assistant messages that lose every block keep an empty text block so the
 * message is still well-formed.
 */
function pairedBlocks(messages: CanonicalMessage[], content: CanonicalBlock[]): CanonicalBlock[] {
  const calls = new Set<string>();
  const results = new Set<string>();
  for (const message of messages) {
    for (const id of toolCallIds(message.content)) calls.add(id);
    for (const id of toolResultRefs(message.content)) results.add(id);
  }
  const out: CanonicalBlock[] = [];
  for (const block of content) {
    if (block.type === "tool_call") {
      if (results.has(block.id)) out.push(block);
      continue;
    }
    if (block.type === "tool_result") {
      if (calls.has(block.tool_call_id)) out.push(block);
      continue;
    }
    out.push(block);
  }
  return out.length > 0 ? out : [text("")];
}

/**
 * Unpair-safe tool-result content: keeps results whose call is present.
 */
function pairedResultContent(calls: Set<string>, blocks: CanonicalBlock[]): CanonicalBlock[] {
  return blocks.filter((block) => block.type !== "tool_result" || calls.has(block.tool_call_id));
}
