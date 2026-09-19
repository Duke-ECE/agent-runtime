/**
 * Translation from pi agent events to runtime.v2 ChatResponse payloads.
 *
 * Kept pure and separate from the service so the streaming contract can be
 * tested without a model, a lease, or a gRPC stream. Provisional deltas carry
 * the pending canonical message id; nothing here persists anything — the
 * durable boundaries are the tool hooks, not the stream.
 */

export interface V2TextDelta {
  message_id: string;
  delta: string;
}

export interface V2ToolCall {
  message_id: string;
  tool_call_id: string;
  tool: string;
  arguments_json: string;
}

export interface V2ToolResult {
  message_id: string;
  tool_call_id: string;
  status: "TOOL_RESULT_STATUS_SUCCESS" | "TOOL_RESULT_STATUS_ERROR" | "TOOL_RESULT_STATUS_CANCELLED";
  content: Array<{ text: { text: string } }>;
  error_code: string;
}

export interface V2Error {
  request_message_id: string;
  code: string;
  message: string;
  retryable: boolean;
}

export interface V2Done {
  request_message_id: string;
  status: string;
  revision: number;
  aggregate_usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
}

export type V2StreamEvent =
  | { kind: "text_delta"; payload: V2TextDelta }
  | { kind: "tool_call"; payload: V2ToolCall }
  | { kind: "tool_result"; payload: V2ToolResult }
  | { kind: "error"; payload: V2Error };

/** Transient provider failures are worth retrying; a bad request is not. */
export function isRetryableLlmError(message: string | undefined): boolean {
  if (!message) return false;
  return /rate.?limit|429|timeout|timed out|overloaded|502|503|529|econnreset|fetch failed|network/i.test(message);
}

/** The text parts of a tool execution result, flattened. */
export function toolResultText(result: unknown): string {
  const content = (result as { content?: unknown })?.content;
  if (Array.isArray(content)) {
    const parts = content
      .filter((part): part is { type: string; text?: unknown } =>
        Boolean(part && typeof part === "object" && (part as { type?: unknown }).type === "text"),
      )
      .map((part) => String(part.text ?? ""));
    if (parts.length > 0) return parts.join("\n");
  }
  if (typeof result === "string") return result;
  try {
    return JSON.stringify(result ?? null);
  } catch {
    return String(result);
  }
}

/** Tool result content as a v2 ContentBlock array (text-only for now). */
export function toolResultContent(result: unknown): Array<{ text: { text: string } }> {
  const content = (result as { content?: unknown })?.content;
  if (Array.isArray(content)) {
    const parts = content
      .filter((part): part is { type: string; text?: unknown } =>
        Boolean(part && typeof part === "object" && (part as { type?: unknown }).type === "text"),
      )
      .map((part) => ({ text: { text: String(part.text ?? "") } }));
    if (parts.length > 0) return parts;
  }
  return [{ text: { text: toolResultText(result) } }];
}

/** Minimal shape of the pi events this translation consumes. */
export interface AgentEventLike {
  type: string;
  assistantMessageEvent?: { type: string; delta?: string };
  toolCallId?: string;
  toolName?: string;
  args?: unknown;
  result?: unknown;
  isError?: boolean;
  message?: {
    role?: string;
    stopReason?: string;
    errorMessage?: string;
    usage?: { input?: number; output?: number };
  };
}

export interface Translation {
  events: V2StreamEvent[];
  /** Input/output tokens this event reported, when it completed a model call. */
  usageDelta?: { input: number; output: number };
}

/**
 * Translate one agent event. Text and tool events become provisional stream
 * frames; a completed model call with an error or abort becomes an error frame
 * and reports no usage. Usage is reported per model call, never aggregated here.
 */
export function translateAgentEvent(event: AgentEventLike, messageId: string): Translation {
  switch (event.type) {
    case "message_update": {
      const delta = event.assistantMessageEvent;
      if (delta?.type === "text_delta" && typeof delta.delta === "string") {
        return { events: [{ kind: "text_delta", payload: { message_id: messageId, delta: delta.delta } }] };
      }
      return { events: [] };
    }
    case "tool_execution_start":
      return {
        events: [
          {
            kind: "tool_call",
            payload: {
              message_id: messageId,
              tool_call_id: String(event.toolCallId ?? ""),
              tool: String(event.toolName ?? ""),
              arguments_json: JSON.stringify(event.args ?? {}),
            },
          },
        ],
      };
    case "tool_execution_end":
      return {
        events: [
          {
            kind: "tool_result",
            payload: {
              message_id: messageId,
              tool_call_id: String(event.toolCallId ?? ""),
              status: event.isError ? "TOOL_RESULT_STATUS_ERROR" : "TOOL_RESULT_STATUS_SUCCESS",
              content: toolResultContent(event.result),
              error_code: event.isError ? "tool_error" : "",
            },
          },
        ],
      };
    case "message_end": {
      const message = event.message;
      if (!message || message.role !== "assistant") return { events: [] };
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        const detail = message.errorMessage ?? "LLM request failed";
        return {
          events: [
            {
              kind: "error",
              payload: {
                request_message_id: messageId,
                code: message.stopReason === "aborted" ? "aborted" : "llm_error",
                message: detail,
                retryable: message.stopReason === "aborted" ? false : isRetryableLlmError(detail),
              },
            },
          ],
        };
      }
      if (message.usage) {
        return { events: [], usageDelta: { input: message.usage.input ?? 0, output: message.usage.output ?? 0 } };
      }
      return { events: [] };
    }
    default:
      return { events: [] };
  }
}

/** Aggregate per-model-call usage into the done frame's totals. */
export function aggregateUsage(
  calls: Array<{ input: number; output: number } | undefined>,
): { input_tokens: number; output_tokens: number; total_tokens: number } {
  let input = 0;
  let output = 0;
  for (const call of calls) {
    if (!call) continue;
    input += call.input;
    output += call.output;
  }
  return { input_tokens: input, output_tokens: output, total_tokens: input + output };
}
