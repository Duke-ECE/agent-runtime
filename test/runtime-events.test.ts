import assert from "node:assert/strict";
import { test } from "node:test";
import {
  aggregateUsage,
  isRetryableLlmError,
  toolResultContent,
  toolResultText,
  translateAgentEvent,
} from "../src/runtime-events.js";

test("text deltas become provisional frames carrying the pending message id", () => {
  const result = translateAgentEvent(
    { type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "hel" } },
    "msg-a1",
  );
  assert.deepEqual(result.events, [{ kind: "text_delta", payload: { message_id: "msg-a1", delta: "hel" } }]);
});

test("non-text assistant events are not streamed", () => {
  const result = translateAgentEvent(
    { type: "message_update", assistantMessageEvent: { type: "thinking_delta", delta: "hmm" } },
    "msg-a1",
  );
  assert.deepEqual(result.events, []);
});

test("tool starts and ends carry the original tool-call id", () => {
  const start = translateAgentEvent(
    { type: "tool_execution_start", toolCallId: "call_1", toolName: "bash", args: { command: "pwd" } },
    "msg-a1",
  );
  assert.deepEqual(start.events[0], {
    kind: "tool_call",
    payload: { message_id: "msg-a1", tool_call_id: "call_1", tool: "bash", arguments_json: '{"command":"pwd"}' },
  });

  const end = translateAgentEvent(
    { type: "tool_execution_end", toolCallId: "call_1", toolName: "bash", result: { content: [{ type: "text", text: "/workspace" }] }, isError: false },
    "msg-a1",
  );
  assert.deepEqual(end.events[0], {
    kind: "tool_result",
    payload: {
      message_id: "msg-a1",
      tool_call_id: "call_1",
      status: "TOOL_RESULT_STATUS_SUCCESS",
      content: [{ text: { text: "/workspace" } }],
      error_code: "",
    },
  });

  const failed = translateAgentEvent(
    { type: "tool_execution_end", toolCallId: "call_1", toolName: "bash", result: "boom", isError: true },
    "msg-a1",
  );
  const payload = failed.events[0];
  assert.equal(payload.kind, "tool_result");
  if (payload.kind === "tool_result") {
    assert.equal(payload.payload.status, "TOOL_RESULT_STATUS_ERROR");
    assert.equal(payload.payload.error_code, "tool_error");
  }
});

test("a failed model call becomes an error frame and reports no usage", () => {
  const result = translateAgentEvent(
    { type: "message_end", message: { role: "assistant", stopReason: "error", errorMessage: "429 rate limited" } },
    "msg-r1",
  );
  assert.deepEqual(result.events, [
    {
      kind: "error",
      payload: { request_message_id: "msg-r1", code: "llm_error", message: "429 rate limited", retryable: true },
    },
  ]);
  assert.equal(result.usageDelta, undefined);
});

test("an aborted model call is not retryable", () => {
  const result = translateAgentEvent(
    { type: "message_end", message: { role: "assistant", stopReason: "aborted", errorMessage: "aborted" } },
    "msg-r1",
  );
  const event = result.events[0];
  assert.equal(event.kind, "error");
  if (event.kind === "error") {
    assert.equal(event.payload.code, "aborted");
    assert.equal(event.payload.retryable, false);
  }
});

test("usage is reported per model call, never aggregated by the translator", () => {
  const first = translateAgentEvent(
    { type: "message_end", message: { role: "assistant", stopReason: "stop", usage: { input: 10, output: 4 } } },
    "msg-a1",
  );
  const second = translateAgentEvent(
    { type: "message_end", message: { role: "assistant", stopReason: "stop", usage: { input: 30, output: 9 } } },
    "msg-a2",
  );
  assert.deepEqual(first.usageDelta, { input: 10, output: 4 });
  assert.deepEqual(second.usageDelta, { input: 30, output: 9 });

  const total = aggregateUsage([first.usageDelta, undefined, second.usageDelta]);
  assert.deepEqual(total, { input_tokens: 40, output_tokens: 13, total_tokens: 53 });
});

test("user message events and unknown events are ignored", () => {
  assert.deepEqual(translateAgentEvent({ type: "message_end", message: { role: "user" } }, "m").events, []);
  assert.deepEqual(translateAgentEvent({ type: "agent_start" }, "m").events, []);
});

test("retryability is judged from the provider message", () => {
  assert.equal(isRetryableLlmError("429 Too Many Requests"), true);
  assert.equal(isRetryableLlmError("fetch failed"), true);
  assert.equal(isRetryableLlmError("invalid api key"), false);
  assert.equal(isRetryableLlmError(undefined), false);
});

test("tool result rendering falls back to serialized details", () => {
  assert.equal(toolResultText({ content: [{ type: "text", text: "a" }, { type: "text", text: "b" }] }), "a\nb");
  assert.equal(toolResultText("plain"), "plain");
  assert.deepEqual(toolResultContent({ details: { ok: true } }), [{ text: { text: '{"details":{"ok":true}}' } }]);
});
