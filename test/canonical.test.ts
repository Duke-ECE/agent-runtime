import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import {
  canonicalToPi,
  duplicateToolCallIds,
  historyToPi,
  text,
  toolCall,
  toolResult,
  unresolvedToolCalls,
  usageFromPi,
  usageToPi,
  validateBlocks,
  type CanonicalMessage,
} from "../src/canonical.js";

const identity = { api: "openai-completions", provider: "openai", model: "gpt-4o-mini" };

function message(overrides: Partial<CanonicalMessage>): CanonicalMessage {
  return {
    id: "msg-1",
    session_id: "sess-1",
    seq: 1,
    request_message_id: "msg-r1",
    role: "assistant",
    content: [text("hello")],
    format_version: 1,
    status: "complete",
    ...overrides,
  };
}

test("validateBlocks accepts the canonical block shapes", () => {
  assert.equal(validateBlocks([text("hi")]), null);
  assert.equal(validateBlocks([toolCall("call_1", "bash", { command: "pwd" })]), null);
  assert.equal(
    validateBlocks([toolResult("call_1", "success", [text("/workspace")])]),
    null,
  );
});

test("validateBlocks rejects empty, unknown, and malformed blocks", () => {
  assert.match(String(validateBlocks([])), /must not be empty/);
  assert.match(String(validateBlocks([{ type: "nope" } as never])), /unknown type/);
  assert.match(String(validateBlocks([{ type: "tool_call", id: "", name: "bash", arguments_json: "{}" }])), /needs id and name/);
  assert.match(
    String(validateBlocks([{ type: "tool_call", id: "c", name: "bash", arguments_json: "{not json" }])),
    /not valid JSON/,
  );
  assert.match(
    String(validateBlocks([{ type: "tool_result", tool_call_id: "c", status: "maybe" as never, content: [] }])),
    /status/,
  );
});

test("tool-call and result links are read by id, not adjacency", () => {
  const blocks = [toolCall("call_1", "bash", {}), toolCall("call_2", "read", {}), toolResult("call_2", "success", [text("ok")])];
  assert.deepEqual(unresolvedToolCalls(blocks), ["call_1"]);
  assert.deepEqual(unresolvedToolCalls([...blocks, toolResult("call_1", "error", [], "boom")]), []);
});

test("duplicateToolCallIds finds ids reused across messages", () => {
  const first = message({ id: "m1", content: [toolCall("call_1", "bash", {})] });
  const second = message({ id: "m2", content: [toolCall("call_1", "bash", {})], role: "assistant" });
  assert.deepEqual(duplicateToolCallIds([first, second]), ["call_1"]);
  assert.deepEqual(duplicateToolCallIds([first]), []);
});

test("canonicalToPi rebuilds user, assistant, and tool messages", () => {
  const user = canonicalToPi(message({ role: "user", content: [text("run pwd")] }), identity)[0] as UserMessage;
  assert.equal(user.role, "user");
  assert.deepEqual(user.content, [{ type: "text", text: "run pwd" }]);

  const assistant = canonicalToPi(
    message({ content: [text("running"), toolCall("call_1", "bash", { command: "pwd" })], usage: { input_tokens: 10, cached_input_tokens: 4, output_tokens: 2, reasoning_tokens: 0, total_tokens: 12 } }),
    identity,
  )[0] as AssistantMessage;
  assert.equal(assistant.role, "assistant");
  assert.equal(assistant.stopReason, "toolUse");
  assert.equal(assistant.usage.input, 10);
  assert.deepEqual(assistant.content[1], { type: "toolCall", id: "call_1", name: "bash", arguments: { command: "pwd" } });

  const tool = canonicalToPi(
    message({
      role: "tool",
      content: [toolResult("call_1", "success", [text("/workspace")])],
      provider_metadata: { tool_names: { call_1: "bash" } },
    }),
    identity,
  )[0] as ToolResultMessage;
  assert.equal(tool.role, "toolResult");
  assert.equal(tool.toolCallId, "call_1");
  assert.equal(tool.toolName, "bash");
  assert.equal(tool.isError, false);
  assert.deepEqual(tool.content, [{ type: "text", text: "/workspace" }]);
});

test("historyToPi excludes incomplete assistant output but keeps user and tool messages", () => {
  const history = [
    message({ id: "m1", role: "user", seq: 1, content: [text("go")] }),
    message({ id: "m2", role: "assistant", seq: 2, status: "partial", content: [text("half a sen")] }),
    message({ id: "m3", role: "tool", seq: 3, content: [toolResult("call_1", "error", [text("boom")])] }),
  ];
  const pi = historyToPi(history, identity);
  assert.equal(pi.length, 2);
  assert.equal(pi[0].role, "user");
  assert.equal(pi[1].role, "toolResult");
});

test("usage normalization does not double-count cached input", () => {
  const canonical = usageFromPi({ input: 100, output: 20, cacheRead: 30, cacheWrite: 0, totalTokens: 120, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } });
  assert.deepEqual(canonical, {
    input_tokens: 100,
    cached_input_tokens: 30,
    output_tokens: 20,
    reasoning_tokens: 0,
    total_tokens: 120,
  });
  const back = usageToPi(canonical);
  assert.equal(back.input, 100);
  assert.equal(back.cacheRead, 30);
  assert.equal(usageFromPi(undefined), undefined);
});
