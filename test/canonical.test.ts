import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import {
  canonicalToPi,
  duplicateToolCallIds,
  truncateForModel,
  historyToPi,
  text,
  toolCall,
  toolResult,
  unresolvedToolCalls,
  unresolvedToolCallsInHistory,
  usageFromPi,
  usageToPi,
  validateBlocks,
  type CanonicalBlock,
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
    message({ id: "m2", role: "assistant", seq: 2, content: [toolCall("call_1", "bash", {})] }),
    message({ id: "m3", role: "tool", seq: 3, content: [toolResult("call_1", "error", [text("boom")])] }),
    message({ id: "m4", role: "assistant", seq: 4, status: "partial", content: [text("half a sen")] }),
  ];
  const pi = historyToPi(history, identity);
  assert.equal(pi.length, 3, "the partial assistant message is excluded");
  assert.equal(pi[0].role, "user");
  assert.equal(pi[1].role, "assistant");
  assert.equal(pi[2].role, "toolResult");
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

test("short tool results are not touched", () => {
  const blocks = [toolResult("call_1", "success", [text("short output")])];
  const result = truncateForModel(blocks, { maxToolResultChars: 100 }, "msg-t1");
  assert.equal(result.truncated, false);
  assert.equal(result.blocks, blocks, "an untouched result should be returned as-is");
});

test("a tool result exactly at the cap is not truncated", () => {
  const exact = "x".repeat(100);
  const result = truncateForModel([toolResult("call_1", "success", [text(exact)])], { maxToolResultChars: 100 }, "msg-t1");
  assert.equal(result.truncated, false);
  const block = result.blocks[0];
  assert.equal(block.type, "tool_result");
});

test("an oversized tool result is cut with an explicit, named marker", () => {
  const body = "x".repeat(1_000);
  const result = truncateForModel([toolResult("call_1", "success", [text(body)])], { maxToolResultChars: 100 }, "msg-t9");
  assert.equal(result.truncated, true);
  const block = result.blocks[0];
  assert.equal(block.type, "tool_result");
  if (block.type !== "tool_result") return;
  const inner = block.content[0];
  assert.equal(inner.type, "text");
  if (inner.type !== "text") return;
  assert.ok(inner.text.startsWith("x".repeat(100)), "the cap is applied verbatim");
  assert.match(inner.text, /truncated: 900 of 1000 characters omitted/);
  assert.match(inner.text, /preserved in message msg-t9/);
});

test("truncation never mutates the stored result", () => {
  const body = "y".repeat(500);
  const stored = [toolResult("call_1", "success", [text(body)])];
  const result = truncateForModel(stored, { maxToolResultChars: 50 }, "msg-t1");
  assert.equal(result.truncated, true);
  const storedInner = (stored[0] as { content: CanonicalBlock[] }).content[0];
  assert.equal(storedInner.type, "text");
  if (storedInner.type === "text") {
    assert.equal(storedInner.text, body, "the canonical message must keep the full output");
  }
});

test("historyToPi caps tool output but leaves user and assistant content alone", () => {
  const long = "z".repeat(500);
  const history: CanonicalMessage[] = [
    message({ id: "m1", role: "user", seq: 1, content: [text(long)] }),
    message({ id: "m2", role: "assistant", seq: 2, content: [toolCall("call_1", "bash", {})] }),
    message({
      id: "m3",
      role: "tool",
      seq: 3,
      content: [toolResult("call_1", "success", [text(long)])],
    }),
  ];
  const pi = historyToPi(history, identity, { maxToolResultChars: 20 });
  const user = pi[0] as UserMessage;
  const userText = user.content as Array<{ type: string; text?: string }>;
  assert.equal(userText[0].text, long, "user input is never truncated");
  const tool = pi[2] as ToolResultMessage;
  const toolText = tool.content as Array<{ type: string; text?: string }>;
  assert.ok((toolText[0].text ?? "").length < long.length, "tool output is capped");
  assert.match(toolText[0].text ?? "", /truncated:/);
});

test("a tool call with no persisted result is reported as unresolved", () => {
  const history: CanonicalMessage[] = [
    message({ id: "m1", role: "user", seq: 1, content: [text("run pwd")] }),
    message({
      id: "m2",
      role: "assistant",
      seq: 2,
      content: [toolCall("call_crash", "bash", { command: "pwd" })],
    }),
  ];
  assert.deepEqual(unresolvedToolCallsInHistory(history), [
    { assistantMessageId: "m2", toolCallIds: ["call_crash"] },
  ]);
});

test("resolved and parallel tool calls are not reported", () => {
  const history: CanonicalMessage[] = [
    message({ id: "m1", role: "user", seq: 1, content: [text("go")] }),
    message({
      id: "m2",
      role: "assistant",
      seq: 2,
      content: [toolCall("call_a", "bash", {}), toolCall("call_b", "read", {})],
    }),
    message({
      id: "m3",
      role: "tool",
      seq: 3,
      // Parallel results arrive out of order; both must still count as resolved.
      content: [toolResult("call_b", "success", [text("b")]), toolResult("call_a", "error", [text("a")])],
    }),
  ];
  assert.deepEqual(unresolvedToolCallsInHistory(history), []);
});

test("a crash-dangling tool call is never replayed to the provider", () => {
  const history: CanonicalMessage[] = [
    message({ id: "m1", role: "user", seq: 1, content: [text("run pwd")] }),
    message({
      id: "m2",
      role: "assistant",
      seq: 2,
      content: [text("running pwd"), toolCall("call_crash", "bash", { command: "pwd" })],
    }),
  ];
  const pi = historyToPi(history, identity);
  const assistant = pi[1] as AssistantMessage;
  const kinds = (assistant.content as Array<{ type: string }>).map((part) => part.type);
  assert.ok(!kinds.includes("toolCall"), "a dangling call must not reach the provider");
  assert.deepEqual(kinds, ["text"], "the assistant text is kept");
});

test("an orphan tool result whose call is gone is dropped, not sent alone", () => {
  const history: CanonicalMessage[] = [
    message({
      id: "m9",
      role: "tool",
      seq: 9,
      content: [toolResult("call_summarized_away", "success", [text("old output")])],
    }),
  ];
  assert.deepEqual(historyToPi(history, identity), []);
});

test("paired calls and results both survive hydration", () => {
  const history: CanonicalMessage[] = [
    message({ id: "m1", role: "user", seq: 1, content: [text("go")] }),
    message({ id: "m2", role: "assistant", seq: 2, content: [toolCall("call_1", "bash", {})] }),
    message({ id: "m3", role: "tool", seq: 3, content: [toolResult("call_1", "success", [text("ok")])] }),
  ];
  const pi = historyToPi(history, identity);
  assert.equal(pi.length, 3);
  assert.equal((pi[1] as AssistantMessage).content[0].type, "toolCall");
  assert.equal((pi[2] as ToolResultMessage).toolCallId, "call_1");
});
