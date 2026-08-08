import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import { extractTranscriptLlm, transcriptToMessages, ZERO_USAGE } from "../src/hydrate.js";
import type { TranscriptTurn } from "../src/session-client.js";

const OPTS = { api: "openai-completions", provider: "openai-compatible", model: "test-model" };
const TS = "2026-01-02T03:04:05.000Z";
const TS_MS = Date.parse(TS);

function turn(seq: number, role: string, payload: unknown, createdAt: string | undefined = TS): TranscriptTurn {
  return { seq, role, contentJson: typeof payload === "string" ? payload : JSON.stringify(payload), createdAt };
}

test("empty transcript yields no messages, no system prompt, and no llm config", () => {
  assert.deepEqual(transcriptToMessages([], OPTS), {
    messages: [],
    systemPrompt: "",
    llm: { api_key: "", base_url: "", model: "" },
  });
});

test("user and assistant turns map to text messages", () => {
  const { messages } = transcriptToMessages(
    [turn(1, "user", { content: "hi" }), turn(2, "assistant", { content: "hello there" })],
    OPTS,
  );
  assert.equal(messages.length, 2);

  const user = messages[0] as UserMessage;
  assert.equal(user.role, "user");
  assert.deepEqual(user.content, [{ type: "text", text: "hi" }]);
  assert.equal(user.timestamp, TS_MS);

  const assistant = messages[1] as AssistantMessage;
  assert.equal(assistant.role, "assistant");
  assert.deepEqual(assistant.content, [{ type: "text", text: "hello there" }]);
  assert.equal(assistant.api, OPTS.api);
  assert.equal(assistant.provider, OPTS.provider);
  assert.equal(assistant.model, OPTS.model);
  assert.deepEqual(assistant.usage, ZERO_USAGE);
  assert.equal(assistant.stopReason, "stop");
  assert.equal(assistant.timestamp, TS_MS);
});

test("tool_call pairs with the following tool_result of the same tool", () => {
  const { messages } = transcriptToMessages(
    [
      turn(1, "user", { content: "list files" }),
      turn(2, "tool_call", { tool: "bash", arguments_json: '{"command":"ls"}' }),
      turn(3, "tool_result", { tool: "bash", ok: true, output: "a.txt\nb.txt", error: "" }),
    ],
    OPTS,
  );
  assert.equal(messages.length, 3);

  const call = messages[1] as AssistantMessage;
  assert.equal(call.role, "assistant");
  assert.equal(call.stopReason, "toolUse");
  assert.deepEqual(call.usage, ZERO_USAGE);
  assert.deepEqual(call.content, [{ type: "toolCall", id: "hydrated-2", name: "bash", arguments: { command: "ls" } }]);

  const result = messages[2] as ToolResultMessage;
  assert.equal(result.role, "toolResult");
  assert.equal(result.toolCallId, "hydrated-2");
  assert.equal(result.toolName, "bash");
  assert.equal(result.isError, false);
  assert.deepEqual(result.content, [{ type: "text", text: "a.txt\nb.txt" }]);
  assert.equal(result.timestamp, TS_MS);
});

test("failed tool_result hydrates as isError, error text preferred over output", () => {
  const withError = transcriptToMessages(
    [
      turn(1, "tool_call", { tool: "bash", arguments_json: "{}" }),
      turn(2, "tool_result", { tool: "bash", ok: false, output: "", error: "boom" }),
    ],
    OPTS,
  ).messages;
  const failed = withError[1] as ToolResultMessage;
  assert.equal(failed.isError, true);
  assert.deepEqual(failed.content, [{ type: "text", text: "boom" }]);

  const withOutputOnly = transcriptToMessages(
    [
      turn(1, "tool_call", { tool: "bash", arguments_json: "{}" }),
      turn(2, "tool_result", { tool: "bash", ok: false, output: "partial output", error: "" }),
    ],
    OPTS,
  ).messages;
  const fallback = withOutputOnly[1] as ToolResultMessage;
  assert.equal(fallback.isError, true);
  assert.deepEqual(fallback.content, [{ type: "text", text: "partial output" }]);
});

test("malformed, unknown, and orphan entries are skipped", () => {
  const { messages } = transcriptToMessages(
    [
      turn(1, "user", "not json at all"),
      turn(2, "user", { unexpected: true }),
      turn(3, "weird_role", { content: "?" }),
      turn(4, "tool_result", { tool: "bash", ok: true, output: "orphan", error: "" }),
      turn(5, "tool_call", { tool: "bash", arguments_json: "{}" }), // orphan: next turn is a user message
      turn(6, "user", { content: "still here" }),
      turn(7, "tool_call", { tool: "bash", arguments_json: "{}" }), // pairs only with same tool
      turn(8, "tool_result", { tool: "read", ok: true, output: "different tool", error: "" }),
      turn(9, "tool_call", { tool: "bash", arguments_json: "{}" }), // malformed result follows
      turn(10, "tool_result", "{broken"),
    ],
    OPTS,
  );
  assert.equal(messages.length, 1);
  const user = messages[0] as UserMessage;
  assert.equal(user.role, "user");
  assert.deepEqual(user.content, [{ type: "text", text: "still here" }]);
});

test("tool_call with bad arguments_json falls back to {}", () => {
  const { messages } = transcriptToMessages(
    [
      turn(1, "tool_call", { tool: "bash", arguments_json: "{not json" }),
      turn(2, "tool_result", { tool: "bash", ok: true, output: "ok", error: "" }),
    ],
    OPTS,
  );
  assert.equal(messages.length, 2);
  const call = messages[0] as AssistantMessage;
  assert.deepEqual(call.content, [{ type: "toolCall", id: "hydrated-1", name: "bash", arguments: {} }]);
});

test("missing or invalid created_at falls back to now", () => {
  const before = Date.now();
  // Note: turn() defaults createdAt to TS, so the missing case is built literally.
  const missing: TranscriptTurn = { seq: 1, role: "user", contentJson: JSON.stringify({ content: "a" }) };
  const { messages } = transcriptToMessages([missing, turn(2, "user", { content: "b" }, "not a date")], OPTS);
  const after = Date.now();
  assert.equal(messages.length, 2);
  for (const m of messages) {
    assert.ok(m.timestamp >= before && m.timestamp <= after, `timestamp ${m.timestamp} within [${before}, ${after}]`);
  }
});

test("system turns surface as systemPrompt (last wins) and never become messages", () => {
  const result = transcriptToMessages(
    [
      turn(1, "system", { content: "You are a pirate tutor." }),
      turn(2, "user", { content: "hi" }),
      turn(3, "assistant", { content: "ahoy" }),
      turn(4, "system", { content: "You are a terse reviewer." }),
      turn(5, "user", { content: "again" }),
    ],
    OPTS,
  );
  assert.equal(result.systemPrompt, "You are a terse reviewer.");
  assert.deepEqual(
    result.messages.map((m) => m.role),
    ["user", "assistant", "user"],
  );
});

test("malformed system turns are skipped, earlier valid one still wins", () => {
  const result = transcriptToMessages(
    [
      turn(1, "system", { content: "kept" }),
      turn(2, "system", "not json at all"),
      turn(3, "system", { unexpected: true }),
    ],
    OPTS,
  );
  assert.equal(result.systemPrompt, "kept");
  assert.deepEqual(result.messages, []);
});

test("a transcript without system turns yields an empty systemPrompt", () => {
  const result = transcriptToMessages([turn(1, "user", { content: "hi" })], OPTS);
  assert.equal(result.systemPrompt, "");
  assert.equal(result.messages.length, 1);
});

test("assistant turns carrying a persisted usage payload hydrate, the extra key ignored", () => {
  const { messages } = transcriptToMessages(
    [turn(1, "assistant", { content: "hello", usage: { input_tokens: 3, output_tokens: 2 } })],
    OPTS,
  );
  assert.equal(messages.length, 1);
  const assistant = messages[0] as AssistantMessage;
  assert.deepEqual(assistant.content, [{ type: "text", text: "hello" }]);
  assert.deepEqual(assistant.usage, ZERO_USAGE);
});

test("config turns surface as llm (last wins) and never become messages", () => {
  const result = transcriptToMessages(
    [
      turn(1, "config", { llm: { api_key: "sk-old", base_url: "https://old.example/v1", model: "old-model" } }),
      turn(2, "user", { content: "hi" }),
      turn(3, "assistant", { content: "hello" }),
      turn(4, "config", { llm: { api_key: "sk-new", base_url: "https://new.example/v1", model: "new-model" } }),
      turn(5, "user", { content: "again" }),
    ],
    OPTS,
  );
  assert.deepEqual(result.llm, { api_key: "sk-new", base_url: "https://new.example/v1", model: "new-model" });
  assert.deepEqual(
    result.messages.map((m) => m.role),
    ["user", "assistant", "user"],
  );
});

test("malformed config turns are skipped, earlier valid one still wins; missing fields default to ''", () => {
  const result = transcriptToMessages(
    [
      turn(1, "config", { llm: { api_key: "sk-kept" } }), // base_url/model missing -> ""
      turn(2, "config", "not json at all"),
      turn(3, "config", { unexpected: true }),
      turn(4, "config", { llm: "not an object" }),
      turn(5, "config", { llm: { api_key: 42, base_url: null, model: "m" } }), // non-string fields -> ""
    ],
    OPTS,
  );
  // The last VALID config turn wins: seq 5 parses (with lenient defaults).
  assert.deepEqual(result.llm, { api_key: "", base_url: "", model: "m" });
  assert.deepEqual(result.messages, []);

  const onlyValid = transcriptToMessages(
    [turn(1, "config", { llm: { api_key: "sk-kept" } }), turn(2, "config", "{broken")],
    OPTS,
  );
  assert.deepEqual(onlyValid.llm, { api_key: "sk-kept", base_url: "", model: "" });
});

test("a transcript without config turns yields an all-empty llm", () => {
  const result = transcriptToMessages([turn(1, "user", { content: "hi" })], OPTS);
  assert.deepEqual(result.llm, { api_key: "", base_url: "", model: "" });
  assert.equal(result.messages.length, 1);
});

test("extractTranscriptLlm: last config turn wins, undefined without one, malformed skipped", () => {
  assert.equal(extractTranscriptLlm([turn(1, "user", { content: "hi" })]), undefined);
  assert.equal(extractTranscriptLlm([]), undefined);
  assert.equal(extractTranscriptLlm([turn(1, "config", "{broken")]), undefined);
  assert.deepEqual(
    extractTranscriptLlm([
      turn(1, "config", { llm: { api_key: "sk-1", base_url: "https://a/v1", model: "m1" } }),
      turn(2, "config", "not json"),
      turn(3, "config", { llm: { api_key: "sk-2", base_url: "https://b/v1", model: "m2" } }),
    ]),
    { api_key: "sk-2", base_url: "https://b/v1", model: "m2" },
  );
  // A key-less triple (api_key "") is still a valid extraction — callers
  // decide whether it takes over.
  assert.deepEqual(extractTranscriptLlm([turn(1, "config", { llm: { api_key: "", base_url: "", model: "" } })]), {
    api_key: "",
    base_url: "",
    model: "",
  });
});
