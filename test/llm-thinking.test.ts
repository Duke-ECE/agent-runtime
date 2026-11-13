import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { createModel, createStreamFn, resolveThinkingMode, type SessionLlmConfig } from "../src/llm.js";
import { transcriptToMessages, ZERO_USAGE } from "../src/hydrate.js";

test("DeepSeek Flash maps each session thinking mode into the request body", async () => {
  const requests: unknown[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (part) => (body += part));
    req.on("end", () => {
      requests.push(JSON.parse(body));
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(
        'data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n\n' +
          'data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n' +
          "data: [DONE]\n\n",
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address() as AddressInfo;
    for (const mode of [undefined, "off", "low", "high", "max", "provider_default"] as const) {
      const config: SessionLlmConfig = { apiKey: "test", baseUrl: `http://127.0.0.1:${port}/v1`, model: "deepseek-flash" };
      const model = createModel(config);
      // Keep the configured DeepSeek host while routing the model connection
      // to the local server so the outgoing JSON can be inspected.
      const stream = await createStreamFn({ ...config, baseUrl: "https://api.deepseek.com", thinkingMode: mode })(
        { ...model, baseUrl: config.baseUrl },
        { messages: [{ role: "user", content: "Say OK", timestamp: Date.now() }] },
      );
      const reply = await stream.result();
      assert.equal(reply.stopReason, "stop");
    }
    const sent = requests as Array<{ thinking?: unknown; reasoning_effort?: string }>;
    assert.deepEqual(sent.slice(0, 2).map((r) => r.thinking), [{ type: "disabled" }, { type: "disabled" }]);
    for (const [index, mode] of [[2, "low"], [3, "high"], [4, "max"]] as const) {
      assert.deepEqual(sent[index].thinking, { type: "enabled" });
      assert.equal(sent[index].reasoning_effort, mode);
    }
    assert.equal(sent[5].thinking, undefined);
    assert.equal(sent[5].reasoning_effort, undefined);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("unsupported explicit modes fail instead of silently changing providers", () => {
  const other = { baseUrl: "https://other.example.com/v1", model: "deepseek-flash", thinkingMode: "low" };
  assert.throws(() => resolveThinkingMode(other), /not supported/);
  assert.equal(resolveThinkingMode({ ...other, thinkingMode: undefined }), "provider_default");
  assert.throws(() => resolveThinkingMode({ ...other, thinkingMode: "invalid" }), /invalid thinking_mode/);
});

test("hydrated private state replays reasoning with tool calls", async () => {
  let sent: any;
  const server = createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (part) => (body += part));
    req.on("end", () => {
      sent = JSON.parse(body);
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end('data: {"id":"test","object":"chat.completion.chunk","created":0,"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address() as AddressInfo;
    const config: SessionLlmConfig = { apiKey: "test", baseUrl: "https://api.deepseek.com", model: "deepseek-flash", thinkingMode: "low" };
    const assistant = {
      role: "assistant", api: "openai-completions", provider: "openai-compatible", model: "deepseek-flash",
      content: [
        { type: "thinking", thinking: "check files first", thinkingSignature: "reasoning_content" },
        { type: "toolCall", id: "call-1", name: "bash", arguments: { command: "ls" } },
      ], usage: ZERO_USAGE, stopReason: "toolUse", timestamp: Date.now(),
    };
    const result = { role: "toolResult", toolCallId: "call-1", toolName: "bash", content: [{ type: "text", text: "a.txt" }], isError: false, timestamp: Date.now() };
    const turns = [
      { seq: 1, role: "user", contentJson: JSON.stringify({ content: "list files" }) },
      { seq: 2, role: "assistant", contentJson: JSON.stringify({ content: "done" }) },
      { seq: 3, role: "agent_state", contentJson: JSON.stringify({ message: assistant }) },
      { seq: 4, role: "agent_state", contentJson: JSON.stringify({ message: result }) },
    ];
    const history = transcriptToMessages(turns, { api: "openai-completions", provider: "openai-compatible", model: "deepseek-flash" }).messages;
    const model = { ...createModel(config), baseUrl: `http://127.0.0.1:${port}/v1` };
    const stream = await createStreamFn(config)(model, { messages: [...history, { role: "user", content: "continue", timestamp: Date.now() }] as any });
    await stream.result();
    const replayed = sent.messages.find((m: any) => m.role === "assistant" && m.tool_calls);
    assert.equal(replayed.reasoning_content, "check files first");
    assert.equal(replayed.tool_calls[0].id, "call-1");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
