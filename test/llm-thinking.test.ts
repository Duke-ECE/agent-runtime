import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { createModel, createStreamFn, resolveThinkingMode, type SessionLlmConfig } from "../src/llm.js";

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
