import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { createModel, createStreamFn, type SessionLlmConfig } from "../src/llm.js";

test("DeepSeek Flash disables default thinking without changing other endpoints", async () => {
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
    for (const host of ["api.deepseek.com", "other.example.com"]) {
      const config: SessionLlmConfig = { apiKey: "test", baseUrl: `http://127.0.0.1:${port}/v1`, model: "deepseek-flash" };
      const model = createModel(config);
      // The production guard uses the configured host; route both cases to
      // the local server so the outgoing JSON can be inspected.
      const endpoint = host === "api.deepseek.com" ? "https://api.deepseek.com" : "https://other.example.com";
      const stream = await createStreamFn({ ...config, baseUrl: endpoint })(
        { ...model, baseUrl: config.baseUrl },
        { messages: [{ role: "user", content: "Say OK", timestamp: Date.now() }] },
      );
      const reply = await stream.result();
      assert.equal(reply.stopReason, "stop");
    }
    assert.deepEqual((requests[0] as { thinking?: unknown }).thinking, { type: "disabled" });
    assert.equal((requests[1] as { thinking?: unknown }).thinking, undefined);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
