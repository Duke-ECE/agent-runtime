import assert from "node:assert/strict";
import { createServer, type Server as HttpServer } from "node:http";
import { existsSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { createRuntime } from "../src/server.js";
import type { ServiceConfig } from "../src/config.js";
import { SANDBOX_NOT_CONNECTED } from "../src/tools.js";

function resolveProtoPath(rel: string): string {
  for (const base of ["../../", "../"]) {
    const candidate = fileURLToPath(new URL(base + rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(`${rel} not found`);
}

function loadProto(rel: string): any {
  const packageDefinition = protoLoader.loadSync(resolveProtoPath(rel), {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  return grpc.loadPackageDefinition(packageDefinition);
}

// --- Fake OpenAI-compatible LLM endpoint ------------------------------------

type LlmReply = { sse: string } | { status: number };

function chunk(delta: Record<string, unknown>, finishReason: string | null = null, usage?: unknown) {
  return {
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage ? { usage } : {}),
  };
}

const USAGE = { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 };

function sse(...parts: unknown[]): string {
  return parts.map((p) => `data: ${JSON.stringify(p)}\n\n`).join("") + "data: [DONE]\n\n";
}

function textReply(text: string): LlmReply {
  const mid = Math.ceil(text.length / 2);
  return {
    sse: sse(
      chunk({ role: "assistant", content: text.slice(0, mid) }),
      chunk({ content: text.slice(mid) }),
      chunk({}, "stop", USAGE),
    ),
  };
}

function toolCallReply(): LlmReply {
  return {
    sse: sse(
      chunk({
        role: "assistant",
        tool_calls: [
          { index: 0, id: "call_1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } },
        ],
      }),
      chunk({}, "tool_calls", USAGE),
    ),
  };
}

/** Serves scripted streaming chat-completions responses; script gets the 1-based request index. */
async function startFakeLlm(script: (requestIndex: number) => LlmReply): Promise<{
  url: string;
  requests: () => number;
  close: () => Promise<void>;
}> {
  let requests = 0;
  const server: HttpServer = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      const reply = script(++requests);
      if ("status" in reply) {
        res.writeHead(reply.status, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "llm boom" } }));
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(reply.sse);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    requests: () => requests,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// --- Fake session.v1.SessionService ------------------------------------------

interface AppendCall {
  session_id: string;
  user_id: string;
  messages: Array<{ seq: number; role: string; content_json: string; created_at: string }>;
  token?: string;
}

async function startFakeSessionManager(): Promise<{
  addr: string;
  appends: AppendCall[];
  close: () => Promise<void>;
}> {
  const appends: AppendCall[] = [];
  const proto = loadProto("proto/session/v1/session.proto");
  const server = new grpc.Server();
  const unimplemented = (call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) =>
    callback(Object.assign(new Error("unimplemented"), { code: grpc.status.UNIMPLEMENTED }), null);
  server.addService(proto.session.v1.SessionService.service, {
    createSession: unimplemented,
    getSession: unimplemented,
    listSessions: unimplemented,
    endSession: unimplemented,
    getTranscript: unimplemented,
    appendTurn(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const token = call.metadata.get("x-service-token")[0];
      appends.push({ ...call.request, token: typeof token === "string" ? token : undefined });
      callback(null, {});
    },
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p),
    );
  });
  return {
    addr: `127.0.0.1:${port}`,
    appends,
    close: () => new Promise((resolve) => server.tryShutdown(() => resolve())),
  };
}

// --- Runtime under test ------------------------------------------------------

async function setup(
  t: import("node:test").TestContext,
  script: (requestIndex: number) => LlmReply,
  opts: { withSessionManager: boolean },
): Promise<{ chat: (req: unknown) => Promise<any[]>; sessionId: string; appends: AppendCall[] }> {
  const llm = await startFakeLlm(script);
  const sessionManager = opts.withSessionManager ? await startFakeSessionManager() : undefined;

  const config: ServiceConfig = {
    port: 0,
    maxSessions: 20,
    sessionTtlMinutes: 30,
    llm: { apiKey: "test-key", baseUrl: llm.url, model: "test-model" },
    sessionManagerAddr: sessionManager?.addr,
    serviceToken: opts.withSessionManager ? "test-token" : undefined,
  };
  const { server } = createRuntime(config);
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p),
    );
  });
  const proto = loadProto("proto/runtime/v1/agent.proto");
  const raw = new proto.runtime.v1.AgentService(`127.0.0.1:${port}`, grpc.credentials.createInsecure()) as any;

  t.after(async () => {
    raw.close();
    await new Promise<void>((resolve) => server.tryShutdown(() => resolve()));
    await sessionManager?.close();
    await llm.close();
  });

  const session = (await new Promise<{ session_id: string }>((resolve, reject) => {
    raw.createSession({ user_id: "alice" }, (err: grpc.ServiceError | null, res: { session_id: string }) =>
      err ? reject(err) : resolve(res),
    );
  }));

  const chat = (req: unknown): Promise<any[]> =>
    new Promise((resolve, reject) => {
      const events: any[] = [];
      raw.chat(req)
        .on("data", (e: unknown) => events.push(e))
        .on("end", () => resolve(events))
        .on("error", reject);
    });

  return { chat, sessionId: session.session_id, appends: sessionManager?.appends ?? [] };
}

async function waitFor(cond: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!cond()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await new Promise((r) => setTimeout(r, 10));
  }
}

test("successful chat turn is written through to session-manager once", async (t) => {
  const { chat, sessionId, appends } = await setup(t, () => textReply("Hello world"), { withSessionManager: true });

  const events = await chat({ session_id: sessionId, content: "hi", user_id: "alice" });
  const deltas = events.filter((e) => e.text_delta).map((e) => e.text_delta.delta);
  assert.equal(deltas.join(""), "Hello world");
  const done = events.find((e) => e.done)?.done;
  assert.ok(done, "expected a done event");
  assert.equal(Number(done.input_tokens), 3);
  assert.equal(Number(done.output_tokens), 2);

  await waitFor(() => appends.length === 1);
  assert.equal(appends.length, 1);
  const append = appends[0];
  assert.equal(append.session_id, sessionId);
  assert.equal(append.user_id, "alice");
  assert.equal(append.token, "test-token");
  assert.deepEqual(
    append.messages.map((m) => m.role),
    ["user", "assistant"],
  );
  assert.deepEqual(
    append.messages.map((m) => JSON.parse(m.content_json)),
    [{ content: "hi" }, { content: "Hello world" }],
  );
  for (const m of append.messages) {
    assert.equal(m.seq, 0); // assigned server-side
    assert.ok(m.created_at);
  }
});

test("tool events are recorded as tool_call/tool_result messages", async (t) => {
  const { chat, sessionId, appends } = await setup(
    t,
    (i) => (i === 1 ? toolCallReply() : textReply("all done")),
    { withSessionManager: true },
  );

  const events = await chat({ session_id: sessionId, content: "list files", user_id: "alice" });
  assert.ok(events.find((e) => e.tool_call));
  assert.ok(events.find((e) => e.tool_result));
  assert.ok(events.find((e) => e.done));

  await waitFor(() => appends.length === 1);
  assert.equal(appends.length, 1);
  const messages = appends[0].messages;
  assert.deepEqual(
    messages.map((m) => m.role),
    ["user", "assistant", "tool_call", "tool_result"],
  );
  assert.deepEqual(JSON.parse(messages[2].content_json), { tool: "bash", arguments_json: '{"command":"ls"}' });
  assert.deepEqual(JSON.parse(messages[3].content_json), {
    tool: "bash",
    ok: false,
    output: "",
    error: SANDBOX_NOT_CONNECTED,
  });
});

test("failed turn is not written through", async (t) => {
  const { chat, sessionId, appends } = await setup(t, () => ({ status: 500 }), { withSessionManager: true });

  const events = await chat({ session_id: sessionId, content: "hi", user_id: "alice" });
  assert.equal(events.length, 1);
  assert.ok(events[0].error, "expected an error event");

  // Give a (buggy) append a chance to arrive before asserting none did.
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(appends.length, 0);
});

test("chat works with SESSION_MANAGER_ADDR unset", async (t) => {
  const { chat, sessionId } = await setup(t, () => textReply("no writer"), { withSessionManager: false });

  const events = await chat({ session_id: sessionId, content: "hi", user_id: "alice" });
  assert.equal(events.filter((e) => e.text_delta).map((e) => e.text_delta.delta).join(""), "no writer");
  assert.ok(events.find((e) => e.done));
});
