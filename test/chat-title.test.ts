import assert from "node:assert/strict";
import { createServer, type Server as HttpServer } from "node:http";
import { existsSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { createRuntime, type RuntimeOptions } from "../src/server.js";
import type { ServiceConfig } from "../src/config.js";
import { sanitizeTitle } from "../src/title.js";

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
  return { sse: sse(chunk({ role: "assistant", content: text }), chunk({}, "stop", USAGE)) };
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

interface TitleCall {
  session_id: string;
  title: string;
  token?: string;
}

async function startFakeSessionManager(opts: {
  transcript?: Array<{ seq: number; role: string; content_json: string; created_at: string }>;
  setTitleError?: grpc.status;
} = {}): Promise<{
  addr: string;
  titles: TitleCall[];
  close: () => Promise<void>;
}> {
  const titles: TitleCall[] = [];
  const proto = loadProto("proto/session/v1/session.proto");
  const server = new grpc.Server();
  const unimplemented = (call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) =>
    callback(Object.assign(new Error("unimplemented"), { code: grpc.status.UNIMPLEMENTED }), null);
  server.addService(proto.session.v1.SessionService.service, {
    createSession: unimplemented,
    getSession: unimplemented,
    listSessions: unimplemented,
    endSession: unimplemented,
    appendTurn(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      callback(null, {});
    },
    getTranscript(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      callback(null, { messages: opts.transcript ?? [] });
    },
    setTitle(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const token = call.metadata.get("x-service-token")[0];
      titles.push({ ...call.request, token: typeof token === "string" ? token : undefined });
      if (opts.setTitleError !== undefined) {
        callback(Object.assign(new Error("setTitle unavailable"), { code: opts.setTitleError }), null);
        return;
      }
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
    titles,
    close: () => new Promise((resolve) => server.tryShutdown(() => resolve())),
  };
}

// --- Runtime under test ------------------------------------------------------

async function setup(
  t: import("node:test").TestContext,
  script: (requestIndex: number) => LlmReply,
  opts: {
    runtimeOptions?: RuntimeOptions;
    transcript?: Array<{ seq: number; role: string; content_json: string; created_at: string }>;
    setTitleError?: grpc.status;
  } = {},
): Promise<{
  createSession: (req: unknown) => Promise<{ session_id: string }>;
  chat: (req: unknown) => Promise<any[]>;
  titles: TitleCall[];
  llmRequests: () => number;
}> {
  const llm = await startFakeLlm(script);
  const sessionManager = await startFakeSessionManager({ transcript: opts.transcript, setTitleError: opts.setTitleError });

  const config: ServiceConfig = {
    port: 0,
    maxSessions: 20,
    sessionTtlMinutes: 30,
    llm: { apiKey: "test-key", baseUrl: llm.url, model: "test-model" },
    sessionManagerAddr: sessionManager.addr,
    serviceToken: "test-token",
    hydrationMaxTurns: 50,
  };
  const { server } = createRuntime(config, undefined, opts.runtimeOptions);
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
    await sessionManager.close();
    await llm.close();
  });

  const createSession = (req: unknown): Promise<{ session_id: string }> =>
    new Promise((resolve, reject) => {
      raw.createSession(req, (err: grpc.ServiceError | null, res: { session_id: string }) =>
        err ? reject(err) : resolve(res),
      );
    });

  const chat = (req: unknown): Promise<any[]> =>
    new Promise((resolve, reject) => {
      const events: any[] = [];
      raw.chat(req)
        .on("data", (e: unknown) => events.push(e))
        .on("end", () => resolve(events))
        .on("error", reject);
    });

  return { createSession, chat, titles: sessionManager.titles, llmRequests: llm.requests };
}

async function waitFor(cond: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!cond()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await new Promise((r) => setTimeout(r, 10));
  }
}

/** Give a (buggy) async side effect a chance to fire before asserting it did not. */
function settle(ms = 200): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

test("first completed turn sets a sanitized title exactly once", async (t) => {
  const titleCalls: string[] = [];
  const { createSession, chat, titles } = await setup(t, () => textReply("Hello world"), {
    runtimeOptions: {
      titleGenerator: (_llm, firstUserMessage) => {
        titleCalls.push(firstUserMessage);
        return Promise.resolve('  "Fix the flaky test"\nextra line');
      },
    },
  });
  const { session_id } = await createSession({ user_id: "alice" });

  const events = await chat({ session_id, content: "help me fix the flaky test", user_id: "alice" });
  assert.ok(events.find((e) => e.done));

  await waitFor(() => titles.length === 1);
  assert.deepEqual(titleCalls, ["help me fix the flaky test"]);
  assert.equal(titles[0].session_id, session_id);
  assert.equal(titles[0].title, "Fix the flaky test");
  assert.equal(titles[0].token, "test-token");

  // Second turn must not re-title.
  const second = await chat({ session_id, content: "and the other one", user_id: "alice" });
  assert.ok(second.find((e) => e.done));
  await settle();
  assert.equal(titles.length, 1);
  assert.equal(titleCalls.length, 1);
});

test("hydrated session never gets an auto title", async (t) => {
  let titleCalls = 0;
  const { createSession, chat, titles } = await setup(t, () => textReply("welcome back"), {
    runtimeOptions: {
      titleGenerator: () => {
        titleCalls++;
        return Promise.resolve("should not be used");
      },
    },
    transcript: [
      { seq: 1, role: "user", content_json: JSON.stringify({ content: "earlier question" }), created_at: "2026-01-02T03:04:05.000Z" },
      { seq: 2, role: "assistant", content_json: JSON.stringify({ content: "earlier answer" }), created_at: "2026-01-02T03:04:06.000Z" },
    ],
  });
  const { session_id } = await createSession({ user_id: "alice", session_id: "sess-hydrated" });
  assert.equal(session_id, "sess-hydrated");

  const events = await chat({ session_id, content: "one more thing", user_id: "alice" });
  assert.ok(events.find((e) => e.done));
  await settle();
  assert.equal(titleCalls, 0);
  assert.equal(titles.length, 0);
});

test("title generation failure leaves the chat turn unaffected", async (t) => {
  const { createSession, chat, titles } = await setup(t, () => textReply("still works"), {
    runtimeOptions: { titleGenerator: () => Promise.reject(new Error("title llm boom")) },
  });
  const { session_id } = await createSession({ user_id: "alice" });

  const events = await chat({ session_id, content: "hi", user_id: "alice" });
  assert.equal(events.filter((e) => e.text_delta).map((e) => e.text_delta.delta).join(""), "still works");
  assert.ok(events.find((e) => e.done));
  await settle();
  assert.equal(titles.length, 0);
});

test("session-manager SetTitle failure leaves the chat turn unaffected", async (t) => {
  const { createSession, chat, titles } = await setup(t, () => textReply("fine anyway"), {
    runtimeOptions: { titleGenerator: () => Promise.resolve("A Title") },
    setTitleError: grpc.status.UNAVAILABLE,
  });
  const { session_id } = await createSession({ user_id: "alice" });

  const events = await chat({ session_id, content: "hi", user_id: "alice" });
  assert.equal(events.filter((e) => e.text_delta).map((e) => e.text_delta.delta).join(""), "fine anyway");
  assert.ok(events.find((e) => e.done));

  await waitFor(() => titles.length === 1);
  assert.equal(titles[0].title, "A Title");
});

test("failed first turn is not titled; the next completed turn is", async (t) => {
  let titleCalls = 0;
  const { createSession, chat, titles } = await setup(
    t,
    (i) => (i === 1 ? { status: 500 } : textReply("recovered")),
    {
      runtimeOptions: {
        titleGenerator: () => {
          titleCalls++;
          return Promise.resolve("Recovered Chat");
        },
      },
    },
  );
  const { session_id } = await createSession({ user_id: "alice" });

  const failed = await chat({ session_id, content: "hi", user_id: "alice" });
  assert.ok(failed[0].error);
  await settle();
  assert.equal(titleCalls, 0);
  assert.equal(titles.length, 0);

  const events = await chat({ session_id, content: "again", user_id: "alice" });
  assert.ok(events.find((e) => e.done));
  await waitFor(() => titles.length === 1);
  assert.equal(titleCalls, 1);
  assert.equal(titles[0].title, "Recovered Chat");
});

test("default title generator uses the session LLM and its output is sanitized", async (t) => {
  // Request 1 is the chat turn; request 2 is the one-shot title call.
  const { createSession, chat, titles, llmRequests } = await setup(t, (i) =>
    i === 1 ? textReply("chat answer") : textReply('"How to fix flaky tests"'),
  );
  const { session_id } = await createSession({ user_id: "alice" });

  const events = await chat({ session_id, content: "how do I fix flaky tests?", user_id: "alice" });
  assert.ok(events.find((e) => e.done));

  await waitFor(() => titles.length === 1);
  assert.equal(llmRequests(), 2);
  assert.equal(titles[0].session_id, session_id);
  assert.equal(titles[0].title, "How to fix flaky tests");
});

test("sanitizeTitle normalizes raw LLM output", () => {
  assert.equal(sanitizeTitle("  Hello World  \n"), "Hello World");
  assert.equal(sanitizeTitle("first line\nsecond line"), "first line");
  assert.equal(sanitizeTitle('"Quoted Title"'), "Quoted Title");
  assert.equal(sanitizeTitle("'single quoted'"), "single quoted");
  assert.equal(sanitizeTitle("“curly”"), "curly");
  assert.equal(sanitizeTitle("   \n  "), "");
  assert.equal(sanitizeTitle('""'), "");
  const long = "x".repeat(100);
  assert.equal(sanitizeTitle(long).length, 80);
});
