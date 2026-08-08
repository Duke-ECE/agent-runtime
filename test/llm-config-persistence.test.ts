import assert from "node:assert/strict";
import { createServer, type Server as HttpServer } from "node:http";
import { existsSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { createRuntime, type Runtime } from "../src/server.js";
import type { ServiceConfig } from "../src/config.js";

function loadProto(rel: string): any {
  for (const base of ["../../", "../"]) {
    const candidate = fileURLToPath(new URL(base + rel, import.meta.url));
    if (existsSync(candidate)) {
      const packageDefinition = protoLoader.loadSync(candidate, {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      });
      return grpc.loadPackageDefinition(packageDefinition);
    }
  }
  throw new Error(`${rel} not found`);
}

// --- Fake OpenAI-compatible LLM endpoint ------------------------------------

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

/** Serves a scripted streaming chat-completions reply per request (1-based index). */
async function startFakeLlm(replies: string[]): Promise<{ url: string; close: () => Promise<void> }> {
  let requests = 0;
  const server: HttpServer = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      const text = replies[Math.min(requests++, replies.length - 1)];
      const mid = Math.ceil(text.length / 2);
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(
        sse(
          chunk({ role: "assistant", content: text.slice(0, mid) }),
          chunk({ content: text.slice(mid) }),
          chunk({}, "stop", USAGE),
        ),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// --- Fake session.v1.SessionService ------------------------------------------

interface TranscriptMessage {
  seq: number;
  role: string;
  content_json: string;
  created_at: string;
}

interface AppendCall {
  session_id: string;
  user_id: string;
  messages: Array<{ seq: number; role: string; content_json: string; created_at: string }>;
}

const TS = "2026-01-02T03:04:05.000Z";

function transcriptMessage(seq: number, role: string, payload: unknown): TranscriptMessage {
  return { seq, role, content_json: JSON.stringify(payload), created_at: TS };
}

async function startFakeSessionManager(transcript: TranscriptMessage[]): Promise<{
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
    setTitle: unimplemented,
    getTranscript(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      callback(null, { messages: transcript });
    },
    appendTurn(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      appends.push({ ...call.request });
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
  opts: { replies: string[]; transcript?: TranscriptMessage[] | ((llmUrl: string) => TranscriptMessage[]) },
): Promise<{
  runtime: Runtime;
  llmUrl: string;
  createSession: (req: unknown) => Promise<{ session_id: string }>;
  chat: (req: unknown) => Promise<any[]>;
  appends: AppendCall[];
}> {
  const llm = await startFakeLlm(opts.replies);
  const transcript = typeof opts.transcript === "function" ? opts.transcript(llm.url) : (opts.transcript ?? []);
  const sessionManager = await startFakeSessionManager(transcript);

  const config: ServiceConfig = {
    port: 0,
    maxSessions: 20,
    sessionTtlMinutes: 30,
    llm: { apiKey: "env-key", baseUrl: llm.url, model: "env-model" },
    sessionManagerAddr: sessionManager.addr,
    serviceToken: "test-token",
    hydrationMaxTurns: 50,
  };
  // Stub the title generator so fresh-session titles don't consume scripted
  // LLM replies or interleave with the chat requests under test.
  const runtime = createRuntime(config, undefined, { titleGenerator: () => Promise.resolve("") });
  const port = await new Promise<number>((resolve, reject) => {
    runtime.server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p),
    );
  });
  const proto = loadProto("proto/runtime/v1/agent.proto");
  const raw = new proto.runtime.v1.AgentService(`127.0.0.1:${port}`, grpc.credentials.createInsecure()) as any;

  t.after(async () => {
    raw.close();
    await new Promise<void>((resolve) => runtime.server.tryShutdown(() => resolve()));
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
  return { runtime, llmUrl: llm.url, createSession, chat, appends: sessionManager.appends };
}

async function waitFor(cond: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!cond()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await new Promise((r) => setTimeout(r, 10));
  }
}

const roles = (append: AppendCall) => append.messages.map((m) => m.role);

/** The session's resolved LLM triple as the runtime sees it. */
function resolvedLlm(runtime: Runtime, sessionId: string): { apiKey?: string; baseUrl: string; model: string } {
  return runtime.sessions.get(sessionId).agent.llm;
}

test("fresh session records its frozen LLM triple on the first AppendTurn, then not again", async (t) => {
  const { runtime, llmUrl, createSession, chat, appends } = await setup(t, { replies: ["reply one", "reply two"] });
  const { session_id } = await createSession({
    user_id: "alice",
    llm: { api_key: "sk-custom", base_url: llmUrl, model: "custom-model" },
  });

  // The request llm wins over env for a fresh session.
  assert.deepEqual(resolvedLlm(runtime, session_id), { apiKey: "sk-custom", baseUrl: llmUrl, model: "custom-model" });

  await chat({ session_id, content: "hi", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  // No system prompt: the config turn leads the turn's messages.
  assert.deepEqual(roles(appends[0]), ["config", "user", "assistant"]);
  assert.deepEqual(JSON.parse(appends[0].messages[0].content_json), {
    llm: { api_key: "sk-custom", base_url: llmUrl, model: "custom-model" },
  });

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 2);
  // Unchanged triple: not re-recorded.
  assert.deepEqual(roles(appends[1]), ["user", "assistant"]);
});

test("resume: transcript config beats the request llm (frozen triple wins)", async (t) => {
  const { runtime, llmUrl, createSession, chat, appends } = await setup(t, {
    replies: ["reply one"],
    transcript: (url) => [
      transcriptMessage(1, "config", { llm: { api_key: "sk-frozen", base_url: url, model: "frozen-model" } }),
      transcriptMessage(2, "user", { content: "hi" }),
      transcriptMessage(3, "assistant", { content: "hello" }),
    ],
  });
  const { session_id } = await createSession({
    user_id: "alice",
    session_id: "sess-frozen",
    llm: { api_key: "sk-ignored", base_url: "http://127.0.0.1:1/v1", model: "ignored-model" },
  });

  // The transcript's frozen triple wins over the request llm entirely.
  assert.deepEqual(resolvedLlm(runtime, session_id), {
    apiKey: "sk-frozen",
    baseUrl: llmUrl,
    model: "frozen-model",
  });
  assert.equal(runtime.sessions.get(session_id).agent.agent.state.model.id, "frozen-model");

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  // Marker seeded from the transcript: unchanged triple is not re-recorded.
  assert.deepEqual(roles(appends[0]), ["user", "assistant"]);
});

test("resume: request llm is used when the transcript has no config turn", async (t) => {
  const { runtime, llmUrl, createSession, chat, appends } = await setup(t, {
    replies: ["reply one"],
    transcript: [transcriptMessage(1, "user", { content: "hi" })],
  });
  const { session_id } = await createSession({
    user_id: "alice",
    session_id: "sess-pre-config",
    llm: { api_key: "sk-request", base_url: llmUrl, model: "request-model" },
  });

  assert.deepEqual(resolvedLlm(runtime, session_id), {
    apiKey: "sk-request",
    baseUrl: llmUrl,
    model: "request-model",
  });

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  // Pre-config-turn transcript: the resolved triple is recorded once.
  assert.deepEqual(roles(appends[0]), ["config", "user", "assistant"]);
  assert.deepEqual(JSON.parse(appends[0].messages[0].content_json), {
    llm: { api_key: "sk-request", base_url: llmUrl, model: "request-model" },
  });
});

test("resume: a config turn with an empty api_key does not override a keyed request llm", async (t) => {
  const { runtime, llmUrl, createSession, chat, appends } = await setup(t, {
    replies: ["reply one"],
    transcript: [
      transcriptMessage(1, "config", { llm: { api_key: "", base_url: "", model: "" } }),
      transcriptMessage(2, "user", { content: "hi" }),
    ],
  });
  const { session_id } = await createSession({
    user_id: "alice",
    session_id: "sess-empty-frozen",
    llm: { api_key: "sk-request", base_url: llmUrl, model: "request-model" },
  });

  // Empty frozen key: the request llm resolution takes over.
  assert.deepEqual(resolvedLlm(runtime, session_id), {
    apiKey: "sk-request",
    baseUrl: llmUrl,
    model: "request-model",
  });

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  // The recorded triple (all-empty) differs from the resolved one: re-recorded.
  assert.deepEqual(roles(appends[0]), ["config", "user", "assistant"]);
  assert.deepEqual(JSON.parse(appends[0].messages[0].content_json), {
    llm: { api_key: "sk-request", base_url: llmUrl, model: "request-model" },
  });
});

test("template LLM changed after the session started: resume keeps the ORIGINAL triple", async (t) => {
  const { runtime, llmUrl, createSession, chat, appends } = await setup(t, {
    replies: ["reply one"],
    transcript: (url) => [
      transcriptMessage(1, "config", { llm: { api_key: "sk-original", base_url: url, model: "original-model" } }),
      transcriptMessage(2, "user", { content: "hi" }),
      transcriptMessage(3, "assistant", { content: "hello" }),
    ],
  });
  // The template's LLM was edited since: the backend re-resolves the template
  // and sends its CURRENT triple on resume. It must be ignored.
  const { session_id } = await createSession({
    user_id: "alice",
    session_id: "sess-template-edited",
    llm: { api_key: "sk-changed", base_url: "http://127.0.0.1:1/v1", model: "changed-model" },
  });

  assert.deepEqual(resolvedLlm(runtime, session_id), {
    apiKey: "sk-original",
    baseUrl: llmUrl,
    model: "original-model",
  });
  // Hydrated history is stamped with the frozen model's identity too.
  const state = runtime.sessions.get(session_id).agent.agent.state;
  assert.equal(state.model.id, "original-model");
  assert.deepEqual(
    (state.messages as Array<{ role: string; model?: string }>).map((m) => [m.role, m.model]),
    [
      ["user", undefined],
      ["assistant", "original-model"],
    ],
  );

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  assert.deepEqual(roles(appends[0]), ["user", "assistant"]);
});
