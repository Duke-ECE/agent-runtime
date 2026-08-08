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
  opts: { replies: string[]; transcript?: TranscriptMessage[] },
): Promise<{
  createSession: (req: unknown) => Promise<{ session_id: string }>;
  chat: (req: unknown) => Promise<any[]>;
  appends: AppendCall[];
}> {
  const llm = await startFakeLlm(opts.replies);
  const sessionManager = await startFakeSessionManager(opts.transcript ?? []);

  const config: ServiceConfig = {
    port: 0,
    maxSessions: 20,
    sessionTtlMinutes: 30,
    llm: { apiKey: "test-key", baseUrl: llm.url, model: "test-model" },
    sessionManagerAddr: sessionManager.addr,
    serviceToken: "test-token",
  };
  // Stub the title generator so fresh-session titles don't consume scripted
  // LLM replies or interleave with the chat requests under test.
  const { server } = createRuntime(config, undefined, { titleGenerator: () => Promise.resolve("") });
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
  return { createSession, chat, appends: sessionManager.appends };
}

async function waitFor(cond: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!cond()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await new Promise((r) => setTimeout(r, 10));
  }
}

const roles = (append: AppendCall) => append.messages.map((m) => m.role);

test("fresh session with a system prompt records it once, before the first turn's messages", async (t) => {
  const { createSession, chat, appends } = await setup(t, { replies: ["reply one", "reply two"] });
  const { session_id } = await createSession({ user_id: "alice", system_prompt: "You are a pirate tutor." });

  await chat({ session_id, content: "hi", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  assert.deepEqual(roles(appends[0]), ["system", "user", "assistant"]);
  assert.deepEqual(JSON.parse(appends[0].messages[0].content_json), { content: "You are a pirate tutor." });

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 2);
  // Unchanged prompt: not re-recorded.
  assert.deepEqual(roles(appends[1]), ["user", "assistant"]);
});

test("session with an empty system prompt never records a system message", async (t) => {
  const { createSession, chat, appends } = await setup(t, { replies: ["reply one", "reply two"] });
  const { session_id } = await createSession({ user_id: "alice", system_prompt: "" });

  await chat({ session_id, content: "hi", user_id: "alice" });
  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 2);
  for (const append of appends) {
    assert.deepEqual(roles(append), ["user", "assistant"]);
  }
});

test("resumed session with an unchanged prompt does not re-record the transcript's system message", async (t) => {
  const { createSession, chat, appends } = await setup(t, {
    replies: ["reply one"],
    transcript: [
      transcriptMessage(1, "system", { content: "You are a pirate tutor." }),
      transcriptMessage(2, "user", { content: "hi" }),
      transcriptMessage(3, "assistant", { content: "ahoy" }),
    ],
  });
  // Template deleted: the backend sends no system_prompt on resume.
  const { session_id } = await createSession({ user_id: "alice", session_id: "sess-resumed-same" });
  assert.equal(session_id, "sess-resumed-same");

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  assert.deepEqual(roles(appends[0]), ["user", "assistant"]);
});

test("resumed session whose request prompt differs from the recorded one re-records it on the next turn", async (t) => {
  const { createSession, chat, appends } = await setup(t, {
    replies: ["reply one", "reply two"],
    transcript: [
      transcriptMessage(1, "system", { content: "old persona" }),
      transcriptMessage(2, "user", { content: "hi" }),
      transcriptMessage(3, "assistant", { content: "ahoy" }),
    ],
  });
  // Template edited since: the backend sends the new prompt on resume.
  const { session_id } = await createSession({
    user_id: "alice",
    session_id: "sess-resumed-edited",
    system_prompt: "new persona",
  });

  await chat({ session_id, content: "again", user_id: "alice" });
  await waitFor(() => appends.length === 1);
  assert.deepEqual(roles(appends[0]), ["system", "user", "assistant"]);
  assert.deepEqual(JSON.parse(appends[0].messages[0].content_json), { content: "new persona" });

  await chat({ session_id, content: "once more", user_id: "alice" });
  await waitFor(() => appends.length === 2);
  // Now the recorded prompt matches: not repeated.
  assert.deepEqual(roles(appends[1]), ["user", "assistant"]);
});
