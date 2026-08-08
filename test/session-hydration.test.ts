import assert from "node:assert/strict";
import { existsSync } from "node:fs";
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

// --- Fake session.v1.SessionService ------------------------------------------

interface TranscriptMessage {
  seq: number;
  role: string;
  content_json: string;
  created_at: string;
}

interface TranscriptRequest {
  session_id: string;
  token?: string;
}

async function startFakeSessionManager(opts: {
  transcript?: TranscriptMessage[];
  transcriptError?: grpc.status;
}): Promise<{
  addr: string;
  transcriptRequests: TranscriptRequest[];
  close: () => Promise<void>;
}> {
  const transcriptRequests: TranscriptRequest[] = [];
  const proto = loadProto("proto/session/v1/session.proto");
  const server = new grpc.Server();
  const unimplemented = (call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) =>
    callback(Object.assign(new Error("unimplemented"), { code: grpc.status.UNIMPLEMENTED }), null);
  server.addService(proto.session.v1.SessionService.service, {
    createSession: unimplemented,
    getSession: unimplemented,
    listSessions: unimplemented,
    endSession: unimplemented,
    appendTurn: unimplemented,
    getTranscript(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const token = call.metadata.get("x-service-token")[0];
      transcriptRequests.push({
        session_id: call.request.session_id,
        token: typeof token === "string" ? token : undefined,
      });
      if (opts.transcriptError !== undefined) {
        callback(Object.assign(new Error("transcript unavailable"), { code: opts.transcriptError }), null);
        return;
      }
      callback(null, { messages: opts.transcript ?? [] });
    },
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p),
    );
  });
  return {
    addr: `127.0.0.1:${port}`,
    transcriptRequests,
    close: () => new Promise((resolve) => server.tryShutdown(() => resolve())),
  };
}

// --- Runtime under test ------------------------------------------------------

async function setup(
  t: import("node:test").TestContext,
  sessionManagerAddr: string,
): Promise<{
  runtime: Runtime;
  createSession: (req: unknown) => Promise<{ session_id: string }>;
}> {
  const config: ServiceConfig = {
    port: 0,
    maxSessions: 20,
    sessionTtlMinutes: 30,
    llm: { apiKey: "test-key", baseUrl: "http://127.0.0.1:1/v1", model: "test-model" },
    sessionManagerAddr,
    serviceToken: "test-token",
  };
  const runtime = createRuntime(config);
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
  });

  const createSession = (req: unknown): Promise<{ session_id: string }> =>
    new Promise((resolve, reject) => {
      raw.createSession(req, (err: grpc.ServiceError | null, res: { session_id: string }) =>
        err ? reject(err) : resolve(res),
      );
    });
  return { runtime, createSession };
}

const TS = "2026-01-02T03:04:05.000Z";

function transcriptMessage(seq: number, role: string, payload: unknown): TranscriptMessage {
  return { seq, role, content_json: JSON.stringify(payload), created_at: TS };
}

test("CreateSession hydrates the agent from a durable transcript", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(1, "user", { content: "hi" }),
      transcriptMessage(2, "assistant", { content: "hello there" }),
      transcriptMessage(3, "tool_call", { tool: "bash", arguments_json: '{"command":"ls"}' }),
      transcriptMessage(4, "tool_result", { tool: "bash", ok: false, output: "", error: "boom" }),
    ],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-hydrated" });
  assert.equal(created.session_id, "sess-hydrated");

  assert.deepEqual(sessionManager.transcriptRequests, [{ session_id: "sess-hydrated", token: "test-token" }]);

  const messages = runtime.sessions.get("sess-hydrated").agent.agent.state.messages as any[];
  assert.deepEqual(
    messages.map((m) => m.role),
    ["user", "assistant", "assistant", "toolResult"],
  );
  assert.deepEqual(messages[0].content, [{ type: "text", text: "hi" }]);
  assert.equal(messages[0].timestamp, Date.parse(TS));
  assert.deepEqual(messages[1].content, [{ type: "text", text: "hello there" }]);
  assert.equal(messages[1].stopReason, "stop");
  assert.equal(messages[1].api, "openai-completions");
  assert.equal(messages[1].provider, "openai-compatible");
  assert.equal(messages[1].model, "test-model");
  assert.deepEqual(messages[2].content, [
    { type: "toolCall", id: "hydrated-3", name: "bash", arguments: { command: "ls" } },
  ]);
  assert.equal(messages[2].stopReason, "toolUse");
  assert.equal(messages[3].toolCallId, "hydrated-3");
  assert.equal(messages[3].toolName, "bash");
  assert.equal(messages[3].isError, true);
  assert.deepEqual(messages[3].content, [{ type: "text", text: "boom" }]);
});

test("CreateSession starts empty when the transcript fetch fails (fail-open)", async (t) => {
  const sessionManager = await startFakeSessionManager({ transcriptError: grpc.status.NOT_FOUND });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-gone" });
  assert.equal(created.session_id, "sess-gone");
  assert.equal(sessionManager.transcriptRequests.length, 1);
  assert.deepEqual(runtime.sessions.get("sess-gone").agent.agent.state.messages, []);
});

test("CreateSession starts empty when the transcript is empty", async (t) => {
  const sessionManager = await startFakeSessionManager({ transcript: [] });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-fresh" });
  assert.equal(created.session_id, "sess-fresh");
  assert.equal(sessionManager.transcriptRequests.length, 1);
  assert.deepEqual(runtime.sessions.get("sess-fresh").agent.agent.state.messages, []);
});

test("CreateSession without a caller session_id does not fetch a transcript", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [transcriptMessage(1, "user", { content: "hi" })],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice" });
  assert.ok(created.session_id);
  assert.equal(sessionManager.transcriptRequests.length, 0);
  assert.deepEqual(runtime.sessions.get(created.session_id).agent.agent.state.messages, []);
});

test("CreateSession on resume applies the passed system_prompt and tools to the hydrated agent", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(1, "user", { content: "hi" }),
      transcriptMessage(2, "assistant", { content: "hello there" }),
    ],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({
    user_id: "alice",
    session_id: "sess-resumed",
    system_prompt: "You are a terse reviewer.",
    tools: ["read"],
  });
  assert.equal(created.session_id, "sess-resumed");
  assert.equal(sessionManager.transcriptRequests.length, 1);

  const state = runtime.sessions.get("sess-resumed").agent.agent.state;
  assert.equal(state.systemPrompt, "You are a terse reviewer.");
  assert.deepEqual(
    state.tools.map((tool) => tool.name),
    ["read"],
  );
  // History still hydrated alongside the new prompt/tools.
  assert.deepEqual(
    state.messages.map((m: { role: string }) => m.role),
    ["user", "assistant"],
  );
});

test("CreateSession on resume: request system_prompt beats the transcript's last system message", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(1, "system", { content: "You are a pirate tutor." }),
      transcriptMessage(2, "user", { content: "hi" }),
    ],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({
    user_id: "alice",
    session_id: "sess-request-wins",
    system_prompt: "You are a terse reviewer.",
  });
  const state = runtime.sessions.get(created.session_id).agent.agent.state;
  assert.equal(state.systemPrompt, "You are a terse reviewer.");
});

test("CreateSession on resume: transcript's last system message is the fallback prompt", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(1, "system", { content: "old persona" }),
      transcriptMessage(2, "user", { content: "hi" }),
      transcriptMessage(3, "assistant", { content: "hello" }),
      transcriptMessage(4, "system", { content: "newest persona" }),
    ],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-transcript-wins" });
  const state = runtime.sessions.get(created.session_id).agent.agent.state;
  // Last system message wins; system turns never enter the pi history.
  assert.equal(state.systemPrompt, "newest persona");
  assert.deepEqual(
    state.messages.map((m: { role: string }) => m.role),
    ["user", "assistant"],
  );
});

test("CreateSession on resume: no request prompt and no transcript system message means none", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [transcriptMessage(1, "user", { content: "hi" })],
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-no-prompt" });
  assert.equal(runtime.sessions.get(created.session_id).agent.agent.state.systemPrompt, "");
});
