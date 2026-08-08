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
  limit?: number;
  before_seq?: number;
  token?: string;
}

async function startFakeSessionManager(opts: {
  transcript?: TranscriptMessage[];
  // Served for full-transcript (limit = 0) recovery fetches; defaults to
  // `transcript`. Lets a test put markers beyond the windowed page.
  fullTranscript?: TranscriptMessage[];
  transcriptError?: grpc.status;
  hasMore?: boolean;
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
        limit: call.request.limit,
        before_seq: call.request.before_seq,
        token: typeof token === "string" ? token : undefined,
      });
      if (opts.transcriptError !== undefined) {
        callback(Object.assign(new Error("transcript unavailable"), { code: opts.transcriptError }), null);
        return;
      }
      const isFullFetch = !call.request.limit;
      callback(null, {
        messages: (isFullFetch ? (opts.fullTranscript ?? opts.transcript) : opts.transcript) ?? [],
        has_more: isFullFetch ? false : (opts.hasMore ?? false),
      });
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
  configOverrides: Partial<ServiceConfig> = {},
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
    hydrationMaxTurns: 50,
    ...configOverrides,
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

  assert.deepEqual(sessionManager.transcriptRequests, [
    { session_id: "sess-hydrated", limit: 50, before_seq: 0, token: "test-token" },
  ]);

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

test("hydration requests only the latest HYDRATION_MAX_TURNS window", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [transcriptMessage(1, "user", { content: "hi" })],
  });
  t.after(sessionManager.close);
  const { createSession } = await setup(t, sessionManager.addr, { hydrationMaxTurns: 7 });

  await createSession({ user_id: "alice", session_id: "sess-windowed" });
  assert.deepEqual(sessionManager.transcriptRequests, [
    { session_id: "sess-windowed", limit: 7, before_seq: 0, token: "test-token" },
  ]);
});

test("hydration with has_more hydrates the window and logs the skipped earlier history", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(41, "user", { content: "latest question" }),
      transcriptMessage(42, "assistant", { content: "latest answer" }),
    ],
    hasMore: true,
  });
  t.after(sessionManager.close);
  const logs: string[] = [];
  const originalLog = console.log;
  console.log = (...args: unknown[]) => {
    logs.push(args.map(String).join(" "));
  };
  t.after(() => {
    console.log = originalLog;
  });
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-has-more" });
  const messages = runtime.sessions.get(created.session_id).agent.agent.state.messages as any[];
  assert.deepEqual(
    messages.map((m) => m.role),
    ["user", "assistant"],
  );
  assert.ok(
    logs.some((line) => line.includes("earlier transcript history skipped") && line.includes("sess-has-more")),
    `expected a skip log line, got: ${JSON.stringify(logs)}`,
  );
});

test("a window boundary mid-tool-pair hydrates cleanly", async (t) => {
  const sessionManager = await startFakeSessionManager({
    // As a latest-window page can look: the window opened between a tool_call
    // and its result (orphan tool_result first) and closed right after a
    // tool_call whose result is beyond the transcript end (dangling call).
    transcript: [
      transcriptMessage(5, "tool_result", { tool: "bash", ok: true, output: "orphan", error: "" }),
      transcriptMessage(6, "user", { content: "hi" }),
      transcriptMessage(7, "assistant", { content: "hello" }),
      transcriptMessage(8, "tool_call", { tool: "bash", arguments_json: "{}" }),
    ],
    hasMore: true,
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-mid-pair" });
  const messages = runtime.sessions.get(created.session_id).agent.agent.state.messages as any[];
  assert.deepEqual(
    messages.map((m) => m.role),
    ["user", "assistant"],
  );
});

test("system fallback works when a system turn is inside the window", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(38, "system", { content: "windowed persona" }),
      transcriptMessage(39, "user", { content: "hi" }),
      transcriptMessage(40, "assistant", { content: "hello" }),
    ],
    hasMore: true, // older history (including any older system turns) is out of the window
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-windowed-persona" });
  const state = runtime.sessions.get(created.session_id).agent.agent.state;
  assert.equal(state.systemPrompt, "windowed persona");
  assert.deepEqual(
    state.messages.map((m: { role: string }) => m.role),
    ["user", "assistant"],
  );
});

test("markers beyond the window are recovered via a full-transcript fetch", async (t) => {
  const sessionManager = await startFakeSessionManager({
    // The window (latest 2): no system/config turns.
    transcript: [
      transcriptMessage(50, "user", { content: "latest question" }),
      transcriptMessage(51, "assistant", { content: "latest answer" }),
    ],
    // The full transcript: the frozen LLM and the persona sit at the head.
    fullTranscript: [
      transcriptMessage(1, "system", { content: "head persona" }),
      transcriptMessage(2, "config", {
        llm: { api_key: "frozen-key", base_url: "https://frozen.example/v1", model: "frozen-model" },
      }),
      transcriptMessage(3, "user", { content: "old question" }),
      transcriptMessage(4, "assistant", { content: "old answer" }),
      transcriptMessage(50, "user", { content: "latest question" }),
      transcriptMessage(51, "assistant", { content: "latest answer" }),
    ],
    hasMore: true,
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({
    user_id: "alice",
    session_id: "sess-marker-recovery",
    // A request llm that must LOSE to the recovered frozen triple.
    llm: { api_key: "request-key", base_url: "https://request.example/v1", model: "request-model" },
  });
  const sess = runtime.sessions.get(created.session_id).agent;
  assert.deepEqual(sess.llm, {
    apiKey: "frozen-key",
    baseUrl: "https://frozen.example/v1",
    model: "frozen-model",
  });
  const state = sess.agent.state;
  assert.equal(state.systemPrompt, "head persona");
  // The conversation context still comes from the window only.
  assert.deepEqual(
    state.messages.map((m: { role: string }) => m.role),
    ["user", "assistant"],
  );
  // Two fetches: the window, then the full recovery fetch (limit unset).
  assert.equal(sessionManager.transcriptRequests.length, 2);
  assert.ok(!sessionManager.transcriptRequests[1].limit);
});

test("no recovery fetch when the window already carries both markers", async (t) => {
  const sessionManager = await startFakeSessionManager({
    transcript: [
      transcriptMessage(40, "system", { content: "windowed persona" }),
      transcriptMessage(41, "config", {
        llm: { api_key: "frozen-key", base_url: "https://frozen.example/v1", model: "frozen-model" },
      }),
      transcriptMessage(42, "user", { content: "hi" }),
      transcriptMessage(43, "assistant", { content: "hello" }),
    ],
    hasMore: true,
  });
  t.after(sessionManager.close);
  const { runtime, createSession } = await setup(t, sessionManager.addr);

  const created = await createSession({ user_id: "alice", session_id: "sess-markers-in-window" });
  assert.equal(runtime.sessions.get(created.session_id).agent.llm.apiKey, "frozen-key");
  assert.equal(sessionManager.transcriptRequests.length, 1);
});
