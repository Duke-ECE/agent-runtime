import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { createRuntime, type Runtime } from "../src/server.js";
import type { ServiceConfig } from "../src/config.js";

function resolveProtoPath(): string {
  for (const rel of ["../../proto/runtime/v1/agent.proto", "../proto/runtime/v1/agent.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("proto/runtime/v1/agent.proto not found");
}

const TEST_CONFIG: ServiceConfig = {
  port: 0,
  maxSessions: 20,
  sessionTtlMinutes: 30,
  llm: { apiKey: undefined, baseUrl: "http://127.0.0.1:1/v1", model: "test-model" },
};

async function setup(t: import("node:test").TestContext): Promise<{
  runtime: Runtime;
  createSession: (req: unknown) => Promise<{ session_id: string }>;
}> {
  const runtime = createRuntime(TEST_CONFIG);
  const port = await new Promise<number>((resolve, reject) => {
    runtime.server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
      err ? reject(err) : resolve(p),
    );
  });
  const packageDefinition = protoLoader.loadSync(resolveProtoPath(), {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const proto = grpc.loadPackageDefinition(packageDefinition) as any;
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

function agentState(runtime: Runtime, sessionId: string): { systemPrompt: string; tools: Array<{ name: string }> } {
  return runtime.sessions.get(sessionId).agent.agent.state;
}

test("CreateSession without the new fields starts with no system prompt and all tools", async (t) => {
  const { runtime, createSession } = await setup(t);

  const created = await createSession({ user_id: "alice" });
  const state = agentState(runtime, created.session_id);
  assert.equal(state.systemPrompt, "");
  assert.deepEqual(
    state.tools.map((tool) => tool.name),
    ["read", "write", "bash", "edit"],
  );
});

test("CreateSession with empty system_prompt and tools starts with no system prompt and all tools", async (t) => {
  const { runtime, createSession } = await setup(t);

  const created = await createSession({ user_id: "alice", system_prompt: "", tools: [] });
  const state = agentState(runtime, created.session_id);
  assert.equal(state.systemPrompt, "");
  assert.deepEqual(
    state.tools.map((tool) => tool.name),
    ["read", "write", "bash", "edit"],
  );
});

test("CreateSession applies a custom system_prompt", async (t) => {
  const { runtime, createSession } = await setup(t);

  const created = await createSession({ user_id: "alice", system_prompt: "You are a pirate tutor." });
  assert.equal(agentState(runtime, created.session_id).systemPrompt, "You are a pirate tutor.");
});

test("CreateSession filters tools to the requested whitelist", async (t) => {
  const { runtime, createSession } = await setup(t);

  const created = await createSession({ user_id: "alice", tools: ["read", "bash"] });
  assert.deepEqual(
    agentState(runtime, created.session_id).tools.map((tool) => tool.name),
    ["read", "bash"],
  );
});

test("CreateSession rejects unknown tool names with INVALID_ARGUMENT", async (t) => {
  const { runtime, createSession } = await setup(t);

  await assert.rejects(
    createSession({ user_id: "alice", tools: ["read", "nope", "alsonope"] }),
    (err: unknown) => {
      assert.equal((err as grpc.ServiceError).code, grpc.status.INVALID_ARGUMENT);
      assert.match(err instanceof Error ? err.message : String(err), /unknown tools: nope, alsonope/);
      return true;
    },
  );
  // The rejected request must not leave a session behind.
  assert.equal(runtime.sessions.list().length, 0);
});
