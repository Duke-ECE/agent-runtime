import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { createRuntime } from "../src/server.js";
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

interface TestClient {
  createSession(req: unknown): Promise<{ session_id: string }>;
  endSession(req: unknown): Promise<unknown>;
  listSessions(req: unknown): Promise<{ sessions: Array<{ id: string; user_id: string }> }>;
  chat(req: unknown): grpc.ClientReadableStream<unknown>;
}

async function setup(): Promise<{ client: TestClient; close: () => Promise<void> }> {
  const { server } = createRuntime(TEST_CONFIG);
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) =>
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
  const raw = new proto.runtime.v1.AgentService(
    `127.0.0.1:${port}`,
    grpc.credentials.createInsecure(),
  ) as any;
  const unary =
    (method: string) =>
    (req: unknown): Promise<any> =>
      new Promise((resolve, reject) => {
        raw[method](req, (err: grpc.ServiceError | null, res: unknown) => (err ? reject(err) : resolve(res)));
      });
  const client: TestClient = {
    createSession: unary("createSession"),
    endSession: unary("endSession"),
    listSessions: unary("listSessions"),
    chat: (req) => raw.chat(req),
  };
  const close = () =>
    new Promise<void>((resolve) => {
      raw.close();
      server.tryShutdown(() => resolve());
    });
  return { client, close };
}

/** Collect a Chat stream: resolves with data events on clean end, rejects with the gRPC error otherwise. */
function chatEvents(stream: grpc.ClientReadableStream<unknown>): Promise<unknown[]> {
  return new Promise((resolve, reject) => {
    const events: unknown[] = [];
    stream.on("data", (e) => events.push(e));
    stream.on("end", () => resolve(events));
    stream.on("error", reject);
  });
}

function assertGrpcError(err: unknown, code: grpc.status): void {
  assert.ok(err instanceof Error, "expected an error");
  assert.equal((err as grpc.ServiceError).code, code);
  assert.match(err.message, /session belongs to another user/);
}

test("per-user session ownership is enforced across RPCs", async (t) => {
  const { client, close } = await setup();
  t.after(close);

  const alice = await client.createSession({ user_id: "alice" });
  const bob = await client.createSession({ user_id: "bob" });
  assert.ok(alice.session_id && bob.session_id);

  await t.test("ListSessions filters to the caller when user_id is set", async () => {
    const mine = await client.listSessions({ user_id: "alice" });
    assert.deepEqual(
      mine.sessions.map((s) => s.id),
      [alice.session_id],
    );
    assert.deepEqual(mine.sessions[0].user_id, "alice");
  });

  await t.test("ListSessions with empty user_id returns everything (open mode)", async () => {
    const all = await client.listSessions({});
    assert.deepEqual(
      all.sessions.map((s) => s.id).sort(),
      [alice.session_id, bob.session_id].sort(),
    );
  });

  await t.test("Chat from a foreign user -> PERMISSION_DENIED", async () => {
    await assert.rejects(chatEvents(client.chat({ session_id: alice.session_id, content: "hi", user_id: "bob" })),
      (err: unknown) => {
        assertGrpcError(err, grpc.status.PERMISSION_DENIED);
        return true;
      },
    );
  });

  await t.test("Chat from a foreign user on unknown session stays NOT_FOUND", async () => {
    await assert.rejects(chatEvents(client.chat({ session_id: "sess-nope", content: "hi", user_id: "bob" })),
      (err: unknown) => {
        assert.equal((err as grpc.ServiceError).code, grpc.status.NOT_FOUND);
        return true;
      },
    );
  });

  // With no LLM API key the server answers Chat with a data-level error event
  // and a clean end; reaching that branch proves the ownership check passed.
  await t.test("Chat by the owner passes the ownership check", async () => {
    const events = (await chatEvents(
      client.chat({ session_id: alice.session_id, content: "hi", user_id: "alice" }),
    )) as Array<{ error?: { message: string } }>;
    assert.equal(events.length, 1);
    assert.match(events[0].error?.message ?? "", /LLM is not configured/);
  });

  await t.test("Chat with empty user_id is allowed (open mode)", async () => {
    const events = (await chatEvents(client.chat({ session_id: bob.session_id, content: "hi" }))) as Array<{
      error?: { message: string };
    }>;
    assert.equal(events.length, 1);
    assert.match(events[0].error?.message ?? "", /LLM is not configured/);
  });

  await t.test("EndSession from a foreign user -> PERMISSION_DENIED, session survives", async () => {
    await assert.rejects(client.endSession({ session_id: alice.session_id, user_id: "bob" }), (err: unknown) => {
      assertGrpcError(err, grpc.status.PERMISSION_DENIED);
      return true;
    });
    const all = await client.listSessions({});
    assert.equal(all.sessions.length, 2);
  });

  await t.test("EndSession with empty user_id still works (open mode)", async () => {
    await client.endSession({ session_id: bob.session_id });
    const all = await client.listSessions({});
    assert.deepEqual(
      all.sessions.map((s) => s.id),
      [alice.session_id],
    );
  });

  await t.test("EndSession by the owner succeeds", async () => {
    await client.endSession({ session_id: alice.session_id, user_id: "alice" });
    const all = await client.listSessions({});
    assert.equal(all.sessions.length, 0);
  });

  await t.test("EndSession on unknown session stays NOT_FOUND", async () => {
    await assert.rejects(client.endSession({ session_id: "sess-nope", user_id: "alice" }), (err: unknown) => {
      assert.equal((err as grpc.ServiceError).code, grpc.status.NOT_FOUND);
      return true;
    });
  });
});
