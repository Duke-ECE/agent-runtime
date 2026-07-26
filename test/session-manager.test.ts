import assert from "node:assert/strict";
import { test } from "node:test";
import { status } from "@grpc/grpc-js";
import { SessionManager } from "../src/session-manager.js";

interface MockAgent {
  aborted: boolean;
}

function makeManager(overrides: Partial<{ maxSessions: number; ttlMs: number }> = {}) {
  const destroyed: MockAgent[] = [];
  const manager = new SessionManager<MockAgent>({
    maxSessions: overrides.maxSessions ?? 20,
    ttlMs: overrides.ttlMs ?? 30 * 60_000,
    destroyAgent: (agent) => {
      agent.aborted = true;
      destroyed.push(agent);
    },
  });
  return { manager, destroyed, createAgent: () => ({ aborted: false }) };
}

test("create assigns sess- ids and lists sessions", () => {
  const { manager, createAgent } = makeManager();
  const s = manager.create("alice", createAgent);
  assert.match(s.id, /^sess-[0-9a-f]{16}$/);
  assert.equal(s.userId, "alice");
  assert.equal(s.status, "active");
  assert.equal(manager.size, 1);
  assert.deepEqual(manager.list().map((x) => x.id), [s.id]);
});

test("cap exceeded -> RESOURCE_EXHAUSTED", () => {
  const { manager, createAgent } = makeManager({ maxSessions: 1 });
  manager.create("alice", createAgent);
  assert.throws(() => manager.create("bob", createAgent), (err: unknown) => {
    assert.equal((err as { code: number }).code, status.RESOURCE_EXHAUSTED);
    return true;
  });
  assert.equal(manager.size, 1);
});

test("create adopts a caller-provided id", () => {
  const { manager, createAgent } = makeManager();
  const s = manager.create("alice", createAgent, "sess-from-manager");
  assert.equal(s.id, "sess-from-manager");
  assert.equal(manager.get("sess-from-manager").userId, "alice");
});

test("create with a colliding caller-provided id -> ALREADY_EXISTS", () => {
  const { manager, createAgent } = makeManager();
  manager.create("alice", createAgent, "sess-dup");
  assert.throws(() => manager.create("bob", createAgent, "sess-dup"), (err: unknown) => {
    assert.equal((err as { code: number }).code, status.ALREADY_EXISTS);
    return true;
  });
  assert.equal(manager.size, 1);
  assert.equal(manager.get("sess-dup").userId, "alice");
});

test("get/end unknown session -> NOT_FOUND", () => {
  const { manager } = makeManager();
  for (const op of [() => manager.get("sess-nope"), () => manager.end("sess-nope")]) {
    assert.throws(op, (err: unknown) => {
      assert.equal((err as { code: number }).code, status.NOT_FOUND);
      return true;
    });
  }
});

test("end removes the session and destroys its agent; reuse -> NOT_FOUND", () => {
  const { manager, createAgent, destroyed } = makeManager();
  const s = manager.create("alice", createAgent);
  manager.end(s.id);
  assert.equal(manager.size, 0);
  assert.equal(destroyed.length, 1);
  assert.equal(destroyed[0].aborted, true);
  assert.throws(() => manager.get(s.id), (err: unknown) => {
    assert.equal((err as { code: number }).code, status.NOT_FOUND);
    return true;
  });
});

test("reaper removes sessions idle past the TTL, keeps fresh ones", () => {
  const { manager, createAgent, destroyed } = makeManager({ ttlMs: 1_000 });
  const old = manager.create("alice", createAgent);
  // Simulate idleness by backdating lastActive.
  old.lastActive = new Date(Date.now() - 2_000);
  const fresh = manager.create("bob", createAgent);

  const removed = manager.reap();
  assert.deepEqual(removed, [old.id]);
  assert.equal(manager.size, 1);
  assert.deepEqual(manager.list().map((s) => s.id), [fresh.id]);
  assert.equal(destroyed.length, 1);
});

test("touch keeps a session alive across reaps", () => {
  const { manager, createAgent } = makeManager({ ttlMs: 1_000 });
  const s = manager.create("alice", createAgent);
  s.lastActive = new Date(Date.now() - 2_000);
  manager.touch(s.id);
  assert.deepEqual(manager.reap(), []);
  assert.equal(manager.size, 1);
});

test("start/stop manage the reaper interval without keeping the process alive", () => {
  const { manager } = makeManager();
  manager.start();
  manager.start(); // idempotent
  manager.stop();
  manager.stop(); // safe no-op
});
