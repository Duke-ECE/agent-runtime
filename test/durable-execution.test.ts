import assert from "node:assert/strict";
import { test } from "node:test";
import { text, type CanonicalMessage } from "../src/canonical.js";
import {
  DurableRpcError,
  type DurableCheckpointRecord,
  type DurableConfigRecord,
  type DurableLease,
  type DurableSessionClient,
  type DurableSessionRecord,
  type MessageDraftInput,
} from "../src/durable-client.js";
import { DurableExecution, batchHashOf, requestHashOf } from "../src/durable-execution.js";

function sessionRecord(overrides: Partial<DurableSessionRecord> = {}): DurableSessionRecord {
  return {
    id: "sess-1",
    userId: "u1",
    status: "active",
    agentId: "",
    title: "",
    lastSeq: 1,
    revision: 1,
    activeCheckpointId: "",
    activeRequestMessageId: "msg-1",
    ...overrides,
  };
}

function draft(id: string, body: string): MessageDraftInput {
  return { id, role: "assistant", content: [text(body)], status: "complete" };
}

/** Minimal in-memory session-manager stand-in for the execution lifecycle. */
class FakeClient implements DurableSessionClient {
  readonly beginCalls: Array<Record<string, unknown>> = [];
  readonly appendCalls: Array<Record<string, unknown>> = [];
  readonly finishCalls: Array<Record<string, unknown>> = [];
  readonly renewCalls: Array<Record<string, unknown>> = [];
  releaseCalls = 0;
  deduplicated = false;
  appendError?: Error;
  renewError?: Error;
  private revision = 1;
  private generation = 1;

  async createSession(): Promise<{ session: DurableSessionRecord; config: DurableConfigRecord }> {
    throw new Error("not used");
  }

  async getExecutionContext() {
    return { session: sessionRecord({ revision: this.revision }), config: {} as DurableConfigRecord };
  }

  async getMessages() {
    return { messages: [], hasMore: false, revision: this.revision };
  }

  async beginRequest(req: any) {
    this.beginCalls.push(req);
    return {
      session: sessionRecord({ revision: this.revision }),
      requestMessage: {
        id: "msg-1",
        session_id: "sess-1",
        seq: 1,
        request_message_id: "msg-1",
        role: "user" as const,
        content: req.content,
        format_version: 1,
        status: "complete" as const,
      },
      deduplicated: this.deduplicated,
    };
  }

  async acquireLease(req: any): Promise<{ session: DurableSessionRecord; lease: DurableLease }> {
    return {
      session: sessionRecord({ revision: ++this.revision }),
      lease: { owner: req.owner, generation: this.generation, expiresAt: undefined },
    };
  }

  async renewLease(req: any): Promise<DurableLease> {
    this.renewCalls.push(req);
    if (this.renewError) throw this.renewError;
    return { owner: req.owner, generation: this.generation, expiresAt: undefined };
  }

  async releaseLease(): Promise<boolean> {
    this.releaseCalls++;
    return true;
  }

  async appendMessages(req: any) {
    this.appendCalls.push(req);
    if (this.appendError) throw this.appendError;
    const messages = req.messages.map((m: MessageDraftInput, index: number) => ({
      id: m.id,
      session_id: "sess-1",
      seq: 10 + index,
      request_message_id: "msg-1",
      role: m.role,
      content: m.content,
      format_version: 1,
      status: m.status,
    }));
    return { session: sessionRecord({ revision: ++this.revision }), messages, deduplicated: false };
  }

  async finishRequest(req: any) {
    this.finishCalls.push(req);
    return {
      session: sessionRecord({ revision: ++this.revision, activeRequestMessageId: "" }),
      requestMessage: {
        id: "msg-1",
        session_id: "sess-1",
        seq: 1,
        request_message_id: "msg-1",
        role: "user" as const,
        content: [text("go")],
        format_version: 1,
        status: "complete" as const,
      },
      messages: [],
      deduplicated: false,
    };
  }

  async cancelRequest(): Promise<{ requestMessage: CanonicalMessage }> {
    throw new Error("not used");
  }

  async publishCheckpoint(req: any): Promise<{ session: DurableSessionRecord; checkpoint: DurableCheckpointRecord; deduplicated: boolean }> {
    return {
      session: sessionRecord({ revision: ++this.revision }),
      checkpoint: { ...req.checkpoint, sessionId: "sess-1" },
      deduplicated: false,
    };
  }

  close(): void {}
}

const baseOptions = {
  sessionId: "sess-1",
  userId: "u1",
  clientRequestId: "req-1",
  content: [text("run pwd")],
  owner: "runtime-a",
};

test("start admits the request, takes a lease, and records the revision", async () => {
  const client = new FakeClient();
  const execution = await DurableExecution.start({ client, ...baseOptions, leaseTtlSeconds: 90 });
  try {
    assert.equal(execution.requestMessageId, "msg-1");
    assert.equal(execution.leaseGeneration, 1);
    assert.equal(execution.deduplicated, false);
    assert.equal(execution.revision, 2, "acquiring the lease advances the revision");
    assert.equal(client.beginCalls.length, 1);
    assert.equal(client.beginCalls[0].requestHash, requestHashOf([text("run pwd")]));
  } finally {
    execution.dispose();
  }
});

test("a deduplicated admission is a reconnect and never takes a lease", async () => {
  const client = new FakeClient();
  client.deduplicated = true;
  const execution = await DurableExecution.start({ client, ...baseOptions });
  try {
    assert.equal(execution.deduplicated, true);
    assert.equal(execution.leaseGeneration, 0);
    // Writing would be a stale mutation: there is no lease to fence it.
    await assert.rejects(execution.persist([draft("msg-a1", "x")]), /no execution lease/);
    assert.equal(client.appendCalls.length, 0);
  } finally {
    execution.dispose();
  }
});

test("persist sends the current fence and advances the revision", async () => {
  const client = new FakeClient();
  const execution = await DurableExecution.start({ client, ...baseOptions });
  try {
    const stored = await execution.persist([draft("msg-a1", "running pwd")]);
    assert.equal(stored.length, 1);
    assert.equal(client.appendCalls.length, 1);
    const sent = client.appendCalls[0];
    assert.equal((sent.guard as any).leaseGeneration, 1);
    assert.equal((sent.guard as any).expectedRevision, 2);
    assert.equal((sent.guard as any).mutationHash, batchHashOf([draft("msg-a1", "running pwd")]));
    assert.equal(execution.revision, 3);

    // A second batch uses a fresh mutation id, so retries cannot collide.
    await execution.persist([draft("msg-t1", "/workspace")]);
    assert.notEqual((client.appendCalls[0].guard as any).mutationId, (client.appendCalls[1].guard as any).mutationId);
  } finally {
    execution.dispose();
  }
});

test("finish writes the terminal state once and stops renewal", async () => {
  const client = new FakeClient();
  const execution = await DurableExecution.start({ client, ...baseOptions, renewIntervalMs: 1 });
  await execution.finish("completed", [draft("msg-a1", "done")]);
  assert.equal(client.finishCalls.length, 1);
  assert.equal(client.finishCalls[0].status, "completed");

  // A second finish is a no-op; the region is already terminal.
  await execution.finish("completed", [draft("msg-a2", "again")]);
  assert.equal(client.finishCalls.length, 1);

  // Timers are stopped: no renewals after completion.
  const renewals = client.renewCalls.length;
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(client.renewCalls.length, renewals);
});

test("losing the lease stops writes and notifies the caller", async () => {
  const client = new FakeClient();
  client.renewError = new DurableRpcError(10, "aborted", "lease owner or generation does not match");
  let notified = 0;
  const execution = await DurableExecution.start({
    client,
    ...baseOptions,
    renewIntervalMs: 1,
    onLeaseLost: () => {
      notified++;
    },
  });

  await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(execution.leaseLost, true);
  assert.equal(notified, 1, "the caller is told exactly once so it can abort inference");

  await assert.rejects(execution.persist([draft("msg-a1", "x")]), (err: unknown) => {
    assert.ok(err instanceof DurableRpcError);
    assert.equal(err.kind, "aborted");
    return true;
  });
  assert.equal(client.appendCalls.length, 0, "a fenced-out runtime must not write");

  // Finishing is also refused: another runtime now owns the terminal write.
  await assert.rejects(execution.finish("completed"), /lease owner or generation/);
  assert.equal(client.finishCalls.length, 0);
});

test("a state conflict is not retryable but an abort is", () => {
  assert.equal(new DurableRpcError(9, "conflict", "busy").retryable, false);
  assert.equal(new DurableRpcError(10, "aborted", "stale").retryable, true);
});

test("request hashes are deterministic and content-sensitive", () => {
  assert.equal(requestHashOf([text("a")]), requestHashOf([text("a")]));
  assert.notEqual(requestHashOf([text("a")]), requestHashOf([text("b")]));
});
