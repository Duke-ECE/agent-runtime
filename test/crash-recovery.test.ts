import assert from "node:assert/strict";
import { test } from "node:test";
import { text, type CanonicalBlock, type CanonicalMessage } from "../src/canonical.js";
import type {
  DurableCheckpointRecord,
  DurableConfigRecord,
  DurableLease,
  DurableSessionClient,
  DurableSessionRecord,
  MessageDraftInput,
} from "../src/durable-client.js";
import { DurableExecution } from "../src/durable-execution.js";

/**
 * Crash recovery: what survives a process death at each durable boundary.
 *
 * Mocking session-manager per call cannot express this — the point is that the
 * durable state outlives the runtime. This fake therefore owns real state (the
 * request root, its messages, the lease generation, and terminality) and the
 * test drives two runtimes against it, with a simulated process death in
 * between.
 */

interface RootState {
  clientRequestId: string;
  requestHash: string;
  messages: MessageDraftInput[];
  terminal?: "completed" | "failed" | "cancelled" | "interrupted";
}

class DurableSessionManager implements DurableSessionClient {
  readonly sessionId = "sess-1";
  private readonly roots = new Map<string, RootState>();
  private byClientId = new Map<string, string>();
  private activeRequestId = "";
  private generation = 0;
  private revision = 1;
  /** How many tools the environment actually ran. Replay would show up here. */
  toolExecutions = 0;

  private session(): DurableSessionRecord {
    return {
      id: this.sessionId,
      userId: "u1",
      status: "active",
      agentId: "",
      title: "",
      lastSeq: this.roots.size,
      revision: this.revision,
      activeCheckpointId: "",
      activeRequestMessageId: this.activeRequestId,
    };
  }

  private requestMessage(id: string): CanonicalMessage {
    const root = this.roots.get(id);
    return {
      id,
      session_id: this.sessionId,
      seq: 1,
      request_message_id: id,
      role: "user",
      content: [text("go")],
      format_version: 1,
      status: "complete",
      request: {
        client_request_id: root?.clientRequestId ?? "",
        request_hash: root?.requestHash ?? "",
        status: root?.terminal ? root.terminal : this.activeRequestId === id ? "running" : "queued",
        cancellation_requested: false,
      },
    };
  }

  async createSession(): Promise<{ session: DurableSessionRecord; config: DurableConfigRecord }> {
    throw new Error("not used");
  }

  async getExecutionContext() {
    return { session: this.session(), config: {} as DurableConfigRecord };
  }

  async getMessages() {
    return { messages: [] as CanonicalMessage[], hasMore: false, revision: this.revision };
  }

  async beginRequest(req: any) {
    const existingId = this.byClientId.get(req.clientRequestId);
    if (existingId) {
      const root = this.roots.get(existingId)!;
      // Same id, same content: hand back the existing root untouched.
      assert.equal(root.requestHash, req.requestHash);
      return { session: this.session(), requestMessage: this.requestMessage(existingId), deduplicated: true };
    }
    const id = `msg-${this.roots.size + 1}`;
    this.roots.set(id, { clientRequestId: req.clientRequestId, requestHash: req.requestHash, messages: [] });
    this.byClientId.set(req.clientRequestId, id);
    this.activeRequestId = id;
    this.revision++;
    return { session: this.session(), requestMessage: this.requestMessage(id), deduplicated: false };
  }

  async acquireLease(req: any): Promise<{ session: DurableSessionRecord; lease: DurableLease }> {
    // A lease is only granted for the active, non-terminal request.
    assert.equal(this.activeRequestId, req.requestMessageId);
    assert.equal(this.roots.get(req.requestMessageId)?.terminal, undefined);
    this.generation++;
    return { session: this.session(), lease: { owner: req.owner, generation: this.generation } };
  }

  async renewLease(req: any): Promise<DurableLease> {
    return { owner: req.owner, generation: req.generation };
  }

  async releaseLease(): Promise<boolean> {
    return true;
  }

  async appendMessages(req: any) {
    const root = this.roots.get(req.requestMessageId);
    assert.ok(root, "append against an unknown request");
    assert.equal(root.terminal, undefined, "a terminal request must reject appends");
    root.messages.push(...req.messages);
    this.revision++;
    return {
      session: this.session(),
      messages: req.messages.map((m: MessageDraftInput, index: number) => ({
        id: m.id,
        session_id: this.sessionId,
        seq: 10 + index,
        request_message_id: req.requestMessageId,
        role: m.role,
        content: m.content,
        format_version: 1,
        status: m.status,
      })),
      deduplicated: false,
    };
  }

  async finishRequest(req: any) {
    const root = this.roots.get(req.requestMessageId);
    assert.ok(root, "finish against an unknown request");
    assert.equal(root.terminal, undefined, "a terminal request must reject a second finish");
    root.messages.push(...(req.messages ?? []));
    root.terminal = req.status;
    this.activeRequestId = "";
    this.revision++;
    return {
      session: this.session(),
      requestMessage: this.requestMessage(req.requestMessageId),
      messages: [] as CanonicalMessage[],
      deduplicated: false,
    };
  }

  async cancelRequest(): Promise<{ requestMessage: CanonicalMessage }> {
    throw new Error("not used");
  }

  async publishCheckpoint(req: any): Promise<{ session: DurableSessionRecord; checkpoint: DurableCheckpointRecord; deduplicated: boolean }> {
    return { session: this.session(), checkpoint: { ...req.checkpoint, sessionId: this.sessionId }, deduplicated: false };
  }

  close(): void {}

  /** The persisted transcript of a request, as a later runtime would read it. */
  transcript(requestId: string): MessageDraftInput[] {
    return [...(this.roots.get(requestId)?.messages ?? [])];
  }

  isTerminal(requestId: string): boolean {
    return this.roots.get(requestId)?.terminal !== undefined;
  }
}

const options = {
  sessionId: "sess-1",
  userId: "u1",
  clientRequestId: "req-1",
  content: [text("run pwd")],
  owner: "runtime-a",
};

function draft(id: string, role: "assistant" | "tool", content: CanonicalBlock[]): MessageDraftInput {
  return { id, role, content, status: "complete" };
}

test("a crash after admission leaves a reconnectable root and re-executes nothing", async () => {
  const sm = new DurableSessionManager();

  // First runtime: admits, takes the lease, then dies before doing any work.
  const first = await DurableExecution.start({ client: sm, ...options });
  const requestId = first.requestMessageId;
  first.dispose();
  assert.equal(sm.transcript(requestId).length, 0);

  // A restarted runtime with the same client request id finds the root instead
  // of admitting a second one.
  const second = await DurableExecution.start({ client: sm, ...options, owner: "runtime-b" });
  assert.equal(second.deduplicated, true, "the reconnect must not admit a new request");
  assert.equal(second.requestMessageId, requestId);
  assert.equal(sm.transcript(requestId).length, 0, "nothing was executed or written twice");
});

test("a crash after the tool result preserves the pair and does not replay the tool", async () => {
  const sm = new DurableSessionManager();
  const first = await DurableExecution.start({ client: sm, ...options });
  const requestId = first.requestMessageId;

  // The assistant call and its observed result both landed before the crash.
  await first.persist([
    draft("msg-a1", "assistant", [
      { type: "tool_call", id: "call_1", name: "bash", arguments_json: '{"command":"pwd"}' },
    ]),
  ]);
  sm.toolExecutions++;
  await first.persist([
    draft("msg-t1", "tool", [
      { type: "tool_result", tool_call_id: "call_1", status: "success", content: [text("/workspace")] },
    ]),
  ]);
  first.dispose(); // process death: no finish

  assert.equal(sm.isTerminal(requestId), false, "the request is left nonterminal for a retry decision");
  const persisted = sm.transcript(requestId);
  assert.equal(persisted.length, 2);

  // A restarted runtime reconnects. It must not run the tool again: the durable
  // transcript already holds the observed result.
  const second = await DurableExecution.start({ client: sm, ...options, owner: "runtime-b" });
  assert.equal(second.deduplicated, true);
  assert.equal(sm.toolExecutions, 1, "a completed tool must never be replayed");
  assert.equal(sm.transcript(requestId).length, 2, "the pair is intact and unmodified");
});

test("a crash after the terminal write cannot finish or rewrite the request again", async () => {
  const sm = new DurableSessionManager();
  const first = await DurableExecution.start({ client: sm, ...options });
  const requestId = first.requestMessageId;
  await first.persist([draft("msg-t1", "tool", [
    { type: "tool_result", tool_call_id: "call_1", status: "success", content: [text("ok")] },
  ])]);
  await first.finish("completed", [draft("msg-a2", "assistant", [text("done")])]);
  first.dispose();

  assert.equal(sm.isTerminal(requestId), true);
  const sealed = sm.transcript(requestId).length;

  // Duplicate delivery after the crash: the same client request id is
  // deduplicated, so the terminal state is not written a second time.
  const second = await DurableExecution.start({ client: sm, ...options, owner: "runtime-b" });
  assert.equal(second.deduplicated, true);
  // The deduplicated execution holds no lease, so it cannot write at all.
  await assert.rejects(second.persist([draft("msg-a3", "assistant", [text("again")])]), /no execution lease/);
  assert.equal(sm.transcript(requestId).length, sealed, "the sealed transcript was not extended");
  second.dispose();
});
