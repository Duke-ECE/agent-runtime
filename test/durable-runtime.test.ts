import assert from "node:assert/strict";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import { text, type CanonicalBlock, type CanonicalMessage } from "../src/canonical.js";
import { loadConfig } from "../src/config.js";
import type {
  DurableCheckpointRecord,
  DurableConfigRecord,
  DurableLease,
  DurableSessionClient,
  DurableSessionRecord,
  MessageDraftInput,
} from "../src/durable-client.js";
import { createDurableRuntime } from "../src/durable-runtime.js";
import { createModel } from "../src/llm.js";

/**
 * End-to-end handler test: a scripted model stream drives the real pi agent
 * loop, so the durable boundaries exercised here are the ones that run in
 * production. Only the session-manager and the provider are faked.
 */

const config = loadConfig({ SESSION_MANAGER_ADDR: "127.0.0.1:1", SERVICE_TOKEN: "tok" } as NodeJS.ProcessEnv);

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
    activeRequestMessageId: "msg-r1",
    ...overrides,
  };
}

function frozenConfig(): DurableConfigRecord {
  return {
    sessionId: "sess-1",
    formatVersion: 1,
    configHash: "cfg-hash-1",
    systemPrompt: "you are helpful",
    provider: "openai",
    model: "gpt-4o-mini",
    baseUrl: "https://llm.example.com/v1",
    credentialRef: "cred-1",
    contextWindowTokens: 128_000,
    outputLimitTokens: 4_096,
    tools: [],
  };
}

/** In-memory session-manager recording every durable call in order. */
class FakeClient implements DurableSessionClient {
  readonly calls: string[] = [];
  readonly appended: MessageDraftInput[][] = [];
  readonly finished: Array<Record<string, unknown>> = [];
  deduplicated = false;
  revision = 1;
  leaseLost = false;

  async createSession(): Promise<{ session: DurableSessionRecord; config: DurableConfigRecord }> {
    return { session: sessionRecord(), config: frozenConfig() };
  }

  async getExecutionContext() {
    this.calls.push("getExecutionContext");
    return { session: sessionRecord({ revision: this.revision }), config: frozenConfig() };
  }

  async getMessages() {
    return { messages: [] as CanonicalMessage[], hasMore: false, revision: this.revision };
  }

  async beginRequest(req: any) {
    this.calls.push("beginRequest");
    return {
      session: sessionRecord({ revision: this.revision }),
      requestMessage: {
        id: "msg-r1",
        session_id: "sess-1",
        seq: 1,
        request_message_id: "msg-r1",
        role: "user" as const,
        content: req.content as CanonicalBlock[],
        format_version: 1,
        status: "complete" as const,
      },
      deduplicated: this.deduplicated,
    };
  }

  async acquireLease(req: any): Promise<{ session: DurableSessionRecord; lease: DurableLease }> {
    this.calls.push("acquireLease");
    return { session: sessionRecord({ revision: ++this.revision }), lease: { owner: req.owner, generation: 1 } };
  }

  async renewLease(): Promise<DurableLease> {
    this.calls.push("renewLease");
    return { owner: "rt", generation: 1 };
  }

  async releaseLease(): Promise<boolean> {
    this.calls.push("releaseLease");
    return true;
  }

  async appendMessages(req: any) {
    this.calls.push("appendMessages");
    this.appended.push(req.messages);
    const messages = req.messages.map((m: MessageDraftInput, index: number) => ({
      id: m.id,
      session_id: "sess-1",
      seq: 10 + index,
      request_message_id: "msg-r1",
      role: m.role,
      content: m.content,
      format_version: 1,
      status: m.status,
      reply_to_message_id: m.replyToMessageId,
    }));
    return { session: sessionRecord({ revision: ++this.revision }), messages, deduplicated: false };
  }

  async finishRequest(req: any) {
    this.calls.push("finishRequest");
    this.finished.push(req);
    return {
      session: sessionRecord({ revision: ++this.revision, activeRequestMessageId: "" }),
      requestMessage: {
        id: "msg-r1",
        session_id: "sess-1",
        seq: 1,
        request_message_id: "msg-r1",
        role: "user" as const,
        content: [text("go")],
        format_version: 1,
        status: "complete" as const,
      },
      messages: [] as CanonicalMessage[],
      deduplicated: false,
    };
  }

  async cancelRequest(): Promise<{ requestMessage: CanonicalMessage }> {
    this.calls.push("cancelRequest");
    throw new Error("not used");
  }

  async publishCheckpoint(req: any): Promise<{ session: DurableSessionRecord; checkpoint: DurableCheckpointRecord; deduplicated: boolean }> {
    this.calls.push("publishCheckpoint");
    return { session: sessionRecord({ revision: ++this.revision }), checkpoint: { ...req.checkpoint, sessionId: "sess-1" }, deduplicated: false };
  }

  close(): void {}
}

/** A scripted provider stream: yields the given events, then resolves a message. */
function scriptedStream(events: unknown[], finalMessage: Record<string, unknown>): any {
  const stream = {
    async *[Symbol.asyncIterator]() {
      for (const event of events) yield event;
    },
    async result() {
      return finalMessage;
    },
    push() {},
    end() {},
  };
  return stream;
}

function assistantMessage(parts: unknown[], stopReason = "stop"): Record<string, unknown> {
  return {
    role: "assistant",
    content: parts,
    api: "openai-completions",
    provider: "openai-compatible",
    model: "gpt-4o-mini",
    usage: { input: 11, output: 3, cacheRead: 0, cacheWrite: 0, totalTokens: 14, cost: {} },
    stopReason,
    timestamp: Date.now(),
  };
}

interface FakeCall {
  writes: any[];
  errors: unknown[];
  ended: boolean;
  request: unknown;
  cancelled: boolean;
}

function fakeCall(request: unknown): FakeCall & grpc.ServerWritableStream<any, any> {
  const call = {
    writes: [] as any[],
    errors: [] as unknown[],
    ended: false,
    request,
    cancelled: false,
  } as unknown as FakeCall & {
    emit(event: string, err: unknown): boolean;
    write(payload: unknown): void;
    end(): void;
  };
  call.emit = (_event: string, err: unknown) => {
    call.errors.push(err);
    return true;
  };
  call.write = (payload: unknown) => {
    call.writes.push(payload);
  };
  call.end = () => {
    call.ended = true;
  };
  return call as FakeCall & grpc.ServerWritableStream<any, any>;
}

function chatRequest(content: CanonicalBlock[], clientRequestId = "req-1"): unknown {
  return {
    session_id: "sess-1",
    user_id: "u1",
    client_request_id: clientRequestId,
    content: content.map((block) =>
      block.type === "text" ? { text: { text: block.text } } : {},
    ),
  };
}

function runtimeFor(client: FakeClient, events: unknown[], finalMessage: Record<string, unknown>) {
  return createDurableRuntime(config, undefined, {
    client,
    modelFactory: (llm) => ({
      model: createModel(llm),
      streamFn: () => scriptedStream(events, finalMessage) as never,
    }),
  });
}

function waitFor(check: () => boolean, timeoutMs = 2_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const timer = setInterval(() => {
      if (check()) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error("timed out waiting for the handler"));
      }
    }, 5);
  });
}

test("a completed turn is admitted, leased, and finished with one terminal write", async () => {
  const client = new FakeClient();
  const events = [
    { type: "start", partial: assistantMessage([]) },
    { type: "text_delta", delta: "hello ", partial: assistantMessage([{ type: "text", text: "hello " }]) },
    { type: "text_delta", delta: "world", partial: assistantMessage([{ type: "text", text: "hello world" }]) },
    { type: "done", reason: "stop", message: assistantMessage([{ type: "text", text: "hello world" }]) },
  ];
  const final = assistantMessage([{ type: "text", text: "hello world" }]);
  const runtime = runtimeFor(client, events, final);
  const service = runtime.service as Record<string, (call: any) => void>;

  const call = fakeCall(chatRequest([text("hi")]));
  service.chat(call);
  await waitFor(() => call.ended || call.errors.length > 0);

  // Admission and ownership happen before any model work.
  const begin = client.calls.indexOf("beginRequest");
  const acquire = client.calls.indexOf("acquireLease");
  assert.ok(begin >= 0, "the request must be admitted");
  assert.ok(acquire > begin, "the lease must be taken after admission");

  // Deltas stream with the pending message id.
  const deltas = call.writes.filter((w) => w.text_delta);
  assert.equal(deltas.length, 2);
  assert.equal(deltas[0].text_delta.delta, "hello ");
  assert.equal(typeof deltas[0].text_delta.message_id, "string");

  // Exactly one terminal write, carrying the final assistant output.
  assert.equal(client.finished.length, 1);
  assert.equal(client.finished[0].status, "completed");
  const finalBatch = client.finished[0].messages as MessageDraftInput[];
  assert.equal(finalBatch.length, 1);
  assert.equal(finalBatch[0].role, "assistant");
  assert.deepEqual(finalBatch[0].content, [{ type: "text", text: "hello world" }]);
  assert.equal(finalBatch[0].usage?.input_tokens, 11);

  const done = call.writes.find((w) => w.done);
  assert.ok(done, "a done frame must be written");
  assert.equal(done.done.status, "EXECUTION_STATUS_COMPLETED");
  assert.deepEqual(done.done.aggregate_usage, { input_tokens: 11, output_tokens: 3, total_tokens: 14 });
  assert.equal(call.errors.length, 0);
});

test("a reconnect is deduplicated and never reruns or rewrites the request", async () => {
  const client = new FakeClient();
  client.deduplicated = true;
  const events = [{ type: "done", reason: "stop", message: assistantMessage([{ type: "text", text: "unused" }]) }];
  const runtime = runtimeFor(client, events, assistantMessage([{ type: "text", text: "unused" }]));
  const service = runtime.service as Record<string, (call: any) => void>;

  const call = fakeCall(chatRequest([text("hi")]));
  service.chat(call);
  await waitFor(() => call.ended || call.errors.length > 0);

  assert.ok(client.calls.includes("beginRequest"));
  assert.ok(!client.calls.includes("acquireLease"), "a reconnect must not take a lease");
  assert.ok(!client.calls.includes("appendMessages"));
  assert.ok(!client.calls.includes("finishRequest"), "a reconnect must not rewrite the terminal state");
  const done = call.writes.find((w) => w.done);
  assert.ok(done, "the reconnect reports existing state");
  assert.equal(done.done.request_message_id, "msg-r1");
});

test("a failed model call is reported and finishes as failed", async () => {
  const client = new FakeClient();
  const failure = assistantMessage([{ type: "text", text: "" }], "error");
  failure.errorMessage = "429 rate limited";
  const events = [{ type: "error", reason: "error", error: failure }];
  const runtime = runtimeFor(client, events, failure);
  const service = runtime.service as Record<string, (call: any) => void>;

  const call = fakeCall(chatRequest([text("hi")]));
  service.chat(call);
  await waitFor(() => call.ended || call.errors.length > 0);

  const errorFrame = call.writes.find((w) => w.error);
  assert.ok(errorFrame, "the failure must be streamed");
  assert.equal(errorFrame.error.retryable, true);
  assert.match(errorFrame.error.message, /429/);

  assert.equal(client.finished.length, 1);
  assert.equal(client.finished[0].status, "failed");
  assert.equal(call.writes.some((w) => w.done), false, "a failed request must not report done");
});
