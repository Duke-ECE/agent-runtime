import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { text, toolCall } from "../src/canonical.js";
import { loadConfig } from "../src/config.js";
import { DurableRpcError, createDurableSessionClient } from "../src/durable-client.js";

/** The vendored v2 contract, resolved the same way the client resolves it. */
function resolveProtoPath(): string {
  for (const rel of ["../proto/session/v2/session.proto", "../../proto/session/v2/session.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("proto/session/v2/session.proto not found");
}

interface Harness {
  addr: string;
  requests: Array<{ method: string; request: any; token?: string }>;
  close(): Promise<void>;
}

async function startFakeSessionManager(
  handlers: Record<string, (request: any, metadata: grpc.Metadata) => unknown>,
): Promise<Harness> {
  const packageDefinition = protoLoader.loadSync(resolveProtoPath(), {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const proto = grpc.loadPackageDefinition(packageDefinition) as any;
  const requests: Harness["requests"] = [];
  const impl: grpc.UntypedServiceImplementation = {};
  const methods = [
    "createSession",
    "getExecutionContext",
    "getMessages",
    "beginRequest",
    "acquireLease",
    "renewLease",
    "releaseLease",
    "appendMessages",
    "finishRequest",
    "cancelRequest",
    "publishCheckpoint",
  ];
  for (const method of methods) {
    impl[method] = (call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) => {
      const tokens = call.metadata.get("x-service-token");
      requests.push({ method, request: call.request, token: tokens.length ? String(tokens[0]) : undefined });
      const handler = handlers[method];
      if (!handler) {
        callback({ code: grpc.status.UNIMPLEMENTED, details: `${method} not stubbed` } as grpc.ServiceError, null);
        return;
      }
      try {
        callback(null, handler(call.request, call.metadata));
      } catch (err) {
        callback(err as grpc.ServiceError, null);
      }
    };
  }
  const server = new grpc.Server();
  server.addService(proto.session.v2.SessionService.service, impl);
  const port = await new Promise<number>((resolve, reject) => {
    server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (err, p) => (err ? reject(err) : resolve(p)));
  });
  return {
    addr: `127.0.0.1:${port}`,
    requests,
    close: () => new Promise<void>((resolve) => server.tryShutdown(() => resolve())),
  };
}

function clientFor(addr: string) {
  const client = createDurableSessionClient(
    loadConfig({ SESSION_MANAGER_ADDR: addr, SERVICE_TOKEN: "test-token" } as NodeJS.ProcessEnv),
  );
  assert.ok(client, "durable client should be created when SESSION_MANAGER_ADDR is set");
  return client;
}

const sessionPayload = {
  id: "sess-1",
  user_id: "u1",
  status: "SESSION_STATUS_ACTIVE",
  agent_id: "agent-1",
  last_seq: "2",
  revision: "3",
  active_request_message_id: "msg-1",
  created_at: { seconds: "1700000000", nanos: 0 },
};

const requestMessagePayload = {
  id: "msg-1",
  session_id: "sess-1",
  seq: "1",
  request_message_id: "msg-1",
  role: "MESSAGE_ROLE_USER",
  content: [{ text: { text: "run pwd" } }],
  format_version: 1,
  message_status: "MESSAGE_STATUS_COMPLETE",
  request: { client_request_id: "req-1", request_hash: "hash-1", status: "EXECUTION_STATUS_QUEUED", cancellation_requested: false },
};

test("beginRequest sends canonical blocks and decodes the admitted root", async () => {
  const harness = await startFakeSessionManager({
    beginRequest: () => ({ session: sessionPayload, request_message: requestMessagePayload, deduplicated: false }),
  });
  try {
    const client = clientFor(harness.addr)!;
    const result = await client.beginRequest({
      sessionId: "sess-1",
      userId: "u1",
      clientRequestId: "req-1",
      requestHash: "hash-1",
      content: [text("run pwd")],
    });

    const sent = harness.requests[0];
    assert.equal(sent.method, "beginRequest");
    assert.equal(sent.token, "test-token", "the service token must be attached as metadata");
    assert.equal(sent.request.p_session_id ?? sent.request.session_id, "sess-1");
    assert.equal(sent.request.content.length, 1);
    assert.equal(sent.request.content[0].text.text, "run pwd");

    // int64 fields arrive as strings and must be normalized to numbers.
    assert.equal(result.session.lastSeq, 2);
    assert.equal(result.session.revision, 3);
    assert.equal(result.requestMessage.seq, 1);
    assert.equal(result.requestMessage.role, "user");
    assert.equal(result.requestMessage.request?.status, "queued");
    assert.deepEqual(result.requestMessage.content, [{ type: "text", text: "run pwd" }]);
    assert.equal(result.session.createdAt, new Date(1_700_000_000_000).toISOString());
  } finally {
    await harness.close();
  }
});

test("appendMessages round-trips ordered blocks, usage, and the guard", async () => {
  const harness = await startFakeSessionManager({
    appendMessages: () => ({
      session: { ...sessionPayload, last_seq: "3" },
      messages: [
        {
          id: "msg-a1",
          session_id: "sess-1",
          seq: "3",
          request_message_id: "msg-1",
          role: "MESSAGE_ROLE_ASSISTANT",
          content: [
            { text: { text: "running pwd" } },
            { tool_call: { id: "call_1", name: "bash", arguments_json: '{"command":"pwd"}' } },
          ],
          format_version: 1,
          message_status: "MESSAGE_STATUS_COMPLETE",
          usage: { input_tokens: "10", output_tokens: "4", total_tokens: "14" },
        },
      ],
      deduplicated: false,
    }),
  });
  try {
    const client = clientFor(harness.addr)!;
    const result = await client.appendMessages({
      sessionId: "sess-1",
      requestMessageId: "msg-1",
      guard: { mutationId: "m-1", mutationHash: "mh-1", expectedRevision: 3, leaseGeneration: 7 },
      messages: [
        {
          id: "msg-a1",
          role: "assistant",
          status: "complete",
          content: [text("running pwd"), toolCall("call_1", "bash", { command: "pwd" })],
          usage: { input_tokens: 10, cached_input_tokens: 0, output_tokens: 4, reasoning_tokens: 0, total_tokens: 14 },
        },
      ],
    });

    const sent = harness.requests[0].request;
    assert.equal(sent.guard.mutation_id, "m-1");
    assert.equal(Number(sent.guard.expected_revision), 3);
    assert.equal(Number(sent.guard.lease_generation), 7);
    assert.equal(sent.messages[0].status, "MESSAGE_STATUS_COMPLETE");
    assert.equal(sent.messages[0].role, "MESSAGE_ROLE_ASSISTANT");
    // arguments stay a JSON string, exactly as the proto carries them.
    assert.equal(sent.messages[0].content[1].tool_call.arguments_json, '{"command":"pwd"}');

    assert.equal(result.messages[0].usage?.input_tokens, 10);
    assert.equal(result.messages[0].content[1].type, "tool_call");
    assert.equal(result.session.lastSeq, 3);
  } finally {
    await harness.close();
  }
});

test("tool results keep their original call link and status", async () => {
  const harness = await startFakeSessionManager({
    appendMessages: () => ({
      session: sessionPayload,
      messages: [
        {
          id: "msg-t1",
          session_id: "sess-1",
          seq: "4",
          request_message_id: "msg-1",
          role: "MESSAGE_ROLE_TOOL",
          content: [
            {
              tool_result: {
                tool_call_id: "call_1",
                status: "TOOL_RESULT_STATUS_ERROR",
                content: [{ text: { text: "boom" } }],
                error_code: "E_BASH",
              },
            },
          ],
          format_version: 1,
          message_status: "MESSAGE_STATUS_COMPLETE",
        },
      ],
      deduplicated: false,
    }),
  });
  try {
    const client = clientFor(harness.addr)!;
    const result = await client.appendMessages({
      sessionId: "sess-1",
      requestMessageId: "msg-1",
      guard: { mutationId: "m-2", mutationHash: "mh-2", leaseGeneration: 1 },
      messages: [
        {
          id: "msg-t1",
          role: "tool",
          status: "complete",
          content: [{ type: "tool_result", tool_call_id: "call_1", status: "error", content: [text("boom")], error_code: "E_BASH" }],
        },
      ],
    });
    const block = result.messages[0].content[0];
    assert.equal(block.type, "tool_result");
    if (block.type === "tool_result") {
      assert.equal(block.tool_call_id, "call_1");
      assert.equal(block.status, "error");
      assert.equal(block.error_code, "E_BASH");
    }
    const sent = harness.requests[0].request;
    assert.equal(sent.messages[0].content[0].tool_result.status, "TOOL_RESULT_STATUS_ERROR");
  } finally {
    await harness.close();
  }
});

test("execution context decodes the frozen config and active checkpoint", async () => {
  const harness = await startFakeSessionManager({
    getExecutionContext: () => ({
      session: sessionPayload,
      config: {
        session_id: "sess-1",
        format_version: 1,
        config_hash: "cfg-hash-1",
        system_prompt: "you are helpful",
        provider: "openai",
        model: "gpt-4o-mini",
        base_url: "https://llm.example.com/v1",
        credential_ref: "cred-1",
        context_window_tokens: "128000",
        output_limit_tokens: "4096",
        tools: [{ name: "bash" }],
      },
      checkpoint: {
        id: "cp-1",
        session_id: "sess-1",
        covered_through_seq: "4",
        source_head_seq: "6",
        source_revision: "5",
        config_hash: "cfg-hash-1",
        active_request_message_id: "msg-1",
        resume_after_seq: "6",
        summary: [{ text: { text: "earlier context" } }],
        format_version: 1,
      },
    }),
  });
  try {
    const client = clientFor(harness.addr)!;
    const state = await client.getExecutionContext("sess-1");
    assert.equal(state.config.contextWindowTokens, 128000);
    assert.equal(state.config.model, "gpt-4o-mini");
    assert.equal(state.checkpoint?.coveredThroughSeq, 4);
    assert.equal(state.checkpoint?.resumeAfterSeq, 6);
    assert.deepEqual(state.checkpoint?.summary, [{ type: "text", text: "earlier context" }]);
  } finally {
    await harness.close();
  }
});

test("a state conflict surfaces as a classified, non-retryable error", async () => {
  const harness = await startFakeSessionManager({
    acquireLease: () => {
      throw { code: grpc.status.FAILED_PRECONDITION, details: "another request is already active on this session" };
    },
  });
  try {
    const client = clientFor(harness.addr)!;
    await assert.rejects(
      client.acquireLease({ sessionId: "sess-1", requestMessageId: "msg-1", owner: "rt", ttlSeconds: 90 }),
      (err: unknown) => {
        assert.ok(err instanceof DurableRpcError);
        assert.equal(err.kind, "conflict");
        assert.equal(err.retryable, false);
        assert.match(err.message, /already active/);
        return true;
      },
    );
  } finally {
    await harness.close();
  }
});

test("a lost fencing race surfaces as a retryable abort", async () => {
  const harness = await startFakeSessionManager({
    renewLease: () => {
      throw { code: grpc.status.ABORTED, details: "lease owner or generation does not match" };
    },
  });
  try {
    const client = clientFor(harness.addr)!;
    await assert.rejects(
      client.renewLease({ sessionId: "sess-1", owner: "rt", generation: 1, ttlSeconds: 90 }),
      (err: unknown) => {
        assert.ok(err instanceof DurableRpcError);
        assert.equal(err.kind, "aborted");
        assert.equal(err.retryable, true);
        return true;
      },
    );
  } finally {
    await harness.close();
  }
});
