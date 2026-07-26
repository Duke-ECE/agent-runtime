import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Agent } from "@earendil-works/pi-agent-core";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { loadConfig, type ServiceConfig } from "./config.js";
import { createModel, createStreamFn, type SessionLlmConfig } from "./llm.js";
import { createSessionWriter, type TurnMessageInput } from "./session-client.js";
import { SessionManager, assertOwner, grpcError } from "./session-manager.js";
import { buildTools, NullExecutor, type ToolExecutor } from "./tools.js";

const SYSTEM_PROMPT =
  "You are a helpful coding assistant for a Duke ECE student. " +
  "You have read/write/bash/edit tools, but they run in the student's sandbox. " +
  "If a tool call fails because the sandbox is not connected, explain that tools are temporarily unavailable and help the user in text instead.";

interface RuntimeSession {
  agent: Agent;
  llm: SessionLlmConfig;
}

function resolveProtoPath(): string {
  // dist/src/server.js -> <root>/proto/... ; also works when run from src/.
  for (const rel of ["../../proto/runtime/v1/agent.proto", "../proto/runtime/v1/agent.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("proto/runtime/v1/agent.proto not found; run `npm run sync-proto`");
}

function isRetryableLlmError(message: string | undefined): boolean {
  if (!message) return false;
  return /rate.?limit|429|timeout|timed out|overloaded|502|503|529|econnreset|fetch failed|network/i.test(message);
}

/**
 * Terminate a server-streaming call with a gRPC error status. grpc-js turns an
 * 'error' event on the server stream into the trailing error status; note that
 * `call.destroy(err)` does NOT deliver the status to the client (verified on
 * grpc-js 1.14.x), so we emit instead.
 */
function failStream(call: grpc.ServerWritableStream<unknown, unknown>, err: Error & { code: grpc.status }): void {
  call.emit("error", err);
}

function toolResultText(result: unknown): string {  if (result && typeof result === "object" && "content" in result) {
    const content = (result as { content?: unknown }).content;
    if (Array.isArray(content)) {
      return content
        .filter((part): part is { type: "text"; text?: unknown } =>
          Boolean(part && typeof part === "object" && (part as { type?: unknown }).type === "text"),
        )
        .map((part) => String(part.text ?? ""))
        .join("\n");
    }
  }
  return typeof result === "string" ? result : JSON.stringify(result ?? null);
}

export interface Runtime {
  server: grpc.Server;
  sessions: SessionManager<RuntimeSession>;
}

export function createRuntime(config: ServiceConfig, executor: ToolExecutor = new NullExecutor()): Runtime {
  const sessions = new SessionManager<RuntimeSession>({
    maxSessions: config.maxSessions,
    ttlMs: config.sessionTtlMinutes * 60_000,
    destroyAgent: ({ agent }) => agent.abort(),
  });
  const sessionWriter = createSessionWriter(config);

  function resolveLlm(req: { api_key?: string; base_url?: string; model?: string } | undefined): SessionLlmConfig {
    return {
      apiKey: req?.api_key || config.llm.apiKey,
      baseUrl: req?.base_url || config.llm.baseUrl,
      model: req?.model || config.llm.model,
    };
  }

  const packageDefinition = protoLoader.loadSync(resolveProtoPath(), {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as {
    runtime: { v1: { AgentService: grpc.ServiceClientConstructor } };
  };

  const service: grpc.UntypedServiceImplementation = {
    createSession(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { user_id, session_id, llm } = call.request as {
        user_id?: string;
        session_id?: string;
        llm?: { api_key?: string; base_url?: string; model?: string };
      };
      if (!user_id) {
        callback(grpcError(grpc.status.INVALID_ARGUMENT, "user_id is required") as grpc.ServiceError, null);
        return;
      }
      try {
        const resolved = resolveLlm(llm);
        const session = sessions.create(
          user_id,
          (sessionId) => ({
            llm: resolved,
            agent: new Agent({
              initialState: {
                systemPrompt: SYSTEM_PROMPT,
                model: createModel(resolved),
                tools: buildTools(executor),
              },
              streamFn: createStreamFn(resolved),
              sessionId,
            }),
          }),
          session_id || undefined,
        );
        callback(null, { session_id: session.id });
      } catch (err) {
        callback(err as grpc.ServiceError, null);
      }
    },

    endSession(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { session_id, user_id } = call.request as { session_id?: string; user_id?: string };
      try {
        assertOwner(sessions.get(session_id ?? ""), user_id);
        sessions.end(session_id ?? "");
        callback(null, {});
      } catch (err) {
        callback(err as grpc.ServiceError, null);
      }
    },

    listSessions(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { user_id } = call.request as { user_id?: string };
      // Empty user_id keeps the old open behavior (list everything, dev only).
      const owned = user_id ? sessions.list().filter((s) => s.userId === user_id) : sessions.list();
      callback(null, {
        sessions: owned.map((s) => ({
          id: s.id,
          user_id: s.userId,
          status: s.status,
          created_at: s.createdAt.toISOString(),
          last_active: s.lastActive.toISOString(),
        })),
      });
    },

    chat(call: grpc.ServerWritableStream<any, any>) {
      const { session_id, content, user_id } = call.request as {
        session_id?: string;
        content?: string;
        user_id?: string;
      };
      let session;
      try {
        session = sessions.get(session_id ?? "");
        assertOwner(session, user_id);
      } catch (err) {
        failStream(call, err as Error & { code: grpc.status });
        return;
      }
      const { agent, llm } = session.agent;
      if (agent.state.isStreaming) {
        failStream(
          call,
          grpcError(grpc.status.FAILED_PRECONDITION, "session already has a chat turn in progress"),
        );
        return;
      }
      if (!llm.apiKey) {
        call.write({
          error: {
            message: "LLM is not configured: no API key (pass llm.api_key in CreateSession or set LLM_API_KEY)",
            retryable: false,
          },
        });
        call.end();
        return;
      }
      sessions.touch(session.id);

      let inputTokens = 0;
      let outputTokens = 0;
      let turnFailed = false;
      // Transcript collection for the session-manager write-through: assistant
      // text is aggregated into one message; tool events are recorded as-is.
      let assistantText = "";
      const toolMessages: TurnMessageInput[] = [];

      const unsubscribe = agent.subscribe((event) => {
        switch (event.type) {
          case "message_update":
            if (event.assistantMessageEvent.type === "text_delta") {
              assistantText += event.assistantMessageEvent.delta;
              call.write({ text_delta: { delta: event.assistantMessageEvent.delta } });
            }
            break;
          case "tool_execution_start": {
            const payload = { tool: event.toolName, arguments_json: JSON.stringify(event.args ?? {}) };
            toolMessages.push({ role: "tool_call", contentJson: JSON.stringify(payload), createdAt: new Date() });
            call.write({ tool_call: payload });
            break;
          }
          case "tool_execution_end": {
            const text = toolResultText(event.result);
            const payload = {
              tool: event.toolName,
              ok: !event.isError,
              output: event.isError ? "" : text,
              error: event.isError ? text : "",
            };
            toolMessages.push({ role: "tool_result", contentJson: JSON.stringify(payload), createdAt: new Date() });
            call.write({ tool_result: payload });
            break;
          }
          case "message_end": {
            const message = event.message;
            if (message.role !== "assistant") break;
            if (message.stopReason === "error" || message.stopReason === "aborted") {
              turnFailed = true;
              call.write({
                error: {
                  message: message.errorMessage ?? "LLM request failed",
                  retryable: isRetryableLlmError(message.errorMessage),
                },
              });
            } else if (message.usage) {
              inputTokens += message.usage.input;
              outputTokens += message.usage.output;
            }
            break;
          }
        }
      });

      agent
        .prompt(content ?? "")
        .then(() => {
          // Skip the write-through on error/abort turns and on client disconnect.
          if (!turnFailed && !call.cancelled) {
            call.write({ done: { input_tokens: inputTokens, output_tokens: outputTokens } });
            const now = new Date();
            const turn: TurnMessageInput[] = [
              { role: "user", contentJson: JSON.stringify({ content: content ?? "" }), createdAt: now },
            ];
            if (assistantText) {
              turn.push({ role: "assistant", contentJson: JSON.stringify({ content: assistantText }), createdAt: now });
            }
            turn.push(...toolMessages);
            sessionWriter?.appendTurn(session.id, session.userId, turn);
          }
          call.end();
        })
        .catch((err: unknown) => {
          try {
            call.write({ error: { message: err instanceof Error ? err.message : String(err), retryable: false } });
          } finally {
            call.end();
          }
        })
        .finally(() => {
          unsubscribe();
          sessions.touch(session.id);
        });
    },
  };

  const server = new grpc.Server();
  server.addService(proto.runtime.v1.AgentService.service, service);
  return { server, sessions };
}

async function main(): Promise<void> {
  const config = loadConfig();
  const { server, sessions } = createRuntime(config);
  sessions.start();

  const address = `0.0.0.0:${config.port}`;
  const boundPort = await new Promise<number>((resolve, reject) => {
    server.bindAsync(address, grpc.ServerCredentials.createInsecure(), (err, port) => {
      if (err) reject(err);
      else resolve(port);
    });
  });
  console.log(`agent-runtime listening on ${address} (port ${boundPort})`);

  const shutdown = () => {
    sessions.stop();
    server.tryShutdown(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
