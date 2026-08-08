import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Agent, type AgentMessage } from "@earendil-works/pi-agent-core";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { loadConfig, type ServiceConfig } from "./config.js";
import { extractTranscriptLlm, extractTranscriptSystemPrompt, transcriptToMessages } from "./hydrate.js";
import { createModel, createStreamFn, type SessionLlmConfig } from "./llm.js";
import { createSessionClient, type TranscriptTurn, type TurnMessageInput } from "./session-client.js";
import { SessionManager, assertOwner, grpcError } from "./session-manager.js";
import { generateTitle, sanitizeTitle, type TitleGenerator } from "./title.js";
import { buildTools, NullExecutor, type ToolExecutor } from "./tools.js";

interface RuntimeSession {
  agent: Agent;
  llm: SessionLlmConfig;
  /** True until the first completed Chat turn has been offered a generated title. */
  titlePending: boolean;
  /**
   * The system prompt the live agent runs with: the CreateSession request's
   * system_prompt when non-empty, else the transcript's last recorded system
   * message, else "" (none). Also what AppendTurn considers "current".
   */
  systemPrompt: string;
  /**
   * Last system prompt recorded in the durable transcript. Seeded from the
   * transcript's last system message during hydration so an unchanged prompt
   * is not re-recorded; undefined when nothing has been recorded yet.
   */
  lastRecordedSystemPrompt?: string;
  /**
   * JSON of the last LLM triple recorded in the durable transcript as a
   * "config" turn (`{api_key, base_url, model}`). Seeded from the
   * transcript's last config turn during hydration so an unchanged triple is
   * not re-recorded; undefined when nothing has been recorded yet.
   */
  lastRecordedLlm?: string;
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

export interface RuntimeOptions {
  /**
   * One-shot title generator for auto session titles. Injectable seam for
   * tests; defaults to a direct model call seeded with the first user message.
   */
  titleGenerator?: TitleGenerator;
}

export function createRuntime(
  config: ServiceConfig,
  executor: ToolExecutor = new NullExecutor(),
  options: RuntimeOptions = {},
): Runtime {
  const titleGenerator = options.titleGenerator ?? generateTitle;
  const sessions = new SessionManager<RuntimeSession>({
    maxSessions: config.maxSessions,
    ttlMs: config.sessionTtlMinutes * 60_000,
    destroyAgent: ({ agent }) => agent.abort(),
  });
  const sessionClient = createSessionClient(config);

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
    async createSession(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { user_id, session_id, llm, system_prompt, tools: toolNames } = call.request as {
        user_id?: string;
        session_id?: string;
        llm?: { api_key?: string; base_url?: string; model?: string };
        system_prompt?: string;
        tools?: string[];
      };
      if (!user_id) {
        callback(grpcError(grpc.status.INVALID_ARGUMENT, "user_id is required") as grpc.ServiceError, null);
        return;
      }
      // Optional tool whitelist: empty/absent = all built-ins; unknown names
      // are rejected before any session state (or hydration fetch) happens.
      const allTools = buildTools(executor);
      let sessionTools = allTools;
      if (toolNames && toolNames.length > 0) {
        const known = new Set(allTools.map((t) => t.name));
        const unknown = toolNames.filter((name) => !known.has(name));
        if (unknown.length > 0) {
          callback(
            grpcError(grpc.status.INVALID_ARGUMENT, `unknown tools: ${unknown.join(", ")}`) as grpc.ServiceError,
            null,
          );
          return;
        }
        sessionTools = allTools.filter((t) => toolNames.includes(t.name));
      }
      try {
        // Hydration fetch first: a caller-provided session id may have a
        // durable transcript in session-manager (e.g. after a runtime
        // restart), and the transcript may carry the session's FROZEN LLM
        // triple (role "config" turns), which must be known before the model
        // is built. Fail-open: any problem leaves the session with empty
        // history.
        let transcriptTurns: TranscriptTurn[] | undefined;
        let transcriptHasMore = false;
        if (session_id && sessionClient) {
          try {
            // Windowed hydration: fetch only the latest HYDRATION_MAX_TURNS
            // transcript messages (before_seq 0 = "from the end") so resume
            // cost stays bounded as sessions grow. transcriptToMessages needs
            // no changes — its existing leniency already skips an orphan
            // tool_result at the window's start or a dangling tool_call at
            // its end. The system-prompt and frozen-LLM markers usually sit
            // at the transcript's HEAD, so a truncated window may contain
            // none — recovered below by a markers-only full fetch.
            const transcript = await sessionClient.getTranscript(session_id, {
              limit: config.hydrationMaxTurns,
              beforeSeq: 0,
            });
            if (transcript && transcript.messages.length > 0) {
              transcriptTurns = transcript.messages;
              transcriptHasMore = transcript.hasMore;
            }
          } catch (err) {
            console.warn(`hydration failed for ${session_id}, starting with empty history:`, err);
          }
        }
        // Marker recovery: system/config turns are written at the head of
        // the transcript (and only when they change), so a long session's
        // window may contain none — which would silently drop the frozen-LLM
        // guarantee (the request llm would take over and the session could
        // switch models) and the deleted-template prompt fallback. When the
        // window is truncated and a marker is missing, fetch the full
        // transcript once and extract ONLY the markers from it; the
        // conversation context stays limited to the window. Fail-open like
        // the windowed fetch.
        let markerTurns = transcriptTurns;
        if (transcriptTurns && transcriptHasMore && sessionClient && session_id) {
          const windowHasConfig = extractTranscriptLlm(transcriptTurns) !== undefined;
          const windowHasSystem = transcriptTurns.some((t) => t.role === "system");
          if (!windowHasConfig || !windowHasSystem) {
            try {
              const full = await sessionClient.getTranscript(session_id); // limit 0 = full
              if (full && full.messages.length > 0) {
                markerTurns = full.messages;
                console.log(
                  `hydration for session ${session_id}: recovered system/config markers from beyond the window`,
                );
              }
            } catch (err) {
              console.warn(`marker recovery failed for ${session_id}, using window markers only:`, err);
            }
          }
        }
        // LLM precedence — deliberately ASYMMETRIC vs the system prompt
        // below: the transcript's frozen config turn WINS over the request
        // llm (snapshot semantics: once a session starts, its model must
        // never switch — editing the agent template's LLM or the platform
        // default must not affect existing sessions), while for the system
        // prompt the request value wins (reference semantics: a live
        // template's current prompt governs). The request/env resolution is
        // the fallback for pre-config-turn transcripts and fresh sessions.
        // A config turn with an empty api_key does NOT take over — it merely
        // records that no key was resolved at the time.
        const frozenLlm = markerTurns ? extractTranscriptLlm(markerTurns) : undefined;
        const resolved: SessionLlmConfig =
          frozenLlm && frozenLlm.api_key
            ? { apiKey: frozenLlm.api_key, baseUrl: frozenLlm.base_url, model: frozenLlm.model }
            : resolveLlm(llm);
        const model = createModel(resolved);
        // Convert the fetched transcript WINDOW into pi messages, stamped
        // with the resolved model's identity, so it can seed
        // initialState.messages. The system-prompt marker comes from
        // markerTurns (the window, or the recovered full transcript).
        let history: AgentMessage[] | undefined;
        let transcriptSystemPrompt = markerTurns ? extractTranscriptSystemPrompt(markerTurns) : "";
        if (transcriptTurns) {
          const hydrated = transcriptToMessages(transcriptTurns, {
            api: model.api,
            provider: model.provider,
            model: model.id,
          });
          if (hydrated.messages.length > 0) {
            history = hydrated.messages;
            console.log(`hydrated session ${session_id} with ${hydrated.messages.length} messages from transcript`);
          }
          if (transcriptHasMore) {
            console.log(
              `hydration for session ${session_id}: earlier transcript history skipped ` +
                `(window limited to the latest ${config.hydrationMaxTurns} messages)`,
            );
          }
        }
        // System prompt precedence: a non-empty request system_prompt wins
        // (live template's current value); otherwise the transcript's last
        // recorded system message takes over (template deleted since); an
        // empty transcript means no system prompt. The runtime carries no
        // built-in default (pi-ai omits the system message for a falsy value).
        // Same code path for fresh and hydrated (resumed) sessions.
        const systemPrompt = system_prompt || transcriptSystemPrompt;
        const session = sessions.create(
          user_id,
          (sessionId) => ({
            llm: resolved,
            // Only a session that began empty gets an auto title after its
            // first completed turn; hydrated/history-seeded sessions don't.
            titlePending: !history,
            systemPrompt,
            // Seed the "last recorded" markers from the transcript so an
            // unchanged prompt/triple is not re-recorded on the next
            // AppendTurn. A transcript without a config turn leaves the LLM
            // marker undefined, so the first AppendTurn records the triple.
            lastRecordedSystemPrompt: transcriptSystemPrompt || undefined,
            lastRecordedLlm: frozenLlm ? JSON.stringify(frozenLlm) : undefined,
            agent: new Agent({
              initialState: {
                systemPrompt,
                model,
                tools: sessionTools,
                ...(history ? { messages: history } : {}),
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
            const turn: TurnMessageInput[] = [];
            // Persist the session's current system prompt as a transcript
            // "system" message whenever it differs from the last one recorded
            // (first turn of a fresh session, or a changed prompt after a
            // resume). AppendTurn is fire-and-log, so the marker is updated
            // optimistically. An empty prompt is never recorded.
            if (session.agent.systemPrompt && session.agent.systemPrompt !== session.agent.lastRecordedSystemPrompt) {
              turn.push({
                role: "system",
                contentJson: JSON.stringify({ content: session.agent.systemPrompt }),
                createdAt: now,
              });
              session.agent.lastRecordedSystemPrompt = session.agent.systemPrompt;
            }
            // Persist the session's FROZEN LLM triple as a transcript
            // "config" message whenever it differs from the last one
            // recorded (first turn of a fresh session, or a pre-config-turn
            // transcript resumed with a different resolution). Placed after
            // the system message, before the user message. The triple is
            // recorded faithfully as-is — api_key may be "" (no key
            // resolved); AppendTurn is fire-and-log, so the marker is
            // updated optimistically.
            const llmSnapshot = {
              api_key: session.agent.llm.apiKey ?? "",
              base_url: session.agent.llm.baseUrl,
              model: session.agent.llm.model,
            };
            const llmJson = JSON.stringify(llmSnapshot);
            if (llmJson !== session.agent.lastRecordedLlm) {
              turn.push({ role: "config", contentJson: JSON.stringify({ llm: llmSnapshot }), createdAt: now });
              session.agent.lastRecordedLlm = llmJson;
            }
            turn.push({ role: "user", contentJson: JSON.stringify({ content: content ?? "" }), createdAt: now });
            if (assistantText) {
              // Persist the turn's token usage (the same counts sent in the
              // SSE done frame) next to the assistant text so the console can
              // show historical usage; the key is omitted when the provider
              // reported none.
              const payload: { content: string; usage?: { input_tokens: number; output_tokens: number } } = {
                content: assistantText,
              };
              if (inputTokens > 0 || outputTokens > 0) {
                payload.usage = { input_tokens: inputTokens, output_tokens: outputTokens };
              }
              turn.push({ role: "assistant", contentJson: JSON.stringify(payload), createdAt: now });
            }
            turn.push(...toolMessages);
            sessionClient?.appendTurn(session.id, session.userId, turn);
            // Auto session title: only the first completed turn of a session
            // that began empty. Fire-and-log — detached from the chat stream,
            // and any failure (LLM or session-manager) only warns.
            if (sessionClient && session.agent.titlePending) {
              session.agent.titlePending = false;
              const llm = session.agent.llm;
              const generate = () => titleGenerator(llm, content ?? "");
              // One delayed retry: the free-tier default model 429s often
              // enough that a single attempt is noticeably lossy.
              const retryOnce = () =>
                new Promise<string>((resolve, reject) => {
                  setTimeout(() => {
                    generate().then(resolve, reject);
                  }, 30_000);
                });
              void generate()
                .catch(() => retryOnce())
                .then((raw) => {
                  const title = sanitizeTitle(raw);
                  if (title) sessionClient.setTitle(session.id, title);
                })
                .catch((err: unknown) => console.warn(`title generation failed for ${session.id}:`, err));
            }
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
