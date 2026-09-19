import { randomUUID } from "node:crypto";
import { Agent, type AgentMessage, type StreamFn } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai";
import * as grpc from "@grpc/grpc-js";
import {
  DEFAULT_TRUNCATION,
  historyToPi,
  text,
  unresolvedToolCallsInHistory,
  usageFromPi,
  type CanonicalBlock,
  type CanonicalMessage,
  type CanonicalUsage,
  type TruncationPolicy,
} from "./canonical.js";
import { compact, summaryText } from "./compaction.js";
import type { ServiceConfig } from "./config.js";
import {
  DurableRpcError,
  canonicalBlocksFromProto,
  createDurableSessionClient,
  type DurableCheckpointRecord,
  type DurableConfigRecord,
  type DurableSessionClient,
  type MessageDraftInput,
} from "./durable-client.js";
import { DurableExecution } from "./durable-execution.js";
import { createModel, createStreamFn, type SessionLlmConfig } from "./llm.js";
import { aggregateUsage, translateAgentEvent, type V2StreamEvent } from "./runtime-events.js";
import {
  elapsedMs,
  jsonLineSink,
  recordCompaction,
  recordLeaseConflict,
  recordUnresolvedToolCalls,
  type TelemetrySink,
} from "./telemetry.js";
import {
  ContextEstimator,
  DEFAULT_BUDGET,
  providerMessages,
  shouldCompact,
  type BudgetConfig,
  type ToolSchema,
} from "./tokens.js";
import { NullExecutor, buildTools, type ToolExecutor } from "./tools.js";

/**
 * runtime.v2.AgentService: durable execution of one request at a time.
 *
 * Canonical history, session configuration, and request state all live in
 * session-manager; this service owns only disposable in-memory state (the live
 * pi agent and the canonical view backing it). pi's tool hooks are the exact
 * seams the design needs: `beforeToolCall` persists the assistant tool-call
 * message before the tool runs, `afterToolCall` persists the observed result
 * before the next model call, and `prepareNextTurnWithContext` checks the input
 * budget and compacts between model calls.
 */

const BASE_BUDGET = DEFAULT_BUDGET;

/** Bounds hydration paging so a pathological session cannot loop forever. */
const MAX_HYDRATION_PAGES = 20;

interface LiveSession {
  agent: Agent;
  config: DurableConfigRecord;
  llm: SessionLlmConfig;
  estimator: ContextEstimator;
  budget: BudgetConfig;
  /** The canonical messages the model context currently represents. */
  canonical: CanonicalMessage[];
  /** Tool schemas as the estimator sees them. */
  schemas: ToolSchema[];
  lastCheckpoint?: DurableCheckpointRecord;
}

export interface DurableRuntimeOptions {
  /** Injectable durable client; defaults to the gRPC client from the config. */
  client?: DurableSessionClient;
  /** Where structured operational records go; defaults to one JSON line each. */
  telemetry?: TelemetrySink;
  budget?: Partial<BudgetConfig>;
  /**
   * Builds the model and stream function for a session. Tests inject a scripted
   * stream so the durable boundaries can be driven without a provider.
   */
  modelFactory?: (llm: SessionLlmConfig) => { model: Model<any>; streamFn: StreamFn };
  /** Summarizer used by compaction; tests inject a deterministic one. */
  summarizer?: (prompt: string, live: { llm: SessionLlmConfig }) => Promise<{ text: string; usage?: CanonicalUsage }>;
}

export interface DurableRuntime {
  service: grpc.UntypedServiceImplementation;
  sessions: Map<string, LiveSession>;
  close(): void;
}

/** AssistantMessage -> canonical blocks, preserving order and tool-call ids. */
function assistantBlocks(message: AssistantMessage): CanonicalBlock[] {
  const out: CanonicalBlock[] = [];
  for (const part of message.content) {
    if (part.type === "text") out.push(text(part.text));
    else if (part.type === "toolCall") {
      out.push({
        type: "tool_call",
        id: part.id,
        name: part.name,
        arguments_json: JSON.stringify(part.arguments ?? {}),
      });
    }
  }
  return out.length > 0 ? out : [text("")];
}

/** Tool execution result -> canonical result content. */
function resultBlocks(result: unknown): CanonicalBlock[] {
  const content = (result as { content?: unknown })?.content;
  const parts: CanonicalBlock[] = [];
  if (Array.isArray(content)) {
    for (const part of content) {
      if (part && typeof part === "object" && (part as { type?: unknown }).type === "text") {
        parts.push(text(String((part as { text?: unknown }).text ?? "")));
      }
    }
  }
  if (parts.length === 0) {
    parts.push(text(typeof result === "string" ? result : JSON.stringify(result ?? null)));
  }
  return parts;
}

/** Stable canonical id for an assistant message within this run. */
function assistantId(message: AssistantMessage): string {
  return `asst-${message.timestamp}`;
}

function toSchemas(tools: Array<{ name: string; description?: string; parameters?: unknown }>): ToolSchema[] {
  return tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters }));
}

function failStream(call: grpc.ServerWritableStream<unknown, unknown>, err: unknown): void {
  let code: grpc.status = (err as { code?: grpc.status }).code ?? grpc.status.INTERNAL;
  if (err instanceof DurableRpcError) {
    switch (err.kind) {
      case "conflict":
      case "not_found":
        code = grpc.status.FAILED_PRECONDITION;
        break;
      case "aborted":
        code = grpc.status.ABORTED;
        break;
      case "invalid":
        code = grpc.status.INVALID_ARGUMENT;
        break;
      case "denied":
        code = grpc.status.PERMISSION_DENIED;
        break;
      default:
        code = grpc.status.INTERNAL;
    }
  }
  call.emit("error", Object.assign(err instanceof Error ? err : new Error(String(err)), { code }));
}

/** Text-only summarizer over the session's frozen model. */
async function summarize(live: LiveSession, prompt: string): Promise<{ text: string; usage?: CanonicalUsage }> {
  const streamFn = createStreamFn(live.llm);
  const model = createModel(live.llm);
  const stream = await streamFn(model, {
    systemPrompt: "You summarize conversation history faithfully and concisely. Never invent details.",
    messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }],
  });
  let out = "";
  let usage: CanonicalUsage | undefined;
  for await (const event of stream) {
    if (event.type === "text_delta") out += event.delta;
    else if (event.type === "done") usage = usageFromPi(event.message.usage);
  }
  return { text: out, usage };
}

/** Rebuild the model-facing context from a checkpoint plus retained messages. */
function contextMessages(
  live: LiveSession,
  checkpoint: DurableCheckpointRecord,
  retained: CanonicalMessage[],
  truncationPolicy: TruncationPolicy,
): AgentMessage[] {
  const messages: AgentMessage[] = [];
  const summary = summaryText(checkpoint);
  if (summary) {
    messages.push({
      role: "user",
      content: [{ type: "text", text: `[Earlier conversation summary]\n${summary}` }],
      timestamp: Date.now(),
    });
  }
  const model = createModel(live.llm);
  messages.push(
    ...historyToPi(retained, { api: model.api, provider: model.provider, model: model.id }, truncationPolicy),
  );
  return messages;
}

export function createDurableRuntime(
  config: ServiceConfig,
  executor: ToolExecutor = new NullExecutor(),
  options: DurableRuntimeOptions = {},
): DurableRuntime {
  const client = options.client ?? createDurableSessionClient(config);
  const sessions = new Map<string, LiveSession>();
  const modelFactory =
    options.modelFactory ?? ((llm: SessionLlmConfig) => ({ model: createModel(llm), streamFn: createStreamFn(llm) }));
  const telemetry = options.telemetry ?? jsonLineSink;
  const toolResultMaxChars = config.toolResultMaxChars ?? DEFAULT_TRUNCATION.maxToolResultChars;
  const truncationPolicy: TruncationPolicy =
    toolResultMaxChars > 0 ? { maxToolResultChars: toolResultMaxChars } : DEFAULT_TRUNCATION;

  if (!client) {
    // SESSION_MANAGER_ADDR unset: v2 has no durable boundary at all, so every
    // request fails loudly rather than running without persistence.
    const detail = "session-manager is not configured";
    const service: grpc.UntypedServiceImplementation = {
      chat(call: grpc.ServerWritableStream<any, any>) {
        call.emit("error", Object.assign(new Error(detail), { code: grpc.status.UNAVAILABLE }));
      },
      cancelRequest(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
        callback({ code: grpc.status.UNAVAILABLE, details: detail } as grpc.ServiceError, null);
      },
      evictSession(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
        callback(null, {});
      },
    };
    return { service, sessions, close: () => {} };
  }
  const durable: DurableSessionClient = client;

  async function liveSession(sessionId: string, frozen: DurableConfigRecord): Promise<LiveSession> {
    const existing = sessions.get(sessionId);
    if (existing) return existing;

    const context = await durable.getExecutionContext(sessionId);
    const checkpoint = context.checkpoint;

    // Deterministic hydration: page backwards until the checkpoint cutoff (or
    // the start of history) is covered, so pagination never silently drops the
    // retained prefix.
    const collected: CanonicalMessage[] = [];
    let beforeSeq = 0;
    let hasMore = true;
    let pages = 0;
    while (hasMore && pages < MAX_HYDRATION_PAGES) {
      const page = await durable.getMessages(sessionId, { limit: config.hydrationMaxTurns, beforeSeq });
      if (page.messages.length === 0) break;
      collected.unshift(...page.messages);
      hasMore = page.hasMore;
      beforeSeq = page.messages[0].seq;
      pages++;
      if (checkpoint && beforeSeq <= checkpoint.coveredThroughSeq + 1) break;
    }
    const retained = checkpoint ? collected.filter((message) => message.seq > checkpoint.coveredThroughSeq) : collected;

    // A crash can leave a tool call persisted without its result. The outcome
    // is unknown, so those calls are excluded from model context (see
    // historyToPi) and reported: the turn needs an explicit retry decision
    // rather than a silent replay.
    for (const dangling of unresolvedToolCallsInHistory(retained)) {
      recordUnresolvedToolCalls(telemetry, {
        sessionId,
        assistantMessageId: dangling.assistantMessageId,
        toolCallCount: dangling.toolCallIds.length,
      });
    }

    const llm: SessionLlmConfig = {
      // The frozen configuration carries a credential reference; until private
      // credential storage exists the process key supplies the secret.
      apiKey: config.llm.apiKey,
      baseUrl: frozen.baseUrl || config.llm.baseUrl,
      model: frozen.model || config.llm.model,
    };
    const { model, streamFn } = modelFactory(llm);
    const identity = { api: model.api, provider: model.provider, model: model.id };

    const messages: AgentMessage[] = [];
    const previous = summaryText(checkpoint);
    if (previous) {
      messages.push({
        role: "user",
        content: [{ type: "text", text: `[Earlier conversation summary]\n${previous}` }],
        timestamp: Date.now(),
      });
    }
    messages.push(...historyToPi(retained, identity, truncationPolicy));

    const allTools = buildTools(executor);
    const wanted = (frozen.tools ?? [])
      .map((tool) => String((tool as { name?: unknown }).name ?? ""))
      .filter((name) => name.length > 0);
    const sessionTools = wanted.length > 0 ? allTools.filter((tool) => wanted.includes(tool.name)) : allTools;

    const live: LiveSession = {
      config: frozen,
      llm,
      estimator: new ContextEstimator(),
      budget: {
        ...BASE_BUDGET,
        ...options.budget,
        contextWindow: frozen.contextWindowTokens > 0 ? frozen.contextWindowTokens : 128_000,
        outputReserve: frozen.outputLimitTokens > 0 ? frozen.outputLimitTokens : 16_384,
      },
      canonical: retained,
      schemas: toSchemas(sessionTools as Array<{ name: string; description?: string; parameters?: unknown }>),
      lastCheckpoint: checkpoint,
      agent: undefined as unknown as Agent,
    };
    live.agent = new Agent({
      initialState: {
        systemPrompt: frozen.systemPrompt,
        model,
        tools: sessionTools,
        ...(messages.length > 0 ? { messages } : {}),
      },
      streamFn,
      sessionId,
    });
    sessions.set(sessionId, live);
    console.log(`hydrated durable session ${sessionId} with ${retained.length} canonical messages`);
    return live;
  }

  async function runChat(call: grpc.ServerWritableStream<any, any>): Promise<void> {
    const request = call.request as {
      session_id?: string;
      user_id?: string;
      client_request_id?: string;
      content?: unknown;
    };
    const sessionId = request.session_id ?? "";
    const userId = request.user_id ?? "";
    const clientRequestId = request.client_request_id ?? "";
    const content = canonicalBlocksFromProto(request.content);
    if (!sessionId || !userId || !clientRequestId || content.length === 0) {
      failStream(
        call,
        Object.assign(new Error("session_id, user_id, client_request_id and content are required"), {
          code: grpc.status.INVALID_ARGUMENT,
        }),
      );
      return;
    }

    let execution: DurableExecution | undefined;
    try {
      const context = await durable.getExecutionContext(sessionId);
      if (context.config.formatVersion <= 0) {
        failStream(
          call,
          Object.assign(new Error("session has no frozen configuration"), { code: grpc.status.FAILED_PRECONDITION }),
        );
        return;
      }
      const live = await liveSession(sessionId, context.config);
      if (live.agent.state.isStreaming) {
        failStream(
          call,
          Object.assign(new Error("session already has a request in progress"), { code: grpc.status.FAILED_PRECONDITION }),
        );
        return;
      }

      execution = await DurableExecution.start({
        client: durable,
        sessionId,
        userId,
        clientRequestId,
        content,
        telemetry,
        onLeaseLost: (err) => {
          recordLeaseConflict(telemetry, { sessionId, requestMessageId: execution?.requestMessageId, reason: err.message });
          live.agent.abort();
        },
      });
      const active = execution;

      // A deduplicated admission is a reconnect: the root already exists, so
      // report its state instead of rerunning tools.
      if (active.deduplicated) {
        call.write({
          done: {
            request_message_id: active.requestMessageId,
            status:
              active.session.activeRequestMessageId === active.requestMessageId
                ? "EXECUTION_STATUS_RUNNING"
                : "EXECUTION_STATUS_COMPLETED",
            revision: active.revision,
          },
        });
        call.end();
        return;
      }

      // The request root is part of what the model sees; without it the
      // estimator would under-count the input by the whole user message and
      // compaction would fire late or never.
      live.canonical.push(active.requestMessage);

      let persistedAssistant: string | undefined;
      let finalAssistant: AssistantMessage | undefined;
      const streamMessageId = `asst-${Date.now()}`;

      // Durable boundary 1: persist the assistant tool-call message before the
      // tool is dispatched, so a crash cannot lose the fact that it ran.
      live.agent.beforeToolCall = async (hook) => {
        active.assertLeaseHeld();
        const id = assistantId(hook.assistantMessage);
        if (persistedAssistant !== id) {
          persistedAssistant = id;
          const stored = await active.persist([
            {
              id,
              role: "assistant",
              content: assistantBlocks(hook.assistantMessage),
              status: "complete",
              usage: usageFromPi(hook.assistantMessage.usage),
            },
          ]);
          live.canonical.push(...stored);
        }
        return undefined;
      };

      // Durable boundary 2: persist the observed result before the next model
      // call, linked to its call by the original tool-call id.
      live.agent.afterToolCall = async (hook) => {
        active.assertLeaseHeld();
        const stored = await active.persist([
          {
            id: `tool-${hook.toolCall.id}`,
            role: "tool",
            status: "complete",
            replyToMessageId: assistantId(hook.assistantMessage),
            content: [
              {
                type: "tool_result",
                tool_call_id: hook.toolCall.id,
                status: hook.isError ? "error" : "success",
                content: resultBlocks(hook.result),
              },
            ],
            providerMetadata: { tool_names: { [hook.toolCall.id]: hook.toolCall.name } },
          },
        ]);
        live.canonical.push(...stored);
        return undefined;
      };

      // Durable boundary 3: budget check before every model call, and
      // compaction at a safe cutoff when the context no longer fits.
      live.agent.prepareNextTurnWithContext = async (hook) => {
        active.assertLeaseHeld();
        const shape = {
          systemPrompt: hook.context.systemPrompt,
          tools: live.schemas,
          summary: summaryText(live.lastCheckpoint),
          messages: providerMessages(live.canonical),
        };
        if (!shouldCompact(live.estimator.estimate(shape), live.budget)) return undefined;
        const startedAt = Date.now();
        try {
          const result = await compact({
            publisher: active,
            summarizer: (prompt) =>
              options.summarizer ? options.summarizer(prompt, live) : summarize(live, prompt),
            messages: live.canonical,
            budget: live.budget,
            active: live.lastCheckpoint,
            configHash: live.config.configHash,
            activeRequestMessageId: active.requestMessageId,
            resumeAfterSeq: active.session.lastSeq,
            sourceRevision: active.revision,
            checkpointId: `cp-${randomUUID()}`,
            provider: live.config.provider,
            model: live.llm.model,
          });
          live.lastCheckpoint = result.checkpoint;
          live.canonical = result.cutoff.retained;
          live.estimator.invalidate();
          recordCompaction(telemetry, {
            sessionId,
            requestMessageId: active.requestMessageId,
            checkpointId: result.checkpoint.id,
            coveredThroughSeq: result.checkpoint.coveredThroughSeq,
            sourceRevision: result.checkpoint.sourceRevision,
            estimatedTokensBefore: result.checkpoint.estimatedTokensBefore,
            estimatedTokensAfter: result.checkpoint.estimatedTokensAfter,
            durationMs: elapsedMs(startedAt),
            summarizerModel: result.checkpoint.summarizerModel,
            promptVersion: result.checkpoint.promptVersion,
          });
          return {
            context: {
              ...hook.context,
              messages: contextMessages(live, result.checkpoint, result.cutoff.retained, truncationPolicy),
            },
          };
        } catch (err) {
          // Leave the previous checkpoint active; the request continues with the
          // context it already has, and the next call re-checks the budget.
          console.warn(`compaction failed for ${sessionId}:`, err);
          return undefined;
        }
      };

      const usageCalls: Array<{ input: number; output: number } | undefined> = [];
      let failure: { code: string; message: string; retryable: boolean } | undefined;

      const unsubscribe = live.agent.subscribe((event) => {
        const translation = translateAgentEvent(event, streamMessageId);
        for (const frame of translation.events) {
          call.write({ [frame.kind]: frame.payload });
          if (frame.kind === "error") {
            failure = { code: frame.payload.code, message: frame.payload.message, retryable: frame.payload.retryable };
          }
        }
        if (translation.usageDelta) usageCalls.push(translation.usageDelta);
        if (event.type === "message_end" && event.message?.role === "assistant") {
          const message = event.message as AssistantMessage;
          if (message.stopReason !== "error" && message.stopReason !== "aborted") {
            finalAssistant = message;
            live.estimator.recordActualUsage(usageFromPi(message.usage), {
              systemPrompt: live.config.systemPrompt,
              tools: live.schemas,
              summary: summaryText(live.lastCheckpoint),
              messages: [],
            });
          }
        }
      });

      const promptText = content
        .map((block) => (block.type === "text" ? block.text : ""))
        .join("\n")
        .trim();

      try {
        await live.agent.prompt(promptText);
        if (failure) {
          // The failure frame was already streamed; record the terminal state.
          await active.finish("failed", [], failure.code);
        } else if (call.cancelled) {
          await active.finish("cancelled", [], "client_disconnected");
        } else {
          const finalBatch: MessageDraftInput[] = [];
          if (finalAssistant && persistedAssistant !== assistantId(finalAssistant)) {
            finalBatch.push({
              id: assistantId(finalAssistant),
              role: "assistant",
              content: assistantBlocks(finalAssistant),
              status: "complete",
              usage: usageFromPi(finalAssistant.usage),
            });
          }
          const stored = await active.finish("completed", finalBatch);
          live.canonical.push(...stored);
          call.write({
            done: {
              request_message_id: active.requestMessageId,
              status: "EXECUTION_STATUS_COMPLETED",
              revision: active.revision,
              aggregate_usage: aggregateUsage(usageCalls),
            },
          });
        }
      } finally {
        unsubscribe();
      }
      call.end();
    } catch (err) {
      if (execution && !execution.leaseLost) {
        try {
          const cancelled = call.cancelled;
          await execution.finish(cancelled ? "cancelled" : "failed", [], cancelled ? "client_disconnected" : "runtime_error");
        } catch {
          // The terminal write is best-effort; the caller still needs the error.
        }
      }
      failStream(call, err);
    }
  }

  const service: grpc.UntypedServiceImplementation = {
    chat(call: grpc.ServerWritableStream<any, any>) {
      void runChat(call);
    },

    cancelRequest(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { session_id, user_id, request_message_id } = call.request as {
        session_id?: string;
        user_id?: string;
        request_message_id?: string;
      };
      void (async () => {
        try {
          await durable.cancelRequest({
            sessionId: session_id ?? "",
            userId: user_id,
            requestMessageId: request_message_id,
          });
          // Stop local inference; the lease holder still writes the terminal state.
          sessions.get(session_id ?? "")?.agent.abort();
          callback(null, { accepted: true });
        } catch (err) {
          callback(
            Object.assign(err instanceof Error ? err : new Error(String(err)), {
              code:
                err instanceof DurableRpcError && err.kind === "conflict"
                  ? grpc.status.FAILED_PRECONDITION
                  : grpc.status.INTERNAL,
            }) as grpc.ServiceError,
            null,
          );
        }
      })();
    },

    evictSession(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      const { session_id } = call.request as { session_id?: string };
      const live = sessions.get(session_id ?? "");
      if (live) {
        live.agent.abort();
        sessions.delete(session_id ?? "");
      }
      callback(null, {});
    },
  };

  return { service, sessions, close: () => durable.close() };
}

export type { V2StreamEvent };
