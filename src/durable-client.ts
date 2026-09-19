import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import {
  FORMAT_VERSION,
  type CanonicalBlock,
  type CanonicalMessage,
  type CanonicalMessageStatus,
  type CanonicalUsage,
  type RequestExecution,
  type ToolResultStatus,
} from "./canonical.js";
import type { ServiceConfig } from "./config.js";

/**
 * Client for session.v2.SessionService — the runtime's durable boundary.
 *
 * Unlike the v1 client, nothing here is fire-and-forget: admission, message
 * batches, terminal transitions, and checkpoint publication are awaited and
 * their outcomes decided by the caller, because the design requires an
 * acknowledged durable boundary before the next model or tool step.
 */

export interface DurableSessionRecord {
  id: string;
  userId: string;
  status: string;
  agentId: string;
  title: string;
  lastSeq: number;
  revision: number;
  activeCheckpointId: string;
  activeRequestMessageId: string;
  createdAt?: string;
  lastActive?: string;
  endedAt?: string;
}

export interface DurableConfigRecord {
  sessionId: string;
  formatVersion: number;
  configHash: string;
  systemPrompt: string;
  provider: string;
  model: string;
  baseUrl: string;
  credentialRef: string;
  contextWindowTokens: number;
  outputLimitTokens: number;
  tools: unknown[];
  createdAt?: string;
}

export interface DurableCheckpointRecord {
  id: string;
  sessionId: string;
  parentCheckpointId: string;
  coveredThroughSeq: number;
  sourceHeadSeq: number;
  sourceRevision: number;
  configHash: string;
  activeRequestMessageId: string;
  resumeAfterSeq: number;
  summary: CanonicalBlock[];
  formatVersion: number;
  promptVersion: string;
  summarizerProvider: string;
  summarizerModel: string;
  estimatedTokensBefore: number;
  estimatedTokensAfter: number;
  createdAt?: string;
}

export interface DurableLease {
  owner: string;
  generation: number;
  expiresAt?: string;
}

export interface ExecutionContext {
  session: DurableSessionRecord;
  config: DurableConfigRecord;
  checkpoint?: DurableCheckpointRecord;
}

export interface MutationGuard {
  mutationId: string;
  mutationHash: string;
  expectedRevision?: number;
  leaseGeneration: number;
}

export interface MessageDraftInput {
  id: string;
  role: "assistant" | "tool";
  content: CanonicalBlock[];
  status: CanonicalMessageStatus;
  replyToMessageId?: string;
  usage?: CanonicalUsage;
  providerMetadata?: Record<string, unknown>;
}

export interface DurableSessionClient {
  createSession(req: {
    userId: string;
    agentId?: string;
    config: Partial<DurableConfigRecord>;
  }): Promise<{ session: DurableSessionRecord; config: DurableConfigRecord }>;
  getExecutionContext(sessionId: string): Promise<ExecutionContext>;
  getMessages(
    sessionId: string,
    opts?: { userId?: string; beforeSeq?: number; limit?: number },
  ): Promise<{ messages: CanonicalMessage[]; hasMore: boolean; revision: number }>;
  beginRequest(req: {
    sessionId: string;
    userId: string;
    clientRequestId: string;
    requestHash: string;
    content: CanonicalBlock[];
    formatVersion?: number;
  }): Promise<{ session: DurableSessionRecord; requestMessage: CanonicalMessage; deduplicated: boolean }>;
  acquireLease(req: {
    sessionId: string;
    requestMessageId: string;
    owner: string;
    ttlSeconds: number;
  }): Promise<{ session: DurableSessionRecord; lease: DurableLease }>;
  renewLease(req: {
    sessionId: string;
    owner: string;
    generation: number;
    ttlSeconds: number;
  }): Promise<DurableLease>;
  releaseLease(req: { sessionId: string; owner: string; generation: number }): Promise<boolean>;
  appendMessages(req: {
    sessionId: string;
    requestMessageId: string;
    guard: MutationGuard;
    messages: MessageDraftInput[];
  }): Promise<{ session: DurableSessionRecord; messages: CanonicalMessage[]; deduplicated: boolean }>;
  finishRequest(req: {
    sessionId: string;
    requestMessageId: string;
    guard: MutationGuard;
    status: "completed" | "failed" | "cancelled" | "interrupted";
    messages?: MessageDraftInput[];
    errorCode?: string;
  }): Promise<{
    session: DurableSessionRecord;
    requestMessage: CanonicalMessage;
    messages: CanonicalMessage[];
    deduplicated: boolean;
  }>;
  cancelRequest(req: {
    sessionId: string;
    userId?: string;
    requestMessageId?: string;
  }): Promise<{ requestMessage: CanonicalMessage }>;
  publishCheckpoint(req: {
    sessionId: string;
    guard: MutationGuard;
    checkpoint: Omit<DurableCheckpointRecord, "sessionId" | "createdAt">;
  }): Promise<{ session: DurableSessionRecord; checkpoint: DurableCheckpointRecord; deduplicated: boolean }>;
  close(): void;
}

/** How a session-manager rejection maps to a retry decision. */
export type DurableErrorKind = "aborted" | "conflict" | "invalid" | "not_found" | "denied" | "unauthenticated" | "internal";

export class DurableRpcError extends Error {
  readonly code: grpc.status;
  readonly kind: DurableErrorKind;

  constructor(code: grpc.status, kind: DurableErrorKind, message: string) {
    super(message);
    this.name = "DurableRpcError";
    this.code = code;
    this.kind = kind;
  }

  /**
   * A lost fencing race or a transient abort: retrying against fresh state is
   * the correct response. State conflicts (busy, terminal, cancelled) are not
   * retryable — the caller must change what it is doing, not repeat it.
   */
  get retryable(): boolean {
    return this.kind === "aborted";
  }
}

export function classifyRpcError(err: grpc.ServiceError): DurableRpcError {
  switch (err.code) {
    case grpc.status.ABORTED:
      return new DurableRpcError(err.code, "aborted", err.details || err.message);
    case grpc.status.FAILED_PRECONDITION:
      return new DurableRpcError(err.code, "conflict", err.details || err.message);
    case grpc.status.INVALID_ARGUMENT:
      return new DurableRpcError(err.code, "invalid", err.details || err.message);
    case grpc.status.NOT_FOUND:
      return new DurableRpcError(err.code, "not_found", err.details || err.message);
    case grpc.status.PERMISSION_DENIED:
      return new DurableRpcError(err.code, "denied", err.details || err.message);
    case grpc.status.UNAUTHENTICATED:
      return new DurableRpcError(err.code, "unauthenticated", err.details || err.message);
    default:
      return new DurableRpcError(err.code, "internal", err.details || err.message);
  }
}

const CALL_TIMEOUT_MS = 10_000;

type UnaryMethod = (
  req: unknown,
  metadata: grpc.Metadata,
  options: grpc.CallOptions,
  callback: (err: grpc.ServiceError | null, res: unknown) => void,
) => void;

interface RawClient {
  [method: string]: UnaryMethod;
  close(): void;
}

function resolveProtoPath(): string {
  for (const rel of ["../../proto/session/v2/session.proto", "../proto/session/v2/session.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("proto/session/v2/session.proto not found; run `npm run sync-proto`");
}

function loadServiceConstructor(): grpc.ServiceClientConstructor {
  const packageDefinition = protoLoader.loadSync(resolveProtoPath(), {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as {
    session: { v2: { SessionService: grpc.ServiceClientConstructor } };
  };
  return proto.session.v2.SessionService;
}

/** int64 fields arrive as strings with `longs: String`. */
function num(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.length > 0) return Number(value);
  return 0;
}

/** Tolerate Date, {seconds,nanos}, RFC 3339, or epoch numbers. */
function timestampToString(value: unknown): string | undefined {
  if (!value) return undefined;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? undefined : value.toISOString();
  if (typeof value === "string") return value;
  if (typeof value === "number") return new Date(value).toISOString();
  if (typeof value === "object") {
    const record = value as { seconds?: unknown; nanos?: unknown };
    if (record.seconds !== undefined) return new Date(num(record.seconds) * 1000).toISOString();
  }
  return undefined;
}

const TOOL_RESULT_STATUS_TO_PROTO: Record<ToolResultStatus, string> = {
  success: "TOOL_RESULT_STATUS_SUCCESS",
  error: "TOOL_RESULT_STATUS_ERROR",
  cancelled: "TOOL_RESULT_STATUS_CANCELLED",
};

const TOOL_RESULT_STATUS_FROM_PROTO: Record<string, ToolResultStatus> = {
  TOOL_RESULT_STATUS_SUCCESS: "success",
  TOOL_RESULT_STATUS_ERROR: "error",
  TOOL_RESULT_STATUS_CANCELLED: "cancelled",
};

const MESSAGE_STATUS_TO_PROTO: Record<CanonicalMessageStatus, string> = {
  complete: "MESSAGE_STATUS_COMPLETE",
  partial: "MESSAGE_STATUS_PARTIAL",
  interrupted: "MESSAGE_STATUS_INTERRUPTED",
};

const MESSAGE_STATUS_FROM_PROTO: Record<string, CanonicalMessageStatus> = {
  MESSAGE_STATUS_COMPLETE: "complete",
  MESSAGE_STATUS_PARTIAL: "partial",
  MESSAGE_STATUS_INTERRUPTED: "interrupted",
};

const ROLE_TO_PROTO: Record<string, string> = {
  user: "MESSAGE_ROLE_USER",
  assistant: "MESSAGE_ROLE_ASSISTANT",
  tool: "MESSAGE_ROLE_TOOL",
};

const ROLE_FROM_PROTO: Record<string, CanonicalMessage["role"]> = {
  MESSAGE_ROLE_USER: "user",
  MESSAGE_ROLE_ASSISTANT: "assistant",
  MESSAGE_ROLE_TOOL: "tool",
};

const EXECUTION_STATUS_FROM_PROTO: Record<string, RequestExecution["status"]> = {
  EXECUTION_STATUS_QUEUED: "queued",
  EXECUTION_STATUS_RUNNING: "running",
  EXECUTION_STATUS_COMPLETED: "completed",
  EXECUTION_STATUS_FAILED: "failed",
  EXECUTION_STATUS_CANCELLED: "cancelled",
  EXECUTION_STATUS_INTERRUPTED: "interrupted",
};

/** Canonical block -> session.v2.ContentBlock (oneof member + snake_case fields). */
function blockToProto(block: CanonicalBlock): Record<string, unknown> {
  switch (block.type) {
    case "text":
      return { text: { text: block.text } };
    case "tool_call":
      return { tool_call: { id: block.id, name: block.name, arguments_json: block.arguments_json } };
    case "tool_result":
      return {
        tool_result: {
          tool_call_id: block.tool_call_id,
          status: TOOL_RESULT_STATUS_TO_PROTO[block.status] ?? "TOOL_RESULT_STATUS_SUCCESS",
          content: block.content.map(blockToProto),
          error_code: block.error_code ?? "",
        },
      };
    case "opaque":
      return {
        opaque: { provider: block.provider, type: block.opaque_type, data_json: Buffer.from(block.data_json) },
      };
  }
}

function blocksToProto(blocks: CanonicalBlock[]): Array<Record<string, unknown>> {
  return blocks.map(blockToProto);
}

/** session.v2.ContentBlock -> canonical block; null for an empty/unknown block. */
function blockFromProto(raw: unknown): CanonicalBlock | null {
  const block = (raw ?? {}) as Record<string, any>;
  const variant: string | undefined =
    typeof block.value === "string"
      ? block.value
      : block.text !== undefined && block.text !== null
        ? "text"
        : block.tool_call
          ? "tool_call"
          : block.tool_result
            ? "tool_result"
            : block.opaque
              ? "opaque"
              : undefined;
  switch (variant) {
    case "text":
      return { type: "text", text: String(block.text?.text ?? "") };
    case "tool_call":
      return {
        type: "tool_call",
        id: String(block.tool_call?.id ?? ""),
        name: String(block.tool_call?.name ?? ""),
        arguments_json: String(block.tool_call?.arguments_json ?? "{}"),
      };
    case "tool_result": {
      const nested = Array.isArray(block.tool_result?.content) ? block.tool_result.content : [];
      const status = TOOL_RESULT_STATUS_FROM_PROTO[String(block.tool_result?.status ?? "")] ?? "error";
      const out: CanonicalBlock = {
        type: "tool_result",
        tool_call_id: String(block.tool_result?.tool_call_id ?? ""),
        status,
        content: nested.map(blockFromProto).filter((b: CanonicalBlock | null): b is CanonicalBlock => b !== null),
      };
      const errorCode = block.tool_result?.error_code;
      if (typeof errorCode === "string" && errorCode) out.error_code = errorCode;
      return out;
    }
    case "opaque":
      return {
        type: "opaque",
        provider: String(block.opaque?.provider ?? ""),
        opaque_type: String(block.opaque?.type ?? ""),
        data_json: toBytes(block.opaque?.data_json),
      };
    default:
      return null;
  }
}

/** Decode a runtime.v2 content array into canonical blocks (handler boundary). */
export function canonicalBlocksFromProto(raw: unknown): CanonicalBlock[] {
  return blocksFromProto(raw);
}

function blocksFromProto(raw: unknown): CanonicalBlock[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map(blockFromProto)
    .filter((block: CanonicalBlock | null): block is CanonicalBlock => block !== null);
}

function toBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (Buffer.isBuffer(value)) return new Uint8Array(value);
  if (typeof value === "string") return new Uint8Array(Buffer.from(value, "base64"));
  return new Uint8Array(0);
}

function usageFromProto(raw: unknown): CanonicalUsage | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const usage = raw as Record<string, unknown>;
  const parsed: CanonicalUsage = {
    input_tokens: num(usage.input_tokens),
    cached_input_tokens: num(usage.cached_input_tokens),
    output_tokens: num(usage.output_tokens),
    reasoning_tokens: num(usage.reasoning_tokens),
    total_tokens: num(usage.total_tokens),
  };
  if (
    parsed.input_tokens === 0 &&
    parsed.cached_input_tokens === 0 &&
    parsed.output_tokens === 0 &&
    parsed.reasoning_tokens === 0 &&
    parsed.total_tokens === 0
  ) {
    return undefined;
  }
  return parsed;
}

function usageToProto(usage: CanonicalUsage): Record<string, number> {
  return {
    input_tokens: usage.input_tokens,
    cached_input_tokens: usage.cached_input_tokens,
    output_tokens: usage.output_tokens,
    reasoning_tokens: usage.reasoning_tokens,
    total_tokens: usage.total_tokens,
  };
}

function messageFromProto(raw: unknown): CanonicalMessage {
  const message = (raw ?? {}) as Record<string, any>;
  const request = message.request as Record<string, any> | undefined;
  const out: CanonicalMessage = {
    id: String(message.id ?? ""),
    session_id: String(message.session_id ?? ""),
    seq: num(message.seq),
    request_message_id: String(message.request_message_id ?? ""),
    role: ROLE_FROM_PROTO[String(message.role ?? "")] ?? "assistant",
    content: blocksFromProto(message.content),
    format_version: num(message.format_version),
    status: MESSAGE_STATUS_FROM_PROTO[String(message.status ?? "")] ?? "complete",
  };
  if (message.reply_to_message_id) out.reply_to_message_id = String(message.reply_to_message_id);
  const usage = usageFromProto(message.usage);
  if (usage) out.usage = usage;
  if (message.provider_metadata && typeof message.provider_metadata === "object") {
    out.provider_metadata = message.provider_metadata as Record<string, unknown>;
  }
  if (request) {
    out.request = {
      client_request_id: String(request.client_request_id ?? ""),
      request_hash: String(request.request_hash ?? ""),
      status: EXECUTION_STATUS_FROM_PROTO[String(request.status ?? "")] ?? "queued",
      cancellation_requested: request.cancellation_requested === true,
    };
    if (request.error_code) out.request.error_code = String(request.error_code);
  }
  const createdAt = timestampToString(message.created_at);
  if (createdAt) out.created_at = createdAt;
  return out;
}

function sessionFromProto(raw: unknown): DurableSessionRecord {
  const session = (raw ?? {}) as Record<string, any>;
  const out: DurableSessionRecord = {
    id: String(session.id ?? ""),
    userId: String(session.user_id ?? ""),
    status: String(session.status ?? "").includes("ENDED") ? "ended" : session.status ? "active" : "",
    agentId: String(session.agent_id ?? ""),
    title: String(session.title ?? ""),
    lastSeq: num(session.last_seq),
    revision: num(session.revision),
    activeCheckpointId: String(session.active_checkpoint_id ?? ""),
    activeRequestMessageId: String(session.active_request_message_id ?? ""),
  };
  const created = timestampToString(session.created_at);
  if (created) out.createdAt = created;
  const lastActive = timestampToString(session.last_active);
  if (lastActive) out.lastActive = lastActive;
  const endedAt = timestampToString(session.ended_at);
  if (endedAt) out.endedAt = endedAt;
  return out;
}

function configFromProto(raw: unknown): DurableConfigRecord {
  const config = (raw ?? {}) as Record<string, any>;
  const out: DurableConfigRecord = {
    sessionId: String(config.session_id ?? ""),
    formatVersion: num(config.format_version),
    configHash: String(config.config_hash ?? ""),
    systemPrompt: String(config.system_prompt ?? ""),
    provider: String(config.provider ?? ""),
    model: String(config.model ?? ""),
    baseUrl: String(config.base_url ?? ""),
    credentialRef: String(config.credential_ref ?? ""),
    contextWindowTokens: num(config.context_window_tokens),
    outputLimitTokens: num(config.output_limit_tokens),
    tools: Array.isArray(config.tools) ? config.tools : [],
  };
  const created = timestampToString(config.created_at);
  if (created) out.createdAt = created;
  return out;
}

function checkpointFromProto(raw: unknown): DurableCheckpointRecord {
  const cp = (raw ?? {}) as Record<string, any>;
  const out: DurableCheckpointRecord = {
    id: String(cp.id ?? ""),
    sessionId: String(cp.session_id ?? ""),
    parentCheckpointId: String(cp.parent_checkpoint_id ?? ""),
    coveredThroughSeq: num(cp.covered_through_seq),
    sourceHeadSeq: num(cp.source_head_seq),
    sourceRevision: num(cp.source_revision),
    configHash: String(cp.config_hash ?? ""),
    activeRequestMessageId: String(cp.active_request_message_id ?? ""),
    resumeAfterSeq: num(cp.resume_after_seq),
    summary: blocksFromProto(cp.summary),
    formatVersion: num(cp.format_version),
    promptVersion: String(cp.prompt_version ?? ""),
    summarizerProvider: String(cp.summarizer_provider ?? ""),
    summarizerModel: String(cp.summarizer_model ?? ""),
    estimatedTokensBefore: num(cp.estimated_tokens_before),
    estimatedTokensAfter: num(cp.estimated_tokens_after),
  };
  const created = timestampToString(cp.created_at);
  if (created) out.createdAt = created;
  return out;
}

function draftToProto(draft: MessageDraftInput): Record<string, unknown> {
  const out: Record<string, unknown> = {
    id: draft.id,
    role: ROLE_TO_PROTO[draft.role],
    content: blocksToProto(draft.content),
    format_version: FORMAT_VERSION,
    status: MESSAGE_STATUS_TO_PROTO[draft.status],
  };
  if (draft.replyToMessageId) out.reply_to_message_id = draft.replyToMessageId;
  if (draft.usage) out.usage = usageToProto(draft.usage);
  if (draft.providerMetadata) out.provider_metadata = draft.providerMetadata;
  return out;
}

function guardToProto(guard: MutationGuard): Record<string, unknown> {
  return {
    mutation_id: guard.mutationId,
    mutation_hash: guard.mutationHash,
    expected_revision: guard.expectedRevision ?? 0,
    lease_generation: guard.leaseGeneration,
  };
}

function configToProto(config: Partial<DurableConfigRecord>): Record<string, unknown> {
  return {
    format_version: config.formatVersion ?? FORMAT_VERSION,
    config_hash: config.configHash ?? "",
    system_prompt: config.systemPrompt ?? "",
    provider: config.provider ?? "",
    model: config.model ?? "",
    base_url: config.baseUrl ?? "",
    credential_ref: config.credentialRef ?? "",
    context_window_tokens: config.contextWindowTokens ?? 0,
    output_limit_tokens: config.outputLimitTokens ?? 0,
    tools: (config.tools ?? []).map((tool) => {
      const t = (tool ?? {}) as Record<string, unknown>;
      return {
        name: String(t.name ?? ""),
        description: String(t.description ?? ""),
        input_schema_json: typeof t.input_schema_json === "string" ? t.input_schema_json : JSON.stringify(t.parameters ?? {}),
        version: String(t.version ?? ""),
      };
    }),
  };
}

function checkpointToProto(
  checkpoint: Omit<DurableCheckpointRecord, "sessionId" | "createdAt">,
): Record<string, unknown> {
  return {
    id: checkpoint.id,
    parent_checkpoint_id: checkpoint.parentCheckpointId,
    covered_through_seq: checkpoint.coveredThroughSeq,
    source_head_seq: checkpoint.sourceHeadSeq,
    source_revision: checkpoint.sourceRevision,
    config_hash: checkpoint.configHash,
    active_request_message_id: checkpoint.activeRequestMessageId,
    resume_after_seq: checkpoint.resumeAfterSeq,
    summary: blocksToProto(checkpoint.summary),
    format_version: checkpoint.formatVersion || FORMAT_VERSION,
    prompt_version: checkpoint.promptVersion,
    summarizer_provider: checkpoint.summarizerProvider,
    summarizer_model: checkpoint.summarizerModel,
    estimated_tokens_before: checkpoint.estimatedTokensBefore,
    estimated_tokens_after: checkpoint.estimatedTokensAfter,
  };
}

/** Returns undefined when SESSION_MANAGER_ADDR is not configured. */
export function createDurableSessionClient(config: ServiceConfig): DurableSessionClient | undefined {
  if (!config.sessionManagerAddr) return undefined;
  const addr = config.sessionManagerAddr;
  let client: RawClient | undefined;

  // Lazy so the runtime starts (and tests run) without session-manager reachable.
  function getClient(): RawClient {
    if (!client) {
      const Service = loadServiceConstructor();
      client = new Service(addr, grpc.credentials.createInsecure()) as unknown as RawClient;
    }
    return client;
  }

  function call<T>(method: string, request: unknown, timeoutMs = CALL_TIMEOUT_MS): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let raw: RawClient;
      try {
        raw = getClient();
      } catch (err) {
        reject(err);
        return;
      }
      const metadata = new grpc.Metadata();
      if (config.serviceToken) metadata.set("x-service-token", config.serviceToken);
      const options: grpc.CallOptions = { deadline: Date.now() + timeoutMs };
      try {
        raw[method](request, metadata, options, (err, res) => {
          if (err) reject(classifyRpcError(err));
          else resolve(res as T);
        });
      } catch (err) {
        reject(err);
      }
    });
  }

  return {
    async createSession(req) {
      const res = await call<Record<string, unknown>>("createSession", {
        user_id: req.userId,
        agent_id: req.agentId ?? "",
        config: configToProto(req.config),
      });
      return { session: sessionFromProto(res.session), config: configFromProto(res.config) };
    },

    async getExecutionContext(sessionId) {
      const res = await call<Record<string, unknown>>("getExecutionContext", { session_id: sessionId });
      const out: ExecutionContext = {
        session: sessionFromProto(res.session),
        config: configFromProto(res.config),
      };
      if (res.checkpoint) out.checkpoint = checkpointFromProto(res.checkpoint);
      return out;
    },

    async getMessages(sessionId, opts) {
      const res = await call<Record<string, unknown>>("getMessages", {
        session_id: sessionId,
        user_id: opts?.userId ?? "",
        before_seq: opts?.beforeSeq ?? 0,
        limit: opts?.limit ?? 0,
      });
      const messages = Array.isArray(res.messages) ? res.messages.map(messageFromProto) : [];
      return { messages, hasMore: res.has_more === true, revision: num(res.revision) };
    },

    async beginRequest(req) {
      const res = await call<Record<string, unknown>>("beginRequest", {
        session_id: req.sessionId,
        user_id: req.userId,
        client_request_id: req.clientRequestId,
        request_hash: req.requestHash,
        content: blocksToProto(req.content),
        format_version: req.formatVersion ?? FORMAT_VERSION,
      });
      return {
        session: sessionFromProto(res.session),
        requestMessage: messageFromProto(res.request_message),
        deduplicated: res.deduplicated === true,
      };
    },

    async acquireLease(req) {
      const res = await call<Record<string, unknown>>("acquireLease", {
        session_id: req.sessionId,
        request_message_id: req.requestMessageId,
        owner: req.owner,
        ttl_seconds: req.ttlSeconds,
      });
      const lease = (res.lease ?? {}) as Record<string, unknown>;
      return {
        session: sessionFromProto(res.session),
        lease: {
          owner: String(lease.owner ?? ""),
          generation: num(lease.generation),
          expiresAt: timestampToString(lease.expires_at),
        },
      };
    },

    async renewLease(req) {
      const res = await call<Record<string, unknown>>("renewLease", {
        session_id: req.sessionId,
        owner: req.owner,
        generation: req.generation,
        ttl_seconds: req.ttlSeconds,
      });
      const lease = (res.lease ?? {}) as Record<string, unknown>;
      return {
        owner: String(lease.owner ?? ""),
        generation: num(lease.generation),
        expiresAt: timestampToString(lease.expires_at),
      };
    },

    async releaseLease(req) {
      await call<Record<string, unknown>>("releaseLease", {
        session_id: req.sessionId,
        owner: req.owner,
        generation: req.generation,
      });
      return true;
    },

    async appendMessages(req) {
      const res = await call<Record<string, unknown>>("appendMessages", {
        session_id: req.sessionId,
        request_message_id: req.requestMessageId,
        guard: guardToProto(req.guard),
        messages: req.messages.map(draftToProto),
      });
      const messages = Array.isArray(res.messages) ? res.messages.map(messageFromProto) : [];
      return { session: sessionFromProto(res.session), messages, deduplicated: res.deduplicated === true };
    },

    async finishRequest(req) {
      const statusToProto: Record<string, string> = {
        completed: "EXECUTION_STATUS_COMPLETED",
        failed: "EXECUTION_STATUS_FAILED",
        cancelled: "EXECUTION_STATUS_CANCELLED",
        interrupted: "EXECUTION_STATUS_INTERRUPTED",
      };
      const res = await call<Record<string, unknown>>("finishRequest", {
        session_id: req.sessionId,
        request_message_id: req.requestMessageId,
        guard: guardToProto(req.guard),
        status: statusToProto[req.status],
        messages: (req.messages ?? []).map(draftToProto),
        error_code: req.errorCode ?? "",
      }, 30_000);
      const messages = Array.isArray(res.messages) ? res.messages.map(messageFromProto) : [];
      return {
        session: sessionFromProto(res.session),
        requestMessage: messageFromProto(res.request_message),
        messages,
        deduplicated: res.deduplicated === true,
      };
    },

    async cancelRequest(req) {
      const res = await call<Record<string, unknown>>("cancelRequest", {
        session_id: req.sessionId,
        user_id: req.userId ?? "",
        request_message_id: req.requestMessageId ?? "",
      });
      return { requestMessage: messageFromProto(res.request_message) };
    },

    async publishCheckpoint(req) {
      const res = await call<Record<string, unknown>>("publishCheckpoint", {
        session_id: req.sessionId,
        guard: guardToProto(req.guard),
        checkpoint: checkpointToProto(req.checkpoint),
      }, 30_000);
      return {
        session: sessionFromProto(res.session),
        checkpoint: checkpointFromProto(res.checkpoint),
        deduplicated: res.deduplicated === true,
      };
    },

    close() {
      client?.close();
      client = undefined;
    },
  };
}
