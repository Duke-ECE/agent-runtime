import { createHash, randomUUID } from "node:crypto";
import type { CanonicalBlock, CanonicalMessage } from "./canonical.js";
import { elapsedMs, jsonLineSink, recordDurableWrite, type TelemetrySink } from "./telemetry.js";
import {
  DurableRpcError,
  type DurableCheckpointRecord,
  type DurableSessionClient,
  type DurableSessionRecord,
  type MessageDraftInput,
  type MutationGuard,
} from "./durable-client.js";

/**
 * The durable lifecycle of one request: admission, a renewable fenced lease,
 * acknowledged incremental message batches, and a terminal transition.
 *
 * Everything the agent loop persists goes through {@link persist} before the
 * next model or tool step, and every mutation carries the current lease
 * generation, so a runtime that lost ownership cannot keep writing. The class
 * holds no conversation state of its own — canonical history lives in
 * session-manager.
 */

/** Initial lease duration and renewal cadence from the design. */
export const LEASE_TTL_SECONDS = 90;
export const LEASE_RENEW_INTERVAL_MS = 20_000;

export interface DurableExecutionOptions {
  client: DurableSessionClient;
  sessionId: string;
  userId: string;
  clientRequestId: string;
  content: CanonicalBlock[];
  /** Fencing owner identity; unique per runtime process. */
  owner?: string;
  leaseTtlSeconds?: number;
  renewIntervalMs?: number;
  /** Invoked once when the lease is lost; the caller must abort inference. */
  onLeaseLost?: (err: Error) => void;
  /** Where durable-write latency records go; defaults to one JSON line each. */
  telemetry?: TelemetrySink;
}

/** Deterministic hash of a request's content, used for dedup conflict detection. */
export function requestHashOf(content: CanonicalBlock[]): string {
  return createHash("sha256").update(JSON.stringify(content)).digest("hex");
}

/** Deterministic hash of a message batch, used as its idempotency hash. */
export function batchHashOf(messages: MessageDraftInput[]): string {
  return createHash("sha256").update(JSON.stringify(messages)).digest("hex");
}

/** Deterministic hash of a checkpoint payload, for its idempotency guard. */
function hashOf(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

export function newOwnerId(): string {
  return `runtime-${randomUUID()}`;
}

export class DurableExecution {
  readonly sessionId: string;
  readonly requestMessageId: string;
  readonly owner: string;
  /** True when session-manager deduplicated the request (a reconnect). */
  readonly deduplicated: boolean;
  /**
   * The request root (the initiating user message). It is part of the
   * model-facing context, so a caller assembling that context must include it —
   * leaving it out under-counts the input by the whole user message.
   */
  readonly requestMessage: CanonicalMessage;

  private readonly client: DurableSessionClient;
  private readonly leaseTtlSeconds: number;
  private readonly renewIntervalMs: number;
  private readonly onLeaseLost?: (err: Error) => void;
  private readonly telemetry: TelemetrySink;
  private renewTimer?: NodeJS.Timeout;
  private mutationSeq = 0;
  private finished = false;
  private leaseFailure?: Error;

  private leaseGenerationValue: number;
  private revisionValue: number;
  private sessionValue: DurableSessionRecord;

  private constructor(
    client: DurableSessionClient,
    session: DurableSessionRecord,
    requestMessage: CanonicalMessage,
    requestMessageId: string,
    owner: string,
    deduplicated: boolean,
    leaseGeneration: number,
    options: Pick<DurableExecutionOptions, "leaseTtlSeconds" | "renewIntervalMs" | "onLeaseLost" | "telemetry">,
  ) {
    this.client = client;
    this.telemetry = options.telemetry ?? jsonLineSink;
    this.sessionValue = session;
    this.requestMessageId = requestMessageId;
    this.requestMessage = requestMessage;
    this.owner = owner;
    this.deduplicated = deduplicated;
    this.leaseGenerationValue = leaseGeneration;
    this.revisionValue = session.revision;
    this.sessionId = session.id;
    this.leaseTtlSeconds = options.leaseTtlSeconds ?? LEASE_TTL_SECONDS;
    this.renewIntervalMs = options.renewIntervalMs ?? LEASE_RENEW_INTERVAL_MS;
    this.onLeaseLost = options.onLeaseLost;
  }

  /**
   * Admit the request and take fenced ownership. A deduplicated admission means
   * the root already exists (a reconnect after a lost acknowledgement): the
   * caller must not rerun tools, it should report the existing state instead.
   */
  static async start(options: DurableExecutionOptions): Promise<DurableExecution> {
    const owner = options.owner ?? newOwnerId();
    const admitted = await options.client.beginRequest({
      sessionId: options.sessionId,
      userId: options.userId,
      clientRequestId: options.clientRequestId,
      requestHash: requestHashOf(options.content),
      content: options.content,
    });
    const requestMessageId = admitted.requestMessage.id;
    let leaseGeneration = 0;
    let session = admitted.session;
    if (!admitted.deduplicated) {
      const leased = await options.client.acquireLease({
        sessionId: options.sessionId,
        requestMessageId,
        owner,
        ttlSeconds: options.leaseTtlSeconds ?? LEASE_TTL_SECONDS,
      });
      leaseGeneration = leased.lease.generation;
      session = leased.session;
    }
    const execution = new DurableExecution(
      options.client,
      session,
      admitted.requestMessage,
      requestMessageId,
      owner,
      admitted.deduplicated,
      leaseGeneration,
      options,
    );
    if (!admitted.deduplicated) execution.startRenewal();
    return execution;
  }

  get session(): DurableSessionRecord {
    return this.sessionValue;
  }

  get revision(): number {
    return this.revisionValue;
  }

  get leaseGeneration(): number {
    return this.leaseGenerationValue;
  }

  /** True once ownership was lost; no further writes may be attempted. */
  get leaseLost(): boolean {
    return this.leaseFailure !== undefined;
  }

  /** Throw when no usable lease is held, so inference stops immediately. */
  assertLeaseHeld(): void {
    if (this.leaseFailure) {
      throw this.leaseFailure;
    }
    if (this.leaseGenerationValue <= 0) {
      // A deduplicated admission (a reconnect) never took ownership, so it has
      // no fence to write under.
      throw new DurableRpcError(10, "aborted", "no execution lease is held for this request");
    }
  }

  private startRenewal(): void {
    this.renewTimer = setInterval(() => {
      void this.renewOnce();
    }, this.renewIntervalMs);
    // Renewal is internal bookkeeping, not a reason to keep the process alive.
    this.renewTimer.unref?.();
  }

  private async renewOnce(): Promise<void> {
    if (this.finished || this.leaseFailure) return;
    try {
      const lease = await this.client.renewLease({
        sessionId: this.sessionId,
        owner: this.owner,
        generation: this.leaseGenerationValue,
        ttlSeconds: this.leaseTtlSeconds,
      });
      this.leaseGenerationValue = lease.generation;
    } catch (err) {
      this.failLease(err instanceof Error ? err : new Error(String(err)));
    }
  }

  private failLease(err: Error): void {
    if (this.leaseFailure) return;
    this.leaseFailure =
      err instanceof DurableRpcError
        ? err
        : new DurableRpcError(2, "aborted", `execution lease lost: ${err.message}`);
    this.stopRenewal();
    try {
      this.onLeaseLost?.(this.leaseFailure);
    } catch {
      // A failing listener must not hide the lease loss.
    }
  }

  private stopRenewal(): void {
    if (this.renewTimer) {
      clearInterval(this.renewTimer);
      this.renewTimer = undefined;
    }
  }

  private nextGuard(): MutationGuard {
    return {
      mutationId: `${this.requestMessageId}:${++this.mutationSeq}`,
      mutationHash: "",
      expectedRevision: this.revisionValue,
      leaseGeneration: this.leaseGenerationValue,
    };
  }

  /**
   * Commit one batch of completed assistant/tool messages. The caller awaits
   * this before dispatching a tool or issuing the next model call, so the
   * durable boundary is acknowledged rather than best-effort.
   */
  async persist(messages: MessageDraftInput[]): Promise<CanonicalMessage[]> {
    if (messages.length === 0) return [];
    this.assertLeaseHeld();
    const guard: MutationGuard = { ...this.nextGuard(), mutationHash: batchHashOf(messages) };
    const startedAt = Date.now();
    let ok = false;
    try {
      const result = await this.client.appendMessages({
        sessionId: this.sessionId,
        requestMessageId: this.requestMessageId,
        guard,
        messages,
      });
      ok = true;
      this.sessionValue = result.session;
      this.revisionValue = result.session.revision;
      return result.messages;
    } finally {
      recordDurableWrite(this.telemetry, {
        sessionId: this.sessionId,
        requestMessageId: this.requestMessageId,
        kind: "append",
        mutationId: guard.mutationId,
        messageCount: messages.length,
        latencyMs: elapsedMs(startedAt),
        ok,
      });
    }
  }

  /** Publish a compaction checkpoint under the current lease. */
  async publishCheckpoint(
    checkpoint: Omit<DurableCheckpointRecord, "sessionId" | "createdAt">,
  ): Promise<DurableCheckpointRecord> {
    this.assertLeaseHeld();
    // The guard needs a hash, exactly as append and finish do: session-manager
    // rejects a mutation without one, so publication would always fail.
    const result = await this.client.publishCheckpoint({
      sessionId: this.sessionId,
      guard: { ...this.nextGuard(), mutationHash: hashOf(checkpoint) },
      checkpoint,
    });
    this.sessionValue = result.session;
    this.revisionValue = result.session.revision;
    return result.checkpoint;
  }

  /**
   * Write the final batch and the terminal state in one transaction, releasing
   * the lease. Safe to call once; later calls are ignored.
   */
  async finish(
    status: "completed" | "failed" | "cancelled" | "interrupted",
    messages: MessageDraftInput[] = [],
    errorCode?: string,
  ): Promise<CanonicalMessage[]> {
    if (this.finished) return [];
    this.finished = true;
    this.stopRenewal();
    // A lost lease may be held by another runtime now; finishing would be a
    // stale write, so only the holder persists the terminal state.
    this.assertLeaseHeld();
    const guard: MutationGuard = {
      ...this.nextGuard(),
      mutationHash: batchHashOf(messages),
    };
    const startedAt = Date.now();
    let ok = false;
    try {
      const result = await this.client.finishRequest({
        sessionId: this.sessionId,
        requestMessageId: this.requestMessageId,
        guard,
        status,
        messages,
        errorCode,
      });
      ok = true;
      this.sessionValue = result.session;
      this.revisionValue = result.session.revision;
      return result.messages;
    } finally {
      recordDurableWrite(this.telemetry, {
        sessionId: this.sessionId,
        requestMessageId: this.requestMessageId,
        kind: "finish",
        mutationId: guard.mutationId,
        messageCount: messages.length,
        latencyMs: elapsedMs(startedAt),
        ok,
      });
    }
  }

  /**
   * Observe cooperative cancellation. The terminal transition still belongs to
   * the holding runtime; this only reports the flag session-manager recorded.
   */
  async cancellationRequested(): Promise<boolean> {
    const state = await this.client.getExecutionContext(this.sessionId);
    if (state.session.activeRequestMessageId !== this.requestMessageId) return false;
    const messages = await this.client.getMessages(this.sessionId, { limit: 1, beforeSeq: 0 });
    const root = messages.messages.find((m) => m.id === this.requestMessageId);
    return root?.request?.cancellation_requested === true;
  }

  /** Release the lease without a terminal transition (e.g. eviction). */
  async abandon(): Promise<void> {
    this.finished = true;
    this.stopRenewal();
    if (this.leaseGenerationValue > 0) {
      try {
        await this.client.releaseLease({
          sessionId: this.sessionId,
          owner: this.owner,
          generation: this.leaseGenerationValue,
        });
      } catch {
        // Releasing a lease that already lapsed is harmless.
      }
    }
  }

  /** Stop local timers; the caller is responsible for the terminal write. */
  dispose(): void {
    this.stopRenewal();
  }
}
