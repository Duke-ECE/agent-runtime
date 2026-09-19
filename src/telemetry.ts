/**
 * Structured telemetry for the durable runtime.
 *
 * The design asks for token reduction, compaction duration and usage, ids,
 * revisions, lease conflicts, write latency, and hydration failures — and for
 * bodies and credentials never to be logged. That is enforced structurally:
 * every record is built from an explicit field list, so there is no code path
 * that can attach message content, a prompt, or an API key to a log line.
 *
 * Records are one JSON object per line, so they can be shipped, grepped, or
 * ignored without a parser.
 */

/** Where records go. Injected so tests capture them instead of printing. */
export type TelemetrySink = (record: TelemetryRecord) => void;

export interface TelemetryRecord {
  event: string;
  [field: string]: string | number | boolean | undefined;
}

/** One JSON object per line; never a multi-line body. */
export const jsonLineSink: TelemetrySink = (record) => {
  console.log(JSON.stringify(record));
};

export interface CompactionTelemetry {
  sessionId: string;
  requestMessageId: string;
  checkpointId: string;
  /** Inclusive original-message cutoff the summary covers. */
  coveredThroughSeq: number;
  /** Session revision the checkpoint was built from. */
  sourceRevision: number;
  estimatedTokensBefore: number;
  estimatedTokensAfter: number;
  /** Wall-clock time the summarizer and publication took. */
  durationMs: number;
  summarizerModel: string;
  promptVersion: string;
}

/**
 * Record one successful compaction. Token reduction is derived, not passed in,
 * so before/after can never disagree with it.
 */
export function recordCompaction(sink: TelemetrySink, t: CompactionTelemetry): void {
  sink({
    event: "compaction",
    session_id: t.sessionId,
    request_message_id: t.requestMessageId,
    checkpoint_id: t.checkpointId,
    covered_through_seq: t.coveredThroughSeq,
    source_revision: t.sourceRevision,
    estimated_tokens_before: t.estimatedTokensBefore,
    estimated_tokens_after: t.estimatedTokensAfter,
    token_reduction: t.estimatedTokensBefore - t.estimatedTokensAfter,
    duration_ms: t.durationMs,
    summarizer_model: t.summarizerModel,
    prompt_version: t.promptVersion,
  });
}

export interface LeaseTelemetry {
  sessionId: string;
  requestMessageId?: string;
  /** Machine-readable reason, e.g. the classifier's kind. */
  reason: string;
}

/** Record a lost or contended execution lease. */
export function recordLeaseConflict(sink: TelemetrySink, t: LeaseTelemetry): void {
  sink({
    event: "lease_conflict",
    session_id: t.sessionId,
    request_message_id: t.requestMessageId,
    reason: t.reason,
  });
}

export interface WriteTelemetry {
  sessionId: string;
  requestMessageId: string;
  /** "append" or "finish". */
  kind: string;
  mutationId: string;
  messageCount: number;
  latencyMs: number;
  /** False when the write was refused or failed; latency still matters then. */
  ok: boolean;
}

/**
 * Record the latency of one acknowledged durable write. Mutation ids are
 * already unique per batch and carry no content, so they are safe to log.
 */
export function recordDurableWrite(sink: TelemetrySink, t: WriteTelemetry): void {
  sink({
    event: "durable_write",
    session_id: t.sessionId,
    request_message_id: t.requestMessageId,
    kind: t.kind,
    mutation_id: t.mutationId,
    message_count: t.messageCount,
    latency_ms: t.latencyMs,
    ok: t.ok,
  });
}

export interface UnresolvedToolTelemetry {
  sessionId: string;
  /** The assistant message that issued the calls. */
  assistantMessageId: string;
  toolCallCount: number;
}

/**
 * Record tool calls found without a persisted result. This is the crash case:
 * the runtime died between persisting the call and persisting the observed
 * result, so the outcome is unknown. They are excluded from model context and
 * the turn needs an explicit retry decision — never a replay.
 */
export function recordUnresolvedToolCalls(sink: TelemetrySink, t: UnresolvedToolTelemetry): void {
  sink({
    event: "unresolved_tool_calls",
    session_id: t.sessionId,
    assistant_message_id: t.assistantMessageId,
    tool_call_count: t.toolCallCount,
  });
}

export interface HydrationTelemetry {
  sessionId: string;
  /** Machine-readable reason; a provider or transport message is a detail, not
   * a body, and is safe to include. */
  reason: string;
  messageCount?: number;
  pageCount?: number;
}

/** Record a failed or truncated hydration. */
export function recordHydrationFailure(sink: TelemetrySink, t: HydrationTelemetry): void {
  sink({
    event: "hydration_failure",
    session_id: t.sessionId,
    reason: t.reason,
    message_count: t.messageCount,
    page_count: t.pageCount,
  });
}

/** Milliseconds elapsed since a start timestamp from {@link now}. */
export function elapsedMs(startedAt: number, now: () => number = Date.now): number {
  return Math.max(0, now() - startedAt);
}
