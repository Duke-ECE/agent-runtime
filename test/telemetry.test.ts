import assert from "node:assert/strict";
import { test } from "node:test";
import {
  elapsedMs,
  recordCompaction,
  recordDurableWrite,
  recordHydrationFailure,
  recordLeaseConflict,
  type TelemetryRecord,
} from "../src/telemetry.js";

function capture(): { records: TelemetryRecord[]; sink: (r: TelemetryRecord) => void } {
  const records: TelemetryRecord[] = [];
  return { records, sink: (record) => records.push(record) };
}

test("a compaction record derives token reduction from before/after", () => {
  const { records, sink } = capture();
  recordCompaction(sink, {
    sessionId: "sess-1",
    requestMessageId: "msg-r1",
    checkpointId: "cp-1",
    coveredThroughSeq: 12,
    sourceRevision: 7,
    estimatedTokensBefore: 9_000,
    estimatedTokensAfter: 1_200,
    durationMs: 840,
    summarizerModel: "gpt-4o-mini",
    promptVersion: "v1",
  });
  assert.equal(records.length, 1);
  const record = records[0];
  assert.equal(record.event, "compaction");
  assert.equal(record.token_reduction, 7_800);
  assert.equal(record.duration_ms, 840);
  assert.equal(record.covered_through_seq, 12);
  assert.equal(record.source_revision, 7);
});

test("every record is flat, single-event, and free of content or credentials", () => {
  const { records, sink } = capture();
  recordCompaction(sink, {
    sessionId: "s", requestMessageId: "r", checkpointId: "c",
    coveredThroughSeq: 1, sourceRevision: 1, estimatedTokensBefore: 10,
    estimatedTokensAfter: 5, durationMs: 1, summarizerModel: "m", promptVersion: "v1",
  });
  recordLeaseConflict(sink, { sessionId: "s", requestMessageId: "r", reason: "aborted" });
  recordDurableWrite(sink, {
    sessionId: "s", requestMessageId: "r", kind: "append",
    mutationId: "r:1", messageCount: 2, latencyMs: 12, ok: true,
  });
  recordHydrationFailure(sink, { sessionId: "s", reason: "timeout", messageCount: 0, pageCount: 3 });

  // Field names that would mean a body or a credential reached the record.
  // `prompt_version` is deliberately allowed: it is a version identifier, not
  // the prompt text.
  const neverAllowed = ["content", "prompt", "api_key", "apikey", "secret", "body"];
  const neverSubstring = /content|api_key|apikey|secret|password/i;
  for (const record of records) {
    assert.equal(typeof record.event, "string");
    assert.ok(!record.event.includes("\n"), "an event must be a single line");
    for (const [key, value] of Object.entries(record)) {
      assert.ok(!neverAllowed.includes(key), `record ${record.event} leaked field ${key}`);
      assert.ok(!neverSubstring.test(key), `record ${record.event} leaked field ${key}`);
      assert.ok(
        value === undefined || ["string", "number", "boolean"].includes(typeof value),
        `record ${record.event} field ${key} is not scalar — bodies could hide in it`,
      );
      if (typeof value === "string") {
        assert.ok(!value.includes("\n"), `record ${record.event} field ${key} is multi-line`);
      }
    }
  }
});

test("lease, write, and hydration records carry the operational fields", () => {
  const { records, sink } = capture();
  recordLeaseConflict(sink, { sessionId: "sess-1", requestMessageId: "msg-1", reason: "aborted" });
  recordDurableWrite(sink, {
    sessionId: "sess-1", requestMessageId: "msg-1", kind: "finish",
    mutationId: "msg-1:3", messageCount: 1, latencyMs: 38, ok: true,
  });
  recordHydrationFailure(sink, { sessionId: "sess-1", reason: "lease expired" });

  assert.deepEqual(records[0], {
    event: "lease_conflict", session_id: "sess-1", request_message_id: "msg-1", reason: "aborted",
  });
  assert.equal(records[1].latency_ms, 38);
  assert.equal(records[1].kind, "finish");
  assert.equal(records[2].event, "hydration_failure");
  assert.equal(records[2].message_count, undefined);
});

test("elapsed time is never negative", () => {
  assert.equal(elapsedMs(1_000, () => 1_250), 250);
  // A clock that goes backwards must not produce a negative duration.
  assert.equal(elapsedMs(1_000, () => 900), 0);
});
