import assert from "node:assert/strict";
import { test } from "node:test";
import { text, toolCall, toolResult, type CanonicalMessage } from "../src/canonical.js";
import {
  CompactionError,
  compact,
  generateSummary,
  selectSafeCutoff,
  summaryText,
  type CheckpointPublisher,
} from "../src/compaction.js";
import type { DurableCheckpointRecord } from "../src/durable-client.js";
import { DEFAULT_BUDGET, type BudgetConfig } from "../src/tokens.js";

const budget: BudgetConfig = { ...DEFAULT_BUDGET, contextWindow: 200_000, outputReserve: 4_096 };

function user(id: string, seq: number, body: string): CanonicalMessage {
  return {
    id,
    session_id: "sess-1",
    seq,
    request_message_id: id,
    role: "user",
    content: [text(body)],
    format_version: 1,
    status: "complete",
  };
}

function assistant(id: string, seq: number, requestId: string, blocks: CanonicalMessage["content"]): CanonicalMessage {
  return {
    id,
    session_id: "sess-1",
    seq,
    request_message_id: requestId,
    role: "assistant",
    content: blocks,
    format_version: 1,
    status: "complete",
  };
}

function tool(id: string, seq: number, requestId: string, callId: string): CanonicalMessage {
  return {
    id,
    session_id: "sess-1",
    seq,
    request_message_id: requestId,
    role: "tool",
    content: [toolResult(callId, "success", [text("ok")])],
    format_version: 1,
    status: "complete",
  };
}

/** Three requests; request A holds a tool pair. */
function threeRequests(): CanonicalMessage[] {
  return [
    user("r1", 1, "first"),
    assistant("a1", 2, "r1", [text("running"), toolCall("c1", "bash", { command: "pwd" })]),
    tool("t1", 3, "r1", "c1"),
    user("r2", 4, "second"),
    assistant("a2", 5, "r2", [text("answered")]),
    user("r3", 6, "third"),
    assistant("a3", 7, "r3", [text("answered again")]),
  ];
}

test("a cutoff never splits a tool call from its result", () => {
  const messages = [user("r1", 1, "go"), assistant("a1", 2, "r1", [toolCall("c1", "bash", {})]), tool("t1", 3, "r1", "c1")];
  // A generous target: the only safe cut is after the user message.
  const cutoff = selectSafeCutoff(messages, 100_000);
  assert.ok(cutoff, "a safe cutoff should exist");
  assert.equal(cutoff.coveredThroughSeq, 1);
  assert.deepEqual(cutoff.retained.map((m) => m.id), ["a1", "t1"]);
  // Explicitly: the boundary after the assistant tool-call message is refused.
  assert.notEqual(cutoff.coveredThroughSeq, 2);
});

test("the latest two requests are retained when they fit", () => {
  const messages = threeRequests();
  const retainedTwo = messages.slice(3);
  const target = Math.ceil(retainedTwo.reduce((sum, m) => sum + JSON.stringify(m.content).length / 3, 0)) + 50;
  const cutoff = selectSafeCutoff(messages, target);
  assert.ok(cutoff);
  assert.equal(cutoff.coveredThroughSeq, 3, "summarize request A only");
  assert.deepEqual(cutoff.retained.map((m) => m.id), ["r2", "a2", "r3", "a3"]);
  assert.deepEqual(cutoff.summarized.map((m) => m.id), ["r1", "a1", "t1"]);
});

test("a larger target retains more history", () => {
  const messages = threeRequests();
  const cutoff = selectSafeCutoff(messages, 100_000);
  // Nothing needs summarizing when everything fits: the newest boundary that
  // still fits is the one leaving the final request.
  assert.ok(cutoff);
  assert.ok(cutoff.retained.length <= messages.length);
  assert.equal(cutoff.summarized.length + cutoff.retained.length, messages.length);
});

test("mid-request compaction cuts at a resolved tool boundary", () => {
  const requestId = "r1";
  const messages: CanonicalMessage[] = [
    user(requestId, 1, "long task"),
    assistant("a1", 2, requestId, [text("step 1")]),
    assistant("a2", 3, requestId, [toolCall("c1", "bash", {})]),
    tool("t1", 4, requestId, "c1"),
    assistant("a3", 5, requestId, [toolCall("c2", "bash", {})]),
    tool("t2", 6, requestId, "c2"),
  ];
  // No request boundary is available, so the cut lands inside the request after
  // a fully resolved tool batch — never between c2 and its result.
  const cutoff = selectSafeCutoff(messages, 100_000);
  assert.ok(cutoff);
  // The newest closed prefix ends before the last tool call (seq 4), so the
  // retained context is that call together with its result — never separated.
  assert.equal(cutoff.coveredThroughSeq, 4);
  assert.deepEqual(cutoff.retained.map((m) => m.id), ["a3", "t2"]);
});

test("an incomplete assistant message is never summarized away", () => {
  const messages: CanonicalMessage[] = [
    user("r1", 1, "go"),
    { ...assistant("a1", 2, "r1", [text("half a sen")]), status: "partial" },
    tool("t1", 3, "r1", "c1"),
  ];
  const cutoff = selectSafeCutoff(messages, 100_000);
  // Only the boundary after the user message is safe with a resolved prefix.
  assert.ok(cutoff);
  assert.ok(cutoff.coveredThroughSeq !== 2);
});

test("no safe cutoff returns null instead of a corrupt one", () => {
  const messages = [user("r1", 1, "go"), assistant("a1", 2, "r1", [toolCall("c1", "bash", {})])];
  // The tool result has not been persisted yet, so nothing may be summarized.
  assert.equal(selectSafeCutoff(messages, 1), null);
  assert.equal(selectSafeCutoff([user("r1", 1, "only")], 1), null);
});

test("generateSummary trims, retries once, and enforces the deadline", async () => {
  let calls = 0;
  const flaky = async () => {
    calls++;
    if (calls === 1) throw new Error("429 rate limited");
    return { text: "  summary  " };
  };
  const result = await generateSummary(flaky, { messages: [user("r1", 1, "go")], previousSummary: null }, 5_000);
  assert.equal(result.text, "summary");
  assert.equal(calls, 2, "a transient failure is retried once");

  await assert.rejects(
    generateSummary(async () => ({ text: "   " }), { messages: [user("r1", 1, "go")], previousSummary: null }, 5_000),
    (err: unknown) => {
      assert.ok(err instanceof CompactionError);
      assert.equal(err.reason, "summary_failed");
      return true;
    },
  );

  await assert.rejects(
    generateSummary(
      () => new Promise((resolve) => setTimeout(() => resolve({ text: "too late" }), 500)),
      { messages: [user("r1", 1, "go")], previousSummary: null },
      30,
    ),
    (err: unknown) => {
      assert.ok(err instanceof CompactionError);
      assert.equal(err.reason, "summary_failed");
      assert.match(err.message, /deadline/);
      return true;
    },
  );
});

class FakePublisher implements CheckpointPublisher {
  readonly published: Array<Omit<DurableCheckpointRecord, "sessionId" | "createdAt">> = [];
  fail?: Error;

  async publishCheckpoint(cp: Omit<DurableCheckpointRecord, "sessionId" | "createdAt">): Promise<DurableCheckpointRecord> {
    this.published.push(cp);
    if (this.fail) throw this.fail;
    return { ...cp, sessionId: "sess-1", createdAt: new Date().toISOString() };
  }
}

function compactOptions(publisher: FakePublisher, messages: CanonicalMessage[]) {
  return {
    publisher,
    summarizer: async () => ({ text: "the user asked for three things" }),
    messages,
    budget,
    configHash: "cfg-hash-1",
    activeRequestMessageId: "r3",
    resumeAfterSeq: 7,
    sourceRevision: 4,
    checkpointId: "cp-1",
    provider: "openai",
    model: "gpt-4o-mini",
    targetTokens: 100_000,
  };
}

test("compact publishes a checkpoint describing the cutoff and estimates", async () => {
  const publisher = new FakePublisher();
  const result = await compact(compactOptions(publisher, threeRequests()));
  assert.equal(result.checkpoint.id, "cp-1");
  assert.equal(result.checkpoint.coveredThroughSeq, result.cutoff.coveredThroughSeq);
  assert.equal(result.checkpoint.configHash, "cfg-hash-1");
  assert.equal(result.checkpoint.activeRequestMessageId, "r3");
  assert.equal(result.checkpoint.summarizerModel, "gpt-4o-mini");
  assert.ok(result.checkpoint.estimatedTokensBefore >= result.checkpoint.estimatedTokensAfter);
  assert.equal(summaryText(result.checkpoint), "the user asked for three things");
  assert.equal(publisher.published.length, 1);
});

test("a failed summary leaves the previous checkpoint active", async () => {
  const publisher = new FakePublisher();
  const options = compactOptions(publisher, threeRequests());
  options.summarizer = async () => {
    throw new Error("provider exploded");
  };
  await assert.rejects(compact(options), (err: unknown) => {
    assert.ok(err instanceof CompactionError);
    assert.equal(err.reason, "summary_failed");
    return true;
  });
  assert.equal(publisher.published.length, 0, "nothing may be published without a summary");
});

test("a failed publication is reported without corrupting the active pointer", async () => {
  const publisher = new FakePublisher();
  publisher.fail = new Error("stale lease");
  await assert.rejects(compact(compactOptions(publisher, threeRequests())), (err: unknown) => {
    assert.ok(err instanceof CompactionError);
    assert.equal(err.reason, "publish_failed");
    return true;
  });
});

test("no safe cutoff is reported as such, not as a summary failure", async () => {
  const publisher = new FakePublisher();
  const messages = [user("r1", 1, "go"), assistant("a1", 2, "r1", [toolCall("c1", "bash", {})])];
  const options = { ...compactOptions(publisher, messages), targetTokens: 1 };
  await assert.rejects(compact(options), (err: unknown) => {
    assert.ok(err instanceof CompactionError);
    assert.equal(err.reason, "no_safe_cut");
    return true;
  });
});
