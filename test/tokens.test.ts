import assert from "node:assert/strict";
import { test } from "node:test";
import { text, type CanonicalMessage } from "../src/canonical.js";
import {
  ContextEstimator,
  DEFAULT_BUDGET,
  compactTarget,
  estimateRequestTokens,
  inputBudget,
  providerMessages,
  shouldCompact,
  type BudgetConfig,
  type RequestShape,
} from "../src/tokens.js";

function message(seq: number, body: string, role: CanonicalMessage["role"] = "user"): CanonicalMessage {
  return {
    id: `msg-${seq}`,
    session_id: "sess-1",
    seq,
    request_message_id: "msg-r1",
    role,
    content: [text(body)],
    format_version: 1,
    status: "complete",
  };
}

function shape(messages: CanonicalMessage[], overrides: Partial<RequestShape> = {}): RequestShape {
  return { systemPrompt: "you are helpful", tools: [], summary: null, messages, ...overrides };
}

const budget: BudgetConfig = { ...DEFAULT_BUDGET, contextWindow: 10_000, outputReserve: 1_000 };

test("inputBudget subtracts the output reserve and safety margin", () => {
  assert.equal(inputBudget(budget), 10_000 - 1_000 - 2_000);
  assert.equal(inputBudget({ ...budget, contextWindow: 1_000 }), 0);
});

test("estimates grow with content and stay conservative", () => {
  const small = estimateRequestTokens(shape([message(1, "hi")]));
  const large = estimateRequestTokens(shape([message(1, "hi"), message(2, "x".repeat(3_000))]));
  assert.ok(large > small);
  // 3000 characters at 3 chars/token is 1000 tokens; framing adds a little.
  assert.ok(large >= 1_000);
});

test("compaction triggers at the configured fraction of the input budget", () => {
  const trigger = inputBudget(budget) * DEFAULT_BUDGET.triggerRatio;
  assert.equal(shouldCompact(trigger, budget), true);
  assert.equal(shouldCompact(trigger - 1, budget), false);
  assert.equal(compactTarget(budget), Math.floor(inputBudget(budget) * DEFAULT_BUDGET.targetRatio));
});

test("providerMessages excludes incomplete assistant output", () => {
  const partial = { ...message(2, "half", "assistant"), status: "interrupted" as const };
  assert.deepEqual(providerMessages([message(1, "go"), partial]).map((m) => m.id), ["msg-1"]);
});

test("the estimator reuses the last actual input usage for an unchanged prefix", () => {
  const estimator = new ContextEstimator();
  const first = shape([message(1, "hello")]);
  estimator.recordActualUsage(
    { input_tokens: 1_000, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 1_000 },
    first,
  );

  // Appending one message: baseline + the appended estimate, not a full rescan.
  const appended = shape([message(1, "hello"), message(2, "a".repeat(300))]);
  const estimate = estimator.estimate(appended);
  assert.ok(estimate > 1_000);
  assert.ok(estimate < 1_200, `expected baseline + appended deltas, got ${estimate}`);

  // The reused estimate is exactly the recorded actual usage plus the delta the
  // full recomputation attributes to the appended message.
  const delta = estimateRequestTokens(appended) - estimateRequestTokens(first);
  assert.equal(estimate, 1_000 + delta);
});

test("changing the model-facing prefix invalidates the baseline", () => {
  const estimator = new ContextEstimator();
  const first = shape([message(1, "hello")]);
  estimator.recordActualUsage(
    { input_tokens: 500, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 500 },
    first,
  );

  // System prompt change.
  const changedPrompt = shape([message(1, "hello")], { systemPrompt: "different" });
  assert.equal(estimator.estimate(changedPrompt), estimateRequestTokens(changedPrompt));

  // Tool schema change.
  const changedTools = shape([message(1, "hello")], { tools: [{ name: "bash" }] });
  assert.equal(estimator.estimate(changedTools), estimateRequestTokens(changedTools));

  // Compaction (summary appears) and hydration (different prefix) both fall
  // back to a full recomputation.
  const compacted = shape([message(1, "hello")], { summary: "earlier context" });
  assert.equal(estimator.estimate(compacted), estimateRequestTokens(compacted));

  estimator.invalidate();
  assert.equal(estimator.estimate(first), estimateRequestTokens(first));
});

test("a rewritten history is never treated as an unchanged prefix", () => {
  const estimator = new ContextEstimator();
  const first = shape([message(1, "hello"), message(2, "world")]);
  estimator.recordActualUsage(
    { input_tokens: 800, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 800 },
    first,
  );
  // Same length, different ids (compaction replaced the head).
  const rewritten = shape([message(9, "hello"), message(10, "world")]);
  assert.equal(estimator.estimate(rewritten), estimateRequestTokens(rewritten));
});

test("a missing provider usage report leaves the baseline unset", () => {
  const estimator = new ContextEstimator();
  const shape = { systemPrompt: "s", tools: [], summary: null, messages: [message(1, "hello")] };

  // The provider omitted usage (or reported zero): the estimator must not
  // pretend the request was measured, or every later estimate would inherit a
  // baseline of zero and under-count the context.
  estimator.recordActualUsage(undefined, shape);
  assert.equal(estimator.estimate(shape), estimateRequestTokens(shape));

  estimator.recordActualUsage(
    { input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 0 },
    shape,
  );
  assert.equal(estimator.estimate(shape), estimateRequestTokens(shape));

  // A real report does set one.
  estimator.recordActualUsage(
    { input_tokens: 5_000, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 5_000 },
    shape,
  );
  assert.equal(estimator.estimate(shape), 5_000);
});
