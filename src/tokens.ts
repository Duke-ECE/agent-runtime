import type { CanonicalBlock, CanonicalMessage, CanonicalUsage } from "./canonical.js";

/**
 * Model-aware token estimation and input-budget accounting.
 *
 * The estimator is a disposable cache, never authoritative session state: it is
 * rebuilt from the assembled request whenever the model, system instructions,
 * tool schemas, or retained context change, and re-calibrated from the actual
 * input usage of the latest individual model call. A conservative local
 * estimate is used when the provider reports no usage.
 */

/** Conservative characters-per-token ratio for the fallback estimator. */
const CHARS_PER_TOKEN = 3;

/** Per-message framing overhead (role markers, delimiters) in tokens. */
const MESSAGE_FRAMING_TOKENS = 4;

/** Per-tool schema overhead in tokens, on top of the serialized schema. */
const TOOL_FRAMING_TOKENS = 6;

export interface ToolSchema {
  name: string;
  description?: string;
  parameters?: unknown;
}

export interface RequestShape {
  /** Effective system instructions. */
  systemPrompt: string;
  /** Tool schemas exposed to the model. */
  tools: ToolSchema[];
  /** Active compaction summary, if any, labelled as historical context. */
  summary: string | null;
  /**
   * The exact messages that will be sent to the provider, in order. Use
   * {@link providerMessages} to derive them from canonical history.
   */
  messages: CanonicalMessage[];
}

export interface BudgetConfig {
  /** The model's real context window in tokens. */
  contextWindow: number;
  /** Tokens reserved for the model's reply. */
  outputReserve: number;
  /** Absolute safety margin subtracted after the reserve. */
  safetyMargin: number;
  /** Compact when the estimate reaches this fraction of the budget. */
  triggerRatio: number;
  /** Aim for at most this fraction of the budget after compaction. */
  targetRatio: number;
}

export const DEFAULT_BUDGET: Omit<BudgetConfig, "contextWindow" | "outputReserve"> = {
  safetyMargin: 2_000,
  triggerRatio: 0.8,
  targetRatio: 0.6,
};

/** Tokens available for input: the window minus the output reserve and margin. */
export function inputBudget(config: BudgetConfig): number {
  return Math.max(0, config.contextWindow - config.outputReserve - config.safetyMargin);
}

/** Conservative token estimate for a string. */
export function estimateTextTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Conservative token estimate for a block list, including nested results. */
export function estimateBlocksTokens(blocks: CanonicalBlock[]): number {
  let total = 0;
  for (const block of blocks) {
    switch (block.type) {
      case "text":
        total += estimateTextTokens(block.text);
        break;
      case "tool_call":
        total += estimateTextTokens(block.name) + estimateTextTokens(block.arguments_json) + 8;
        break;
      case "tool_result":
        total += estimateBlocksTokens(block.content) + 8;
        break;
      case "opaque":
        total += estimateTextTokens(block.opaque_type) + Math.ceil(block.data_json.length / CHARS_PER_TOKEN);
        break;
    }
  }
  return total;
}

/** Conservative token estimate for one canonical message. */
export function estimateMessageTokens(message: CanonicalMessage): number {
  return estimateBlocksTokens(message.content) + MESSAGE_FRAMING_TOKENS;
}

export function estimateMessagesTokens(messages: CanonicalMessage[]): number {
  let total = 0;
  for (const message of messages) total += estimateMessageTokens(message);
  return total;
}

function estimateToolsTokens(tools: ToolSchema[]): number {
  let total = 0;
  for (const tool of tools) {
    total += estimateTextTokens(tool.name) + estimateTextTokens(tool.description ?? "");
    if (tool.parameters !== undefined) total += estimateTextTokens(JSON.stringify(tool.parameters));
    total += TOOL_FRAMING_TOKENS;
  }
  return total;
}

/** Full recomputation of the assembled request's input size. */
export function estimateRequestTokens(shape: RequestShape): number {
  let total = estimateTextTokens(shape.systemPrompt);
  total += estimateToolsTokens(shape.tools);
  if (shape.summary) total += estimateTextTokens(shape.summary) + MESSAGE_FRAMING_TOKENS;
  total += estimateMessagesTokens(shape.messages);
  return total;
}

/** Compact when the estimate reaches the trigger fraction of the budget. */
export function shouldCompact(estimate: number, config: BudgetConfig): boolean {
  const budget = inputBudget(config);
  return budget > 0 && estimate >= budget * config.triggerRatio;
}

/** The estimate compaction should bring the context under, or null when impossible. */
export function compactTarget(config: BudgetConfig): number | null {
  const budget = inputBudget(config);
  if (budget <= 0) return null;
  return Math.floor(budget * config.targetRatio);
}

/**
 * The messages a provider request will actually contain: the eligible retained
 * canonical messages. Partial/interrupted assistant output is excluded by
 * default so unfinished model output is never replayed as if it had completed.
 */
export function providerMessages(messages: CanonicalMessage[]): CanonicalMessage[] {
  return messages.filter((message) => message.role !== "assistant" || message.status === "complete");
}

function shapeKey(shape: RequestShape): string {
  return [
    shape.summary ?? "",
    shape.systemPrompt,
    shape.tools.map((tool) => tool.name).join(","),
  ].join("\u0000");
}

function messageKey(message: CanonicalMessage): string {
  return `${message.id}:${message.seq}`;
}

/**
 * Tracks the last request's shape and the provider's reported input usage so a
 * request whose context is an unchanged prefix of the previous one can be
 * estimated as "last actual input + the newly appended content" instead of
 * being re-serialized from scratch. Any change to the model-facing prefix
 * (system prompt, tools, summary) invalidates the baseline and forces a full
 * recomputation.
 */
export class ContextEstimator {
  private baselineKey?: string;
  private baselinePrefix?: string[];
  private baselineActual?: number;

  /** Record the provider's actual input usage for a request that was sent. */
  recordActualUsage(usage: CanonicalUsage | undefined, shape: RequestShape): void {
    if (!usage || usage.input_tokens <= 0) return;
    this.baselineKey = shapeKey(shape);
    this.baselinePrefix = shape.messages.map(messageKey);
    this.baselineActual = usage.input_tokens;
  }

  /** Invalidate the baseline (model change, compaction, hydration, lease loss). */
  invalidate(): void {
    this.baselineKey = undefined;
    this.baselinePrefix = undefined;
    this.baselineActual = undefined;
  }

  /** Estimate the input size of the next request. */
  estimate(shape: RequestShape): number {
    if (this.baselineKey !== shapeKey(shape) || this.baselinePrefix === undefined || this.baselineActual === undefined) {
      return estimateRequestTokens(shape);
    }
    const keys = shape.messages.map(messageKey);
    if (keys.length < this.baselinePrefix.length) {
      return estimateRequestTokens(shape);
    }
    for (let i = 0; i < this.baselinePrefix.length; i++) {
      if (keys[i] !== this.baselinePrefix[i]) {
        return estimateRequestTokens(shape);
      }
    }
    let appended = 0;
    for (let i = this.baselinePrefix.length; i < shape.messages.length; i++) {
      appended += estimateMessageTokens(shape.messages[i]);
    }
    return this.baselineActual + appended;
  }
}
