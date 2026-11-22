import assert from "node:assert/strict";
import { test } from "node:test";
import { loadCompleteTranscript } from "../src/history.js";
import type { SessionClient, TranscriptTurn } from "../src/session-client.js";

const turn = (seq: number): TranscriptTurn => ({ seq, role: "user", contentJson: "{}" });

test("complete thinking history loads older windows in order", async () => {
  const calls: number[] = [];
  const client = {
    appendTurn() {}, setTitle() {},
    async getTranscript(_sessionId: string, window?: { beforeSeq?: number }) {
      calls.push(window?.beforeSeq ?? 0);
      if (!window?.beforeSeq) return { messages: [turn(4), turn(5)], hasMore: true };
      return { messages: [turn(1), turn(2), turn(3)], hasMore: false };
    },
  } as SessionClient;
  const messages = await loadCompleteTranscript(client, "s-1");
  assert.deepEqual(messages?.map((m) => m.seq), [1, 2, 3, 4, 5]);
  assert.deepEqual(calls, [0, 4]);
});

test("complete thinking history rejects a repeated page", async () => {
  const client = {
    appendTurn() {}, setTitle() {},
    async getTranscript() { return { messages: [turn(4), turn(5)], hasMore: true }; },
  } as SessionClient;
  assert.equal(await loadCompleteTranscript(client, "s-1"), null);
});
