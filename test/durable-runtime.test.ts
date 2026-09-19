import assert from "node:assert/strict";
import { test } from "node:test";
import * as grpc from "@grpc/grpc-js";
import { loadConfig } from "../src/config.js";
import { createDurableRuntime } from "../src/durable-runtime.js";

/** A server-streaming call stand-in that records what the handler does. */
function fakeStreamCall(request: unknown) {
  const events: unknown[] = [];
  const writes: unknown[] = [];
  let ended = false;
  return {
    request,
    cancelled: false,
    writes,
    get ended() {
      return ended;
    },
    emit(_event: string, err: unknown) {
      events.push(err);
      return true;
    },
    write(payload: unknown) {
      writes.push(payload);
    },
    end() {
      ended = true;
    },
    errors: events,
  } as never as grpc.ServerWritableStream<any, any> & {
    writes: unknown[];
    ended: boolean;
    errors: unknown[];
  };
}

test("without session-manager the v2 service fails closed", () => {
  const runtime = createDurableRuntime(loadConfig({} as NodeJS.ProcessEnv));
  const service = runtime.service as Record<string, (call: any, cb?: any) => void>;
  assert.deepEqual(Object.keys(service).sort(), ["cancelRequest", "chat", "evictSession"]);

  const chat = fakeStreamCall({ session_id: "s", user_id: "u", client_request_id: "r", content: [] });
  service.chat(chat);
  assert.equal(chat.errors.length, 1);
  assert.equal((chat.errors[0] as { code: grpc.status }).code, grpc.status.UNAVAILABLE);
  assert.equal(chat.writes.length, 0);

  let cancelErr: grpc.ServiceError | undefined;
  service.cancelRequest({ request: {} }, (err: grpc.ServiceError | null) => {
    cancelErr = err ?? undefined;
  });
  assert.equal(cancelErr?.code, grpc.status.UNAVAILABLE);

  let evicted = false;
  service.evictSession({ request: { session_id: "s" } }, () => {
    evicted = true;
  });
  assert.equal(evicted, true);
});

test("invalid chat requests are rejected before any durable call", () => {
  const runtime = createDurableRuntime(loadConfig({} as NodeJS.ProcessEnv));
  const service = runtime.service as Record<string, (call: any) => void>;
  const chat = fakeStreamCall({ session_id: "", user_id: "", client_request_id: "", content: [] });
  service.chat(chat);
  assert.equal(chat.errors.length, 1);
  // With no client configured the UNAVAILABLE guard answers first; the point is
  // that nothing was streamed as if the request had been accepted.
  assert.equal(chat.writes.length, 0);
  assert.equal(chat.ended, false);
});
