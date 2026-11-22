import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import { CubeToolExecutor } from "../src/cube-executor.js";

function protoPath(): string {
  for (const rel of ["../../proto/sandbox/v3/session_sandbox.proto", "../proto/sandbox/v3/session_sandbox.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("sandbox v3 proto missing");
}

test("Cube executor binds one session and maps all four tools", async () => {
  const definition = protoLoader.loadSync(protoPath(), { keepCase: true, longs: String, defaults: true, oneofs: true });
  const proto = grpc.loadPackageDefinition(definition) as any;
  const server = new grpc.Server();
  const calls: Array<Record<string, unknown>> = [];
  server.addService(proto.sandbox.v3.SessionSandboxService.service, {
    ensureSessionSandbox(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      calls.push({ method: "ensure", token: call.metadata.get("x-service-token")[0], ...call.request });
      callback(null, { state: "SESSION_SANDBOX_STATE_READY" });
    },
    executeTool(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      calls.push({ method: "execute", token: call.metadata.get("x-service-token")[0], ...call.request });
      callback(null, { exit_code: 0, stdout: call.request.read ? "file body" : "ok", stderr: "" });
    },
    checkpointSessionSandbox(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      calls.push({ method: "checkpoint", token: call.metadata.get("x-service-token")[0], ...call.request });
      callback(null, {});
    },
    closeSessionSandbox(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      calls.push({ method: "close", token: call.metadata.get("x-service-token")[0], ...call.request });
      callback(null, {});
    },
  });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, bound) => error ? reject(error) : resolve(bound)));
  const executor = new CubeToolExecutor(`127.0.0.1:${port}`, "secret");
  try {
    assert.deepEqual(await executor.execute("read", { path: "a.txt", offset: 2, limit: 3 }, undefined, "sess-test"), { ok: true, output: "file body" });
    await executor.checkpointSession("sess-test");
    assert.equal(calls.filter((call) => call.method === "checkpoint").length, 0);
    assert.deepEqual(await executor.execute("write", { path: "a.txt", content: "new" }, undefined, "sess-test"), { ok: true, output: "Wrote a.txt" });
    assert.deepEqual(await executor.execute("bash", { command: "echo ok", timeout: 5 }, undefined, "sess-test"), { ok: true, output: "ok" });
    assert.deepEqual(await executor.execute("edit", { path: "a.txt", edits: [{ oldText: "a", newText: "b" }, { oldText: "c", newText: "d" }] }, undefined, "sess-test"), { ok: true, output: "Edited a.txt" });
    await executor.checkpointSession("sess-test");
    await executor.checkpointSession("sess-test");
    assert.equal(calls.filter((call) => call.method === "checkpoint").length, 1);
    await executor.closeSession("sess-test");
    assert.equal(calls.filter((call) => call.method === "ensure").length, 4);
    assert.ok(calls.every((call) => call.session_id === "sess-test"));
    assert.ok(calls.every((call) => call.token === "secret"));
    assert.deepEqual((calls.find((call) => call.read)!.read as Record<string, unknown>), { path: "a.txt", offset_lines: "1", limit_lines: "3" });
    assert.equal((calls.find((call) => call.edit)!.edit as { edits: unknown[] }).edits.length, 2);
  } finally {
    executor.close();
    await new Promise<void>((resolve) => server.tryShutdown(() => resolve()));
  }
});

test("Cube executor cancels an in-flight tool call", async () => {
  const definition = protoLoader.loadSync(protoPath(), { keepCase: true, longs: String, defaults: true, oneofs: true });
  const proto = grpc.loadPackageDefinition(definition) as any;
  const server = new grpc.Server();
  let entered: (() => void) | undefined;
  const inTool = new Promise<void>((resolve) => { entered = resolve; });
  server.addService(proto.sandbox.v3.SessionSandboxService.service, {
    ensureSessionSandbox(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) { callback(null, { state: "SESSION_SANDBOX_STATE_READY" }); },
    executeTool(_call: grpc.ServerUnaryCall<any, any>, _callback: grpc.sendUnaryData<any>) { entered?.(); },
  });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, bound) => error ? reject(error) : resolve(bound)));
  const executor = new CubeToolExecutor(`127.0.0.1:${port}`);
  const controller = new AbortController();
  try {
    const result = executor.execute("bash", { command: "sleep 30" }, controller.signal, "sess-test");
    await inTool;
    controller.abort();
    assert.match((await result).error ?? "", /cancel/i);
  } finally {
    executor.close();
    server.forceShutdown();
  }
});

test("Cube executor marks a failed tool RPC as a fatal turn error", async () => {
  const definition = protoLoader.loadSync(protoPath(), { keepCase: true, longs: String, defaults: true, oneofs: true });
  const proto = grpc.loadPackageDefinition(definition) as any;
  const server = new grpc.Server();
  server.addService(proto.sandbox.v3.SessionSandboxService.service, {
    ensureSessionSandbox(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) { callback(null, { state: "SESSION_SANDBOX_STATE_READY" }); },
    executeTool(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      callback({ code: grpc.status.UNAVAILABLE, message: "checkpoint failed" } as grpc.ServiceError, null);
    },
  });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, bound) => error ? reject(error) : resolve(bound)));
  const executor = new CubeToolExecutor(`127.0.0.1:${port}`);
  try {
    const result = await executor.execute("bash", { command: "touch file" }, undefined, "sess-test");
    assert.equal(result.ok, false);
    assert.match(executor.takeFatalError("sess-test") ?? "", /not have been saved/);
    assert.equal(executor.takeFatalError("sess-test"), undefined);
  } finally {
    executor.close();
    await new Promise<void>((resolve) => server.tryShutdown(() => resolve()));
  }
});

test("Cube executor fails the turn when checkpoint fails", async () => {
  const definition = protoLoader.loadSync(protoPath(), { keepCase: true, longs: String, defaults: true, oneofs: true });
  const proto = grpc.loadPackageDefinition(definition) as any;
  const server = new grpc.Server();
  server.addService(proto.sandbox.v3.SessionSandboxService.service, {
    ensureSessionSandbox(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) { callback(null, { state: "SESSION_SANDBOX_STATE_READY" }); },
    executeTool(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) { callback(null, { exit_code: 0 }); },
    checkpointSessionSandbox(_call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
      callback({ code: grpc.status.UNAVAILABLE, message: "R2 unavailable" } as grpc.ServiceError, null);
    },
  });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, bound) => error ? reject(error) : resolve(bound)));
  const executor = new CubeToolExecutor(`127.0.0.1:${port}`);
  try {
    assert.equal((await executor.execute("write", { path: "a.txt", content: "x" }, undefined, "sess-test")).ok, true);
    await assert.rejects(executor.checkpointSession("sess-test"), /R2 unavailable/);
    assert.match(executor.takeFatalError("sess-test") ?? "", /checkpoint failed/);
  } finally {
    executor.close();
    await new Promise<void>((resolve) => server.tryShutdown(() => resolve()));
  }
});
