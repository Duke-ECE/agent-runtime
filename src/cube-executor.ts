import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import type { ToolExecutionResult, ToolExecutor } from "./tools.js";

type UnaryCallback = (error: grpc.ServiceError | null, response: Record<string, unknown>) => void;
type UnaryMethod = (request: Record<string, unknown>, callback: UnaryCallback) => grpc.ClientUnaryCall;
type CubeClient = grpc.Client & Record<string, UnaryMethod>;

function protoPath(): string {
  for (const rel of ["../../proto/sandbox/v3/session_sandbox.proto", "../proto/sandbox/v3/session_sandbox.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("sandbox.v3 proto missing; run npm run sync-proto");
}

function stringArg(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  if (typeof value !== "string") throw new Error(`${key} must be a string`);
  return value;
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function unary(client: CubeClient, method: string, request: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
  if (signal?.aborted) return Promise.reject(new Error("tool canceled"));
  return new Promise((resolve, reject) => {
    let call: grpc.ClientUnaryCall;
    const onAbort = () => call.cancel();
    call = client[method](request, (error, response) => {
      signal?.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve(response);
    });
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) call.cancel();
  });
}

function toolPayload(tool: string, args: Record<string, unknown>): Record<string, unknown> {
  switch (tool) {
    case "read":
      return {
        read: {
          path: stringArg(args, "path"),
          offset_lines: (positiveInteger(args.offset) ?? 1) - 1,
          limit_lines: positiveInteger(args.limit) ?? 0,
        },
      };
    case "write":
      return { write: { path: stringArg(args, "path"), content: stringArg(args, "content") } };
    case "bash":
      return { bash: { command: stringArg(args, "command") }, timeout_seconds: positiveInteger(args.timeout) ?? 0 };
    case "edit": {
      const edits = args.edits;
      if (!Array.isArray(edits) || edits.length === 0) throw new Error("edits must be a non-empty array");
      return {
        edit: {
          path: stringArg(args, "path"),
          edits: edits.map((item: unknown) => {
            if (!item || typeof item !== "object") throw new Error("invalid edit");
            const entry = item as Record<string, unknown>;
            return { old_text: stringArg(entry, "oldText"), new_text: stringArg(entry, "newText") };
          }),
        },
      };
    }
    default:
      throw new Error(`unknown tool: ${tool}`);
  }
}

export class CubeToolExecutor implements ToolExecutor {
  private readonly client: CubeClient;

  constructor(address: string) {
    const definition = protoLoader.loadSync(protoPath(), { keepCase: true, longs: String, defaults: true, oneofs: true });
    const packageObject = grpc.loadPackageDefinition(definition) as unknown as {
      sandbox: { v3: { SessionSandboxService: grpc.ServiceClientConstructor } };
    };
    this.client = new packageObject.sandbox.v3.SessionSandboxService(address, grpc.credentials.createInsecure()) as CubeClient;
  }

  close(): void { this.client.close(); }

  async closeSession(sessionId: string): Promise<void> {
    if (!sessionId) return;
    await unary(this.client, "closeSessionSandbox", { session_id: sessionId });
  }

  async execute(tool: string, args: Record<string, unknown>, signal?: AbortSignal, sessionId?: string): Promise<ToolExecutionResult> {
    if (!sessionId) return { ok: false, error: "session id is required for sandbox tools" };
    try {
      const payload = toolPayload(tool, args);
      await unary(this.client, "ensureSessionSandbox", { session_id: sessionId }, signal);
      const response = await unary(this.client, "executeTool", { session_id: sessionId, ...payload }, signal);
      const exitCode = Number(response.exit_code ?? 0);
      const stdout = String(response.stdout ?? "");
      const stderr = String(response.stderr ?? "");
      const output = [stdout, stderr].filter(Boolean).join(stderr && stdout ? "\n" : "");
      if (exitCode !== 0) return { ok: false, error: `exit code ${exitCode}${output ? `\n${output}` : ""}` };
      if (tool === "write") return { ok: true, output: `Wrote ${stringArg(args, "path")}` };
      if (tool === "edit") return { ok: true, output: `Edited ${stringArg(args, "path")}` };
      return { ok: true, output: output + (response.truncated ? "\n[output truncated]" : "") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}
