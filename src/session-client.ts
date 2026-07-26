import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import type { ServiceConfig } from "./config.js";

/** One transcript message for AppendTurn; seq is assigned by session-manager. */
export interface TurnMessageInput {
  role: string;
  contentJson: string;
  createdAt: Date;
}

/**
 * Write-through to session-manager (session.v1.SessionService). AppendTurn is
 * fire-and-log: a failure is warned about but never surfaced to the chat path.
 */
export interface SessionWriter {
  appendTurn(sessionId: string, userId: string, messages: TurnMessageInput[]): void;
}

interface SessionServiceClient {
  appendTurn(
    req: unknown,
    metadata: grpc.Metadata,
    callback: (err: grpc.ServiceError | null, res: unknown) => void,
  ): void;
  close(): void;
}

function resolveProtoPath(): string {
  // dist/src/session-client.js -> <root>/proto/... ; also works when run from src/.
  for (const rel of ["../../proto/session/v1/session.proto", "../proto/session/v1/session.proto"]) {
    const candidate = fileURLToPath(new URL(rel, import.meta.url));
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("proto/session/v1/session.proto not found; run `npm run sync-proto`");
}

/** Returns undefined when SESSION_MANAGER_ADDR is not configured. */
export function createSessionWriter(config: ServiceConfig): SessionWriter | undefined {
  if (!config.sessionManagerAddr) return undefined;
  const addr = config.sessionManagerAddr;
  let client: SessionServiceClient | undefined;

  // The gRPC client is created lazily on the first append so the runtime
  // starts (and tests run) without session-manager being reachable.
  function getClient(): SessionServiceClient {
    if (!client) {
      const packageDefinition = protoLoader.loadSync(resolveProtoPath(), {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      });
      const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as {
        session: { v1: { SessionService: grpc.ServiceClientConstructor } };
      };
      client = new proto.session.v1.SessionService(
        addr,
        grpc.credentials.createInsecure(),
      ) as unknown as SessionServiceClient;
    }
    return client;
  }

  return {
    appendTurn(sessionId, userId, messages) {
      const metadata = new grpc.Metadata();
      if (config.serviceToken) metadata.set("x-service-token", config.serviceToken);
      const request = {
        session_id: sessionId,
        user_id: userId,
        messages: messages.map((m) => ({
          seq: 0, // assigned server-side
          role: m.role,
          content_json: m.contentJson,
          created_at: m.createdAt.toISOString(),
        })),
      };
      try {
        getClient().appendTurn(request, metadata, (err) => {
          if (err) console.warn(`session-manager AppendTurn failed for ${sessionId}: ${err.message}`);
        });
      } catch (err) {
        console.warn(`session-manager AppendTurn failed for ${sessionId}:`, err);
      }
    },
  };
}
