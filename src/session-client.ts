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

/** One transcript message as returned by GetTranscript (ordered by seq). */
export interface TranscriptTurn {
  seq: number;
  role: string;
  contentJson: string;
  createdAt?: string;
}

/**
 * Window selector for GetTranscript. Omit (or pass all-zero fields) for the
 * full transcript — the legacy behavior, kept for callers that need it.
 */
export interface TranscriptWindow {
  /** Max messages to return; 0/undefined = full transcript. */
  limit?: number;
  /** Only messages with seq < beforeSeq; 0/undefined = the latest window. */
  beforeSeq?: number;
}

/** GetTranscript result: the requested window plus the pagination marker. */
export interface TranscriptResult {
  messages: TranscriptTurn[];
  /** True when older messages exist before the returned window. */
  hasMore: boolean;
}

/** Bound on how long CreateSession may wait for a transcript fetch. */
const GET_TRANSCRIPT_TIMEOUT_MS = 5_000;

/**
 * Client for session-manager (session.v1.SessionService). AppendTurn and
 * SetTitle are fire-and-log: a failure is warned about but never surfaced to
 * the chat path. GetTranscript backs CreateSession hydration and is fail-open
 * too: any error resolves to null so a session can always start with empty
 * history.
 */
export interface SessionClient {
  appendTurn(sessionId: string, userId: string, messages: TurnMessageInput[]): void;
  setTitle(sessionId: string, title: string): void;
  getTranscript(sessionId: string, window?: TranscriptWindow): Promise<TranscriptResult | null>;
}

interface SessionServiceClient {
  appendTurn(
    req: unknown,
    metadata: grpc.Metadata,
    callback: (err: grpc.ServiceError | null, res: unknown) => void,
  ): void;
  setTitle(
    req: unknown,
    metadata: grpc.Metadata,
    callback: (err: grpc.ServiceError | null, res: unknown) => void,
  ): void;
  getTranscript(
    req: unknown,
    metadata: grpc.Metadata,
    options: grpc.CallOptions,
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
export function createSessionClient(config: ServiceConfig): SessionClient | undefined {
  if (!config.sessionManagerAddr) return undefined;
  const addr = config.sessionManagerAddr;
  let client: SessionServiceClient | undefined;

  // The gRPC client is created lazily on the first call so the runtime
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

    // Fire-and-log like AppendTurn: a title is cosmetic, so failures only warn.
    setTitle(sessionId, title) {
      const metadata = new grpc.Metadata();
      if (config.serviceToken) metadata.set("x-service-token", config.serviceToken);
      try {
        getClient().setTitle({ session_id: sessionId, title }, metadata, (err) => {
          if (err) console.warn(`session-manager SetTitle failed for ${sessionId}: ${err.message}`);
        });
      } catch (err) {
        console.warn(`session-manager SetTitle failed for ${sessionId}:`, err);
      }
    },

    // Token-only GetTranscript (no user_id): the service token authorizes the
    // runtime hydration path on session-manager. Fail-open on any error.
    // `window` selects the latest `limit` messages (before_seq 0 = from the
    // end); omitting it requests the full transcript (limit 0).
    getTranscript(sessionId, window) {
      return new Promise<TranscriptResult | null>((resolve) => {
        let serviceClient: SessionServiceClient;
        try {
          serviceClient = getClient();
        } catch (err) {
          console.warn(`session-manager GetTranscript failed for ${sessionId}:`, err);
          resolve(null);
          return;
        }
        const metadata = new grpc.Metadata();
        if (config.serviceToken) metadata.set("x-service-token", config.serviceToken);
        const options: grpc.CallOptions = { deadline: Date.now() + GET_TRANSCRIPT_TIMEOUT_MS };
        const request = {
          session_id: sessionId,
          limit: window?.limit ?? 0,
          before_seq: window?.beforeSeq ?? 0,
        };
        try {
          serviceClient.getTranscript(request, metadata, options, (err, res) => {
            if (err) {
              console.warn(`session-manager GetTranscript failed for ${sessionId}: ${err.message}`);
              resolve(null);
              return;
            }
            const body = res as {
              messages?: Array<{ seq?: number; role?: string; content_json?: string; created_at?: string }>;
              has_more?: boolean;
            } | null;
            const messages = body?.messages ?? [];
            resolve({
              messages: messages.map((m) => ({
                seq: Number(m.seq ?? 0),
                role: String(m.role ?? ""),
                contentJson: typeof m.content_json === "string" ? m.content_json : String(m.content_json ?? ""),
                createdAt: m.created_at || undefined,
              })),
              hasMore: body?.has_more === true,
            });
          });
        } catch (err) {
          console.warn(`session-manager GetTranscript failed for ${sessionId}:`, err);
          resolve(null);
        }
      });
    },
  };
}
