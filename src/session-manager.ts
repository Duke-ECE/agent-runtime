import { randomBytes } from "node:crypto";
import { status } from "@grpc/grpc-js";

export interface Session<TAgent> {
  id: string;
  userId: string;
  status: "active";
  createdAt: Date;
  lastActive: Date;
  agent: TAgent;
}

export interface SessionManagerOptions<TAgent> {
  maxSessions: number;
  ttlMs: number;
  /** Interval for the idle-session reaper. Default: 60_000 (1 minute). */
  reapIntervalMs?: number;
  /** Optional cleanup when a session is removed (e.g. abort an in-flight run). */
  destroyAgent?: (agent: TAgent) => void;
}

export function grpcError(code: status, message: string): Error & { code: status } {
  return Object.assign(new Error(message), { code });
}

export class SessionManager<TAgent> {
  private readonly sessions = new Map<string, Session<TAgent>>();
  private reaper?: NodeJS.Timeout;

  constructor(private readonly options: SessionManagerOptions<TAgent>) {}

  start(): void {
    if (this.reaper) return;
    this.reaper = setInterval(() => this.reap(), this.options.reapIntervalMs ?? 60_000);
    this.reaper.unref();
  }

  stop(): void {
    if (this.reaper) {
      clearInterval(this.reaper);
      this.reaper = undefined;
    }
  }

  get size(): number {
    return this.sessions.size;
  }

  create(userId: string, createAgent: (sessionId: string, userId: string) => TAgent): Session<TAgent> {
    if (this.sessions.size >= this.options.maxSessions) {
      throw grpcError(status.RESOURCE_EXHAUSTED, `session limit reached (${this.options.maxSessions})`);
    }
    const id = `sess-${randomBytes(8).toString("hex")}`;
    const now = new Date();
    const session: Session<TAgent> = {
      id,
      userId,
      status: "active",
      createdAt: now,
      lastActive: now,
      agent: createAgent(id, userId),
    };
    this.sessions.set(id, session);
    return session;
  }

  get(id: string): Session<TAgent> {
    const session = this.sessions.get(id);
    if (!session) {
      throw grpcError(status.NOT_FOUND, `unknown session: ${id}`);
    }
    return session;
  }

  touch(id: string): void {
    const session = this.sessions.get(id);
    if (session) session.lastActive = new Date();
  }

  end(id: string): void {
    const session = this.get(id);
    this.sessions.delete(id);
    this.options.destroyAgent?.(session.agent);
  }

  list(): Session<TAgent>[] {
    return [...this.sessions.values()];
  }

  /** Remove sessions idle longer than ttlMs. Returns the removed ids. */
  reap(now: Date = new Date()): string[] {
    const removed: string[] = [];
    for (const session of this.sessions.values()) {
      if (now.getTime() - session.lastActive.getTime() > this.options.ttlMs) {
        this.sessions.delete(session.id);
        this.options.destroyAgent?.(session.agent);
        removed.push(session.id);
      }
    }
    return removed;
  }
}
