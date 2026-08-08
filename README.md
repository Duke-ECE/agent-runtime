# agent-runtime

gRPC service that hosts agent sessions for the Duke-ECE managed-agents platform.
Each session embeds a [pi](https://github.com/earendil-works/pi) agent
(`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`) pointed at any
OpenAI-compatible LLM endpoint.

## API

Implements `runtime.v1.AgentService` from the
[Duke-ECE/protos](https://github.com/Duke-ECE/protos) repo
(`proto/runtime/v1/agent.proto`, pinned to tag `v0.7.0`). The proto is loaded
at runtime with `@grpc/grpc-js` + `@grpc/proto-loader` — no codegen.

Methods: `CreateSession`, `EndSession`, `ListSessions`, `Chat` (server-streaming:
`text_delta` / `tool_call` / `tool_result` / `error` / `done`).

To refresh the pinned proto copy:

```bash
npm run sync-proto
```

## Sessions

In-memory session map. `CreateSession` issues `sess-<random>` ids and enforces
`MAX_SESSIONS` (gRPC `RESOURCE_EXHAUSTED` when full). `EndSession` deletes the
session; unknown sessions yield `NOT_FOUND`. A reaper runs every minute and
removes sessions idle longer than `SESSION_TTL_MINUTES`.

When `SESSION_MANAGER_ADDR` is set, each completed Chat turn is written through
to session-manager via `session.v1.AppendTurn` (fire-and-log). If `CreateSession`
then arrives with a caller-provided `session_id` that is not live in memory
(e.g. after a runtime restart), the runtime fetches the durable transcript via
`session.v1.GetTranscript` (token-only, `x-service-token` metadata) and seeds
the new agent's history with it. Hydration is fail-open: any error starts the
session with empty history.

Hydration is **windowed**: only the latest `HYDRATION_MAX_TURNS` transcript
messages are requested (`limit=HYDRATION_MAX_TURNS`, `before_seq=0`), so
resume cost stays bounded as sessions grow. When the response's `has_more` is
true (older history exists before the window) the runtime logs that earlier
history was skipped. A window that opens mid-tool-pair hydrates cleanly — the
orphan `tool_result` at the window's start is skipped like any other
unpairable entry. One tradeoff: the system-prompt fallback lives in the
transcript's LAST `system` turn, so if the window contains no system turn the
fallback is simply absent (empty), the same as resuming a prompt-less
session. This only affects sessions whose template was deleted AND whose
transcript outgrew the window — a live template's request `system_prompt`
always wins over the transcript.

Auto session titles: after the **first completed Chat turn** of a session that
began empty (not hydrated, no seeded history), the runtime makes one detached
one-shot call to the session's own model ("title this conversation", seeded
with the first user message), sanitizes the result (first line, trimmed,
surrounding quotes stripped, capped at 80 chars), and reports it via
`session.v1.SetTitle` (fire-and-log, `x-service-token`). Failures — LLM or
session-manager — only warn and never affect the chat path.

LLM config resolution per session: `CreateSession.llm` (api_key / base_url /
model) wins; anything absent falls back to `LLM_API_KEY` / `LLM_BASE_URL` /
`LLM_MODEL` env vars.

`CreateSession` also accepts an optional `system_prompt` (empty = no system
prompt — the runtime carries no built-in default; the platform's "Default
assistant" template lives in the backend's template store, which sends the
prompt explicitly when one is wanted) and an optional `tools` whitelist
(empty = all built-ins; unknown names are rejected with `INVALID_ARGUMENT`).

The session's system prompt is persisted in the durable transcript as
`role: "system"` messages (content_json `{"content": "<prompt>"}`, the same
shape as user turns): a session whose current prompt is non-empty records it in
the next `AppendTurn` whenever it differs from the last system message already
recorded, placed before the turn's other messages. On resume, prompt precedence
is request `system_prompt` (non-empty) > the transcript's last system message >
none — so a live template's current value wins, while a session whose template
was deleted keeps its persona from the transcript. Hydration seeds the
"last recorded" marker from the transcript's last system message, so an
unchanged prompt is not re-recorded; system turns never become pi conversation
messages (the last one wins, malformed ones are skipped with a warning).

Assistant transcript messages also carry the turn's token usage: content_json
is `{"content": "...", "usage": {"input_tokens": N, "output_tokens": M}}`
(JSON numbers, the same counts sent in the Chat `done` event; the `usage` key
is omitted when the provider reported none). The frontend reads it to show
historical usage. User, system, and tool messages keep their plain shapes.
Hydration ignores the extra key (hydrated assistant messages get zero usage).

## Tools

The agent exposes `read` / `write` / `bash` / `edit` tools (schemas modeled on
pi-coding-agent). Execution goes through a `ToolExecutor` interface; v1 ships
only `NullExecutor`, which fails every call with
`tool execution unavailable: sandbox not connected` so the agent loop continues
and the LLM can tell the user tools are unavailable. A sandbox-backed executor
plugs into `ToolExecutor` later.

## Environment variables

| Variable              | Default                       | Description                                   |
| --------------------- | ----------------------------- | --------------------------------------------- |
| `PORT`                | `50052`                       | gRPC listen port                              |
| `MAX_SESSIONS`        | `20`                          | Max concurrent sessions (`RESOURCE_EXHAUSTED`)|
| `SESSION_TTL_MINUTES` | `30`                          | Idle session TTL; reaper runs every minute    |
| `HYDRATION_MAX_TURNS` | `50`                          | Transcript window fetched on session resume   |
| `LLM_API_KEY`         | _(none)_                      | Fallback API key when `CreateSession.llm` omits it |
| `LLM_BASE_URL`        | `https://api.openai.com/v1`   | Fallback OpenAI-compatible base URL           |
| `LLM_MODEL`           | `gpt-4o-mini`                 | Fallback model id                             |

The server boots fine with no `LLM_*` env set; `Chat` then returns an `error`
event explaining the LLM is not configured.

## Local development

```bash
npm install
npm run build      # tsc -> dist/
npm test           # build + node:test unit tests
npm start          # node dist/src/server.js (listens on :50052)
```

## Deployment

`Dockerfile` builds on `node:22-alpine` and runs the compiled output.
`k8s.yaml` deploys one replica behind a ClusterIP `Service` on port 50052;
the LLM API key comes from the optional secret `agent-runtime-llm` (key
`api_key`). CI/CD (`.github/workflows/ci-cd.yml`) runs build+test, pushes
`ghcr.io/duke-ece/agent-runtime`, and rolls out to the cluster (requires the
`KUBE_CONFIG` repo secret).
