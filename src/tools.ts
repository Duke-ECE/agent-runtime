import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type, type TSchema } from "typebox";

export interface ToolExecutionResult {
  ok: boolean;
  output?: string;
  error?: string;
}

/**
 * Bridge between the agent's tools and the environment that actually executes
 * them (a student sandbox, reached over the network). v1 ships only
 * NullExecutor; a sandbox-backed executor plugs in here later.
 */
export interface ToolExecutor {
  execute(tool: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<ToolExecutionResult>;
}

export const SANDBOX_NOT_CONNECTED = "tool execution unavailable: sandbox not connected";

/** Every tool call fails with a fixed error so the agent loop continues and
 * the LLM can tell the user that tools are unavailable. */
export class NullExecutor implements ToolExecutor {
  async execute(_tool: string, _args: Record<string, unknown>, _signal?: AbortSignal): Promise<ToolExecutionResult> {
    return { ok: false, error: SANDBOX_NOT_CONNECTED };
  }
}

function bridgedTool<T extends TSchema>(
  name: string,
  description: string,
  parameters: T,
  executor: ToolExecutor,
): AgentTool<T> {
  return {
    name,
    label: name,
    description,
    parameters,
    execute: async (_toolCallId, params, signal) => {
      const result = await executor.execute(name, params as Record<string, unknown>, signal);
      // pi convention: throw on failure; the agent reports it to the LLM as a
      // tool error (isError: true) and the loop continues.
      if (!result.ok) {
        throw new Error(result.error ?? "tool execution failed");
      }
      return { content: [{ type: "text", text: result.output ?? "" }], details: {} };
    },
  };
}

// Schemas and descriptions modeled on pi-coding-agent's tool definitions
// (pi/packages/coding-agent/src/core/tools/{read,write,bash,edit}.ts).

const readSchema = Type.Object({
  path: Type.String({ description: "Path to the file to read (relative or absolute)" }),
  offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
  limit: Type.Optional(Type.Number({ description: "Maximum number of lines to read" })),
});

const writeSchema = Type.Object({
  path: Type.String({ description: "Path to the file to write (relative or absolute)" }),
  content: Type.String({ description: "Content to write to the file" }),
});

const bashSchema = Type.Object({
  command: Type.String({ description: "Bash command to execute" }),
  timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (optional, no default timeout)" })),
});

const editSchema = Type.Object({
  path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
  edits: Type.Array(
    Type.Object({
      oldText: Type.String({
        description:
          "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
      }),
      newText: Type.String({ description: "Replacement text for this targeted edit." }),
    }),
    {
      description:
        "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
    },
  ),
});

export function buildTools(executor: ToolExecutor): AgentTool[] {
  return [
    bridgedTool(
      "read",
      "Read the contents of a file. Supports text files. Output may be truncated; use offset/limit for large files.",
      readSchema,
      executor,
    ),
    bridgedTool(
      "write",
      "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
      writeSchema,
      executor,
    ),
    bridgedTool(
      "bash",
      "Execute a bash command in the current working directory. Returns stdout and stderr. Optionally provide a timeout in seconds.",
      bashSchema,
      executor,
    ),
    bridgedTool(
      "edit",
      "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file.",
      editSchema,
      executor,
    ),
  ];
}
