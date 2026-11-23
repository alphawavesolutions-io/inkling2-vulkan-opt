import { convertToLlm, serializeConversation, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { checkpoint, endpoint, modelId, providerId, textOf } from "./checkpoint.mjs";

type CachedPrefix = {
  messages: unknown[];
  tools?: unknown[];
  chatTemplateKwargs?: Record<string, unknown>;
  capturedAt: number;
};

export default function (pi: ExtensionAPI) {
  let lastPrefix: CachedPrefix | undefined;

  // Capture the exact OpenAI chat payload sent to Inkling so compaction can reuse
  // llama-server's live KV cache (~98-99% cache hit -> <1s prefill instead of ~80s).
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== providerId || ctx.model?.id !== modelId) return;
    const payload = event.payload as {
      messages?: unknown[];
      tools?: unknown[];
      chat_template_kwargs?: Record<string, unknown>;
    } | undefined;
    if (payload && Array.isArray(payload.messages) && payload.messages.length > 0) {
      lastPrefix = {
        messages: [...payload.messages],
        tools: Array.isArray(payload.tools) ? [...payload.tools] : undefined,
        chatTemplateKwargs: payload.chat_template_kwargs ? { ...payload.chat_template_kwargs } : undefined,
        capturedAt: Date.now(),
      };
    }
  });

  const clearPrefix = () => {
    lastPrefix = undefined;
  };
  pi.on("session_start", clearPrefix);
  pi.on("session_before_switch", clearPrefix);
  pi.on("session_compact", clearPrefix);
  pi.on("session_tree", clearPrefix);

  pi.on("session_before_compact", async (event, ctx) => {
    const expectedBase = endpoint + "/v1";
    if (ctx.model?.provider !== providerId || ctx.model?.id !== modelId ||
        ctx.model.baseUrl.replace(/\/$/, "") !== expectedBase ||
        ctx.model.contextWindow < 32768) return;
    if (event.signal.aborted) return { cancel: true };

    const prepared = event.preparation;
    const previous = event.branchEntries.findLastIndex((entry) => entry.type === "compaction");
    const previousEntry = event.branchEntries[previous];
    const details = previousEntry?.type === "compaction" ? previousEntry.details as
      { readFiles?: string[]; modifiedFiles?: string[] } | undefined : undefined;
    const modified = new Set([...prepared.fileOps.written, ...prepared.fileOps.edited, ...(details?.modifiedFiles ?? [])]);
    const read = new Set([...prepared.fileOps.read, ...(details?.readFiles ?? [])]);
    const files = {
      readFiles: [...read].filter((path) => !modified.has(path)).sort(),
      modifiedFiles: [...modified].sort(),
    };

    const editIds = new Set(event.branchEntries.flatMap((entry) =>
      entry.type === "context_edit" ? [entry.targetId] : []));
    const retainedStart = previousEntry?.type === "compaction" ?
      event.branchEntries.findIndex((entry) => entry.id === previousEntry.firstKeptEntryId) : 0;
    const evidence = event.branchEntries.slice(retainedStart >= 0 ? retainedStart : previous + 1).flatMap((entry) => {
      if (entry.type !== "message" || editIds.has(entry.id)) return [];
      const message = entry.message;
      if (message.role === "user") return [{ label: "User request · " + entry.id, text: textOf(message.content) }];
      if (message.role !== "toolResult") return [];
      return [{
        label: (message.isError ? "Tool error · " : "Tool output · ") + message.toolName + " · " + entry.id,
        text: textOf(message.content),
      }];
    });

    // Build KV-preserving prefix if we captured a fresh request payload (< 9 min, before idle sleep)
    let kvPrefix: { messages: unknown[]; tools?: unknown[]; chatTemplateKwargs?: Record<string, unknown> } | undefined;
    if (lastPrefix && Date.now() - lastPrefix.capturedAt < 540000) {
      const msgs = [...lastPrefix.messages];
      const lastEntry = [...event.branchEntries].reverse().find((e) => e.type === "message");
      if (lastEntry?.type === "message" && lastEntry.message.role === "assistant") {
        const tailText = textOf(lastEntry.message.content);
        const lastCached = msgs[msgs.length - 1] as { role?: string } | undefined;
        if (tailText && lastCached?.role !== "assistant") {
          msgs.push({ role: "assistant", content: tailText });
        }
      }
      kvPrefix = {
        messages: msgs,
        tools: lastPrefix.tools,
        chatTemplateKwargs: lastPrefix.chatTemplateKwargs,
      };
    }

    try {
      const result = await checkpoint({
        history: serializeConversation(convertToLlm([...prepared.messagesToSummarize, ...prepared.turnPrefixMessages])),
        previous: prepared.previousSummary,
        evidence,
        files,
        focus: event.customInstructions,
        kvPrefix,
        contextWindow: ctx.model.contextWindow,
        signal: event.signal,
      });
      if (event.signal.aborted) return { cancel: true };
      if (!result) return;
      const cached = result.usage?.prompt_tokens_details?.cached_tokens ?? 0;
      const input = Math.max(0, (result.usage?.prompt_tokens ?? 0) - cached);
      const output = result.usage?.completion_tokens ?? 0;
      ctx.ui.notify(
        `Inkling checkpoint (${result.usedKvPreserving ? "KV-cached" : "standalone"}, ${cached} cached tok): ` +
        `${result.tokens} tokens, ${result.evidenceCount} evidence excerpts.`,
        "info",
      );
      return { compaction: {
        summary: result.summary,
        firstKeptEntryId: prepared.firstKeptEntryId,
        tokensBefore: prepared.tokensBefore,
        details: files,
        usage: {
          input, output, cacheRead: cached, cacheWrite: 0, reasoning: 0,
          totalTokens: input + output + cached,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
      } };
    } catch {
      if (event.signal.aborted) return { cancel: true };
      ctx.ui.notify("Inkling checkpoint did not finish safely. Pi will use standard compaction.", "warning");
    }
  });
}
