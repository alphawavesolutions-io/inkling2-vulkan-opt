import { checkpoint, modelId, providerId, textOf } from "./checkpoint.mjs";

export default {
  id: "local.inkling-compaction",
  async setup(ctx) {
    await ctx.session.hook("compaction", async (event) => {
      if (event.model?.providerID !== providerId || event.model?.id !== modelId) return;
      const entries = event.messages ?? [];
      const evidence = entries.flatMap((entry, index) => {
        if (entry.role === "user") return [{ label: "User request · " + (index + 1), text: textOf(entry.content) }];
        const parts = Array.isArray(entry.content) ? entry.content : [];
        const results = parts.filter((part) => part.type === "tool-result" || part.type === "toolResult");
        if (entry.role === "tool" && !results.length) {
          return [{ label: "Tool output · " + (index + 1), text: textOf(entry.content) }];
        }
        return results.map((part) => ({
          label: "Tool output · " + (part.toolName ?? part.name ?? (index + 1)),
          text: textOf(part.output ?? part.result ?? part.content),
        }));
      });
      const history = entries.filter(({ role }) => role !== "system").map((entry) =>
        "[" + entry.role + "] " + textOf(entry.content).slice(0, 3500)).join("\n\n");
      const contextWindow = event.model?.limit?.context ?? 65536;
      try {
        const result = await checkpoint({ history, evidence, contextWindow, signal: event.signal });
        if (result && !event.signal?.aborted) event.result = { summary: result.summary };
      } catch {
        if (!event.signal?.aborted) console.warn("Inkling checkpoint did not finish safely. OpenCode will use standard compaction.");
      }
    });
  },
};
