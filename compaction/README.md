# Universal KV-Preserving Compaction for Inkling-2-Mini

This folder documents a **client-agnostic HTTP compaction pattern** for Inkling-2-Mini on `llama-server` that cuts ~21K-token compaction time from **96.4 s → 15.9 s (6.1× faster)** by preserving **99.4% of the live KV cache** (`79.0 s → 0.91 s` prefill), plus a zero-dependency vanilla JS implementation ([`checkpoint.mjs`](checkpoint.mjs)) and two **optional** reference adapters under [`examples/`](examples/).

---

## Universal HTTP Recipe (Works With Any Client, Language, or Agent Loop)

Instead of serializing the conversation into a brand-new standalone prompt (which starts from token 0 and forces `llama-server` to re-prefill the entire session), send the **exact same prefix** (`messages`, `tools`, and `chat_template_kwargs`) already cached in the server slot and append one compaction user turn:

```json
POST http://127.0.0.1:18089/v1/chat/completions
{
  "model": "inkling-2-mini-uncensored",
  "temperature": 0.2,
  "max_tokens": 2400,
  "cache_prompt": true,
  "reasoning_budget_tokens": 32,
  "tools": [ "...exact same tool definitions from the active session..." ],
  "tool_choice": "none",
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "inkling_checkpoint",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "objective": { "type": "array", "maxItems": 2, "items": { "type": "string", "minLength": 1 } },
          "decisions": { "type": "array", "maxItems": 4, "items": { "type": "string", "minLength": 1 } },
          "verified":  { "type": "array", "maxItems": 4, "items": { "type": "string", "minLength": 1 } },
          "questions": { "type": "array", "maxItems": 3, "items": { "type": "string", "minLength": 1 } },
          "next":      { "type": "array", "maxItems": 3, "items": { "type": "string", "minLength": 1 } }
        },
        "required": ["objective", "decisions", "verified", "questions", "next"],
        "additionalProperties": false
      }
    }
  },
  "messages": [
    "...exact prior messages already in llama-server's KV cache...",
    {
      "role": "user",
      "content": "[SESSION COMPACTION REQUEST — DO NOT CALL TOOLS]\nSummarize the conversation above into the required JSON checkpoint now."
    }
  ]
}
```

### Why Every Field Above Matters on Inkling-2-Mini

1. **Do NOT override `reasoning_effort` on KV-preserving requests**: Inkling's Jinja template injects `Thinking effort level: 0.2` right before the first user turn. Changing `reasoning_effort` to `"none"` alters token ~300 and invalidates 99% of the KV cache.
2. **Keep `tools` identical + set `"tool_choice": "none"`**: Inkling renders tool declarations at token 0 (`<|message_system|>tool_declare...`). Passing identical `tools` keeps tokens `0..N` byte-identical, while `tool_choice: "none"` tells `llama-server` to allow `response_format` JSON schema grammar without tool-calling grammar conflict.
3. **Always set `"reasoning_budget_tokens": 32`**:
   - Setting `0` when `reasoning_effort > 0` triggers an empty `<|content_thinking|><|end_message|>` infinite loop.
   - Omitting `reasoning_budget_tokens` in cold standalone mode (even with `reasoning_effort: "none"`) still lets Inkling emit 1,600+ tokens of reasoning on complex coding transcripts and hit `max_tokens`.
4. **Omit `maxLength` from the JSON schema**: Enforcing `maxLength` inside GBNF over Inkling's 201K vocabulary compiles a large grammar automaton; clamp string lengths in your client code instead.

---

## Using the Zero-Dependency `checkpoint.mjs` Engine

[`checkpoint.mjs`](checkpoint.mjs) has **zero external dependencies** (standard `fetch()` only) and supports both KV-preserving mode (when `kvPrefix` is supplied, with automatic fallback) and cold standalone mode:

```js
import { checkpoint } from "./checkpoint.mjs";

const res = await checkpoint({
  history: serializedTranscript,
  evidence: [{ label: "Tool output · pytest", text: "14 passed in 3.42s" }],
  kvPrefix: { messages: lastSentMessages, tools: lastSentTools, chatTemplateKwargs }, // optional
  contextWindow: 65536,
});
```

Configure via environment variables if your endpoint differs:
- `INKLING_ENDPOINT` (default: `http://127.0.0.1:18089`)
- `INKLING_PROVIDER` (default: `inkling-local`)
- `INKLING_MODEL` (default: `inkling-2-mini-uncensored`)

---

## Optional Reference Adapters (`examples/`)

These two thin adapters are completely optional examples showing how to integrate `checkpoint.mjs` into specific coding harnesses:

- **Pi (`examples/pi/index.ts`)**: Copy `checkpoint.mjs` and `examples/pi/index.ts` into `~/.pi/agent/extensions/inkling-evidence-compaction/` and set `contextWindow` in `~/.pi/agent/models.json` (`65536` for `2×64K` or `32768` for `2×32K`). Uses `before_provider_request` to achieve 99.4% KV-cached compaction.
- **OpenCode (`examples/opencode/index.js`)**: Copy `checkpoint.mjs` and `examples/opencode/index.js` into `~/.config/opencode/plugins/inkling-evidence-compaction/`, set `limit.context` in `~/.config/opencode/opencode.jsonc`, and restart OpenCode. Uses hardened standalone compaction (`reasoning_budget_tokens: 32`, `max_tokens: 2400`).
