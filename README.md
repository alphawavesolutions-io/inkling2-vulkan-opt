# Inkling-2-Mini Vulkan & Context Optimizations (`llama.cpp` + AMD RDNA 3.5)

> **Works with any OpenAI-compatible client or coding harness** (Open WebUI, Cline, Roo Code, Continue, Aider, Zed, SillyTavern, custom Python/JS agents, Pi, OpenCode, etc.).
>
> **+26% to +33% faster raw token generation** (7,145 → 1,881 Vulkan dispatches/token), **6.1× faster session compaction** (96.4 s → 15.9 s via 99.4% KV-cache reuse), and **2× larger per-slot context** (32K → 64K FP16 KV) — with **zero weight changes, zero quantization loss, and 100% bit-exact short-context greedy outputs**.

Tested on **Windows 11 + Vulkan (AMD Radeon 8060S `gfx1151`, Ryzen AI Max+ 395, 64 GB unified LPDDR5X)** running **`Inkling-2-Mini-Uncensored-AD-Q4_K.gguf`** (`inkling2_loop` 26.07B MoE, 8/128 active experts, 24 physical layers looping blocks 6–17 × 3 = 48 executed layers) on top of [`llama.cpp` `b12628`](https://github.com/ggml-org/llama.cpp) (`8a97fa98a`).

---

## How This Repo Is Organized (Client-Agnostic First)

| Layer | Scope | Who Benefits? |
|---|---|---|
| **1. `llama.cpp` Vulkan Patchset (`patches/`)** | Server / GPU engine | **Everyone.** Every client talking to `llama-server` gets the full +26% to +33% decode speedup, +12% to +16% prefill speedup, and -73.7% GPU dispatch reduction automatically. |
| **2. Optimal Server Flags (`scripts/`)** | `llama-server` launch config | **Everyone.** Doubles per-slot context (`2×32K` → `2×64K` FP16 KV at 23.1 GiB) and avoids the `-ub 1024` Vulkan host-spill trap. |
| **3. Universal KV-Preserving Compaction (`compaction/`)** | Standard `/v1/chat/completions` HTTP recipe + zero-dep `checkpoint.mjs` | **Any agent or custom loop** that compacts/summarizes long sessions against `llama-server`. |
| **4. Optional Reference Adapters (`compaction/examples/`)** | Thin drop-in glue for Pi & OpenCode | **Optional examples** showing how to wire the universal compaction engine into two real harnesses. |

---

## Verified Improvements vs. Baseline (Analytics)

### 1. Paired Alternating A/B: Stock `b12628` vs. Patched Build

All Track A comparisons used **identical production flags** (`--ctx-size 65536 --parallel 2 --no-kv-unified -b 512 -ub 256 -ctk f16 -ctv f16`, Flash Attention on, full GPU offload, `reasoning_effort=low`), alternating `control → optimized → control → optimized` on the same machine to rule out thermal or power drift:

| Metric | Stock `b12628` (Baseline) | Optimized (`ea45100`) | Delta vs. Baseline |
|---|---:|---:|---:|
| **Short decode (107 tok, paired A/B median)** | 27.00 tok/s *(37.04 ms/tok)* | **35.84 tok/s** *(27.90 ms/tok; cold peak 36.62)* | **+32.7%** *(-9.14 ms/tok)* |
| **28K long decode (27,998 tok, paired A/B)** | 20.85 tok/s *(47.96 ms/tok)* | **26.26 – 27.44 tok/s** *(36.44–38.08 ms/tok)* | **+25.9% to +31.6%** *(-9.9 to -11.5 ms/tok)* |
| **28K prompt prefill (`-b 512 -ub 256`)** | 242.6 tok/s *(115.4 s)* | **271.5 – 281.0 tok/s** *(99.6–103.1 s)* | **+11.9% to +15.8%** *(-12.3 to -15.8 s)* |
| **Vulkan dispatches / decode token** | 7,145 | **1,881** | **-73.7%** *(-5,264 dispatches/tok)* |
| **GPU kernel time / short token** | 39.89 ms | **26.46 ms** | **-33.7%** *(-13.43 ms/tok)* |
| **Short greedy determinism (6/6 repeats)** | SHA `9d2af481fd1eb5cc` | **SHA `9d2af481fd1eb5cc`** | **100% bit-exact** |

### 2. Optimization Progression Across Engineering Waves

Each commit in [`patches/`](patches/) was benchmarked incrementally so you can see where every millisecond came from:

| Stage (Git Patch) | Dispatches / Tok | GPU Kernel Time | Short Decode (107 tok) | 28K Decode (28K tok) | Key Additions |
|---|---:|---:|---:|---:|---|
| **Baseline** (`llama.cpp` `b12628`) | 7,145 | 39.89 ms | 27.00–28.56 tok/s | 20.85–21.20 tok/s | Unmodified Vulkan backend |
| **Wave 1** (`0001`, `ff2bbbf`) | 3,659 *(-48.8%)* | ~31.0 ms | 32.57 tok/s *(peak 33.38)* | — | Graph cleanup, AMD `Q5_0` MMVQ threshold, fused `TOPK_MOE`, `L2_CLIP`, `SITU`, `CONTROL_APPLY`, `REL_BIAS_MASK`, `SSM_CONV_RES_SCALE` |
| **Wave 2** (`0002`, `dd903dd`) | 2,417 *(-66.2%)* | 28.60 ms | 34.10 tok/s *(peak 34.76)* | 25.44 tok/s | `DEPTH_MIX_CUR+BANK`, `CONTROL_DOWN_Q8_0`, `CONTROL_UP_APPLY_Q8_0`, `REL_BIAS_MASK_DIRECT`, `SIGMOID_MUL`, initial `SSM_CONV_DECODE` (25/192) |
| **Wave 3** (`0003`, `ea45100`) | **1,881 *(-73.7%)*** | **26.46 ms** | **35.84–36.13 tok/s *(peak 36.62)*** | **26.26–27.44 tok/s** | Register-cached 512-thread `DEPTH_MIX`, coalesced 256-thread `CONTROL_DOWN_Q8_0`, **192/192** `SSM_CONV_DECODE` + `DYN`, AMD RDNA3/4 long-KV FA `split_k = 4` |

### 3. Where Every Millisecond Goes (GPU Roofline Analytics)

Measured with `GGML_VK_PERF_LOGGER=1` on short decode (final build `ea45100`):

| Category | GPU Time / Token | Share of 26.46 ms | Notes |
|---|---:|---:|---|
| **Large matrix multiplies** (`Q4_K` MoE gate/up `5.66 ms`, `Q5_0` MoE down `3.82 ms`, `Q8_0` QKV/O `6.96 ms`, `lm_head` `2.15 ms`) | **18.59 ms** | **70.3%** | Hardware memory-bandwidth bound at **190–223 GB/s** (hardware LPDDR5X ceiling ~229 GB/s) |
| **Fused Inkling custom ops** (`192× SSM_CONV_DECODE+DYN` `0.55 ms`, `120× DEPTH_MIX` `0.85 ms`, `192× CONTROL_DOWN/UP` `1.11 ms`, `48× REL_BIAS_MASK_DIRECT` `0.20 ms`, `144× L2_CLIP` `0.30 ms`, `48× SITU` `0.13 ms`, `48× SIGMOID_MUL` `0.16 ms`, `48× TOPK_MOE` `0.27 ms`) | **3.57 ms** | **13.5%** | Cut from **>16 ms** across ~5,500 unfused ops in stock `b12628` |
| **Flash Attention + norms + router + KV writes + small projections** | **4.30 ms** | **16.2%** | Short-context FA is `0.83 ms`; at 28K context, global FA scales with KV length and benefits from `split_k = 4` (+0.84 ms/tok) |
| **Total GPU time / token** | **26.46 ms** | **100%** | Uninstrumented wall-clock is **27.3–27.9 ms/tok** (<1 ms CPU dispatch overhead) |

### 4. End-to-End Accuracy & Quality Verification

Run with `temperature=0`, `reasoning_effort=low`, full output generation (`max_tokens=1600`):

| Task | Context Depth | Decode Speed | Verified Result |
|---|---:|---:|---|
| Python coding (`merge_intervals` 7-case AST + unit test suite) | 107 tokens | **35.62 tok/s** | **7/7 tests PASS** |
| Code review (strict `PASS`/`FAIL` verdict) | 107 tokens | **35.29 tok/s** | **`PASS`** |
| Python coding (`merge_intervals` 7-case AST + unit test suite) | 27,998 tokens | **27.19 tok/s** | **7/7 tests PASS** |
| Code review (strict `PASS`/`FAIL` verdict) | 27,998 tokens | **27.15 tok/s** | **`PASS`** |

### 5. Context Sizing, Batching & VRAM Analytics (Universal Server Config)

| Server Configuration (FP16 KV, 2 Parallel Slots) | Total VRAM | Compute VRAM | Host Spill | 28K Prefill | 28K Decode | Usable Context / Slot |
|---|---:|---:|---:|---:|---:|---:|
| `2×32K`, `-b 512 -ub 256` *(stock flags)* | 19.5 GiB | 1.78 GiB | 84 MiB | **281.0 tok/s** | **27.44 tok/s** | ~18.5K before compact |
| `2×32K`, `-b 1024 -ub 512` | 21.3 GiB | 3.47 GiB | 169 MiB | 281.0 tok/s | 27.18 tok/s | No speed gain, 2× compute VRAM |
| `2×32K`, `-b 2048 -ub 1024` | 20.2 GiB | 2.19 GiB | **6,581 MiB** | **181.7 tok/s** | 25.20 tok/s | **Regression**: overflows 4 GiB Vulkan buffer limit onto host RAM |
| **`2×64K`, `-b 512 -ub 256` *(recommended on 32 GiB+)*** | **23.1 GiB** *(8.1 GiB free)* | **3.32 GiB** | **164 MiB** | **251.8 tok/s** | **27.09 tok/s** | **~50K before compact (~2.7× fewer compactions)** |

Why `2×64K` costs only **+2.0 GiB** of KV cache: only **8 of Inkling's 48 executed layers** use full global attention; the other **40 layers** use 1,024-token sliding windows (`400 MiB` constant regardless of context length).

### 6. Compaction Speed Analytics (~21K-Token Coding Session)

| Compaction Strategy | Wall-Clock Time | KV Cache Hit | Prefill Time | Decode Time | Factual Fidelity |
|---|---:|---:|---:|---:|---|
| **Baseline** (stock `b12628`, cold transcript, `maxLength` GBNF) | 96.35 s | 0% *(0 / 20,742)* | 79.03 s | 17.24 s *(21.5 tok/s)* | 4/4 facts, 0 issues |
| **Cold Standalone Optimized** (patched server, `reasoning_effort=none` + `reasoning_budget_tokens=32`, JS length clamp) | 84.79 s *(-12.0%)* | 0% *(0 / 20,740)* | 71.74 s | 12.91 s *(27.6 tok/s)* | 4/4 facts, 0 issues |
| **KV-Preserving Optimized** (patched server + exact live prefix + `tool_choice="none"` + `reasoning_budget_tokens=32`) | **15.88 s *(6.07× faster, -83.5%)*** | **99.4% *(20,985 / 21,121)*** | **0.91 s *(86.8× faster)*** | **14.85 s *(27.9 tok/s)*** | **4/4 facts, 0 issues** |

---

## Plain-English Summary: What We Changed & Why It Worked

### Why Raw Token Speed Was Slow

Inkling-2-Mini is a looped Mixture-of-Experts architecture with several bespoke per-layer operations: causal short convolutions (176 static + 16 with dynamic per-head taps), cross-loop controller gates, depth-mixing attention over earlier blocks, relative-position attention biases, `L2` head clipping, and `SiTU` activations.

Stock `llama.cpp` had no dedicated Vulkan shaders for those Inkling-specific blocks. Generating **one single token** forced the CPU to dispatch **7,145 tiny GPU jobs**, constantly writing intermediate arrays to RAM and immediately reading them back.

Our patchset writes **10 custom fused Vulkan GLSL shaders** plus targeted graph-optimizer and AMD RDNA3/4 fixes:

- **`ssm_conv_decode.comp` (192/192 convolutions fused)**: Collapses `CONCAT + VIEW + CPY(state) + SSM_CONV [+ dynamic taps] + RESHAPE + ADD + SCALE` into 1 GPU pass per convolution. Pre-expanding external-`view_src` metadata views (`cell` and `dynamic` taps) in `llama-graph.cpp` and `inkling2.cpp` unlocked 100% fusion (`25/192 → 192/192`), eliminating 460+ dispatches/token alone.
- **`depth_mix.comp` (120/120 depth mixers fused)**: Rewritten as a 512-thread register-cached shader with **one single workgroup barrier**, reading `cur` and `states` once instead of 2–3× (`15.1 µs → 7.1 µs` per call).
- **`control_down_q8_0.comp` & `control_up_apply_q8_0.comp` (192/192 loop gates fused)**: Coalesced 256-thread (8 lanes per `Q8_0` block) down-projection fused with dual RMS-norm + concat (`8.15 µs → 6.45 µs`), plus paired `Q8_0` up-projection fused with `tanh` gating (`4.8 µs`).
- **`rel_bias_mask_direct.comp` (48/48 attention masks fused)**: Replaces `PAD + MUL_MAT(f32, m=1025) + CONT + GET_ROWS + ADD + CAST(f16)` with one direct FP16 shader.
- **Additional fusions & AMD tuning**: `l2_clip.comp`, `situ.comp`, `control_apply.comp`, `rel_bias_mask.comp`, `ssm_conv.comp` (`RES_SCALE`), unlocked `SIGMOID_MUL` (48×), additive-eps `TOPK_MOE_SIGMOID_NORM_BIAS` (48×), AMD `Q5_0` MMVQ threshold (`k >= 1024`), and AMD RDNA3/4 long-KV Flash Attention `split_k = 4` (`N == 1 && KV >= 4096`).
- **Critical graph-optimizer catches**: Protected interior real *and* empty (`VIEW`/`RESHAPE`) nodes in `ggml_vk_graph_optimize`, and cleared `do_add_rms_partials` whenever `DEPTH_MIX_CUR`/`CONTROL_DOWN_Q8_0` intercept `RMS_NORM` (preventing stale partial sums from corrupting downstream norms).

### Four Universal Inkling-2-Mini Compaction Rules (Any Agent / Harness)

Whether you build your own agent loop or configure an existing coding tool:

1. **Reuse the live KV cache instead of starting a cold transcript prompt**: Append a single short user turn (`"Summarize the conversation above into the required JSON checkpoint."`) onto the exact `{ messages, tools, chat_template_kwargs }` already in `llama-server`'s slot, with `"tool_choice": "none"` and `"cache_prompt": true`. Prefill drops from **79 s → 0.9 s**.
2. **Never change `reasoning_effort` on a cached prefix**: Inkling's Jinja template emits `Thinking effort level: 0.2` right before the first user turn. Switching it to `"none"` alters token ~300 and busts 99% of the KV cache.
3. **Always set `"reasoning_budget_tokens": 32`**:
   - Setting `0` while `reasoning_effort > 0` triggers an infinite empty `<|content_thinking|><|end_message|>` loop.
   - Omitting `reasoning_budget_tokens` even with `reasoning_effort: "none"` still allows Inkling to think for 1,600+ tokens on dense transcripts (`reasoning_effort: "none"` is only a soft prompt hint). `32` caps thinking cleanly in both KV-preserving and cold standalone modes.
4. **Omit `maxLength` from JSON Schema `response_format`**: Character-length constraints compile huge GBNF automata over Inkling's 201K vocabulary; enforce sentence length limits in client code instead.

---

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── patches/
│   ├── inkling2-vulkan-opt-b12628-combined.patch                      # Single combined patch (`git apply`)
│   ├── 0001-Inkling-Vulkan-opt-Wave-A1-A2a-graph-cleanup-Q5_0-MM.patch
│   ├── 0002-Inkling-Vulkan-opt-Wave-A2e-DEPTH_MIX_CUR-BANK-CONTR.patch
│   └── 0003-Inkling-Vulkan-opt-Wave-A3b-reg-cached-512-thread-DE.patch
├── scripts/
│   ├── build-vulkan-windows.example.cmd                               # Windows MSVC + Ninja + Vulkan SDK template
│   └── Start-Inkling-opt-64k.example.ps1                              # Parameterized 2×64K FP16 server launcher
├── compaction/
│   ├── README.md                                                      # Universal HTTP recipe + optional adapter setup
│   ├── checkpoint.mjs                                                 # Zero-dependency vanilla JS compaction engine
│   └── examples/
│       ├── pi/index.ts                                                # Optional Pi extension reference adapter
│       └── opencode/index.js                                          # Optional OpenCode plugin reference adapter
└── benchmarks/
    ├── METHODOLOGY.md                                                 # Hardware, GGUF SHA-256, flags, and repro steps
    ├── summary.json                                                   # Complete structured benchmark & quality data
    └── bench_ab.py                                                    # Parameterized A/B benchmark script
```

---

## Quick Start: Apply, Build & Run

### 1. Apply the Vulkan patches to `llama.cpp` `b12628`

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b12628   # commit 8a97fa98a
git apply /path/to/patches/inkling2-vulkan-opt-b12628-combined.patch
```

*(Or use `git am /path/to/patches/000*.patch` to preserve all 3 wave commits.)*

### 2. Build `llama-server` with Vulkan

On Windows (MSVC + Ninja + Vulkan SDK with `glslc`): see [`scripts/build-vulkan-windows.example.cmd`](scripts/build-vulkan-windows.example.cmd).

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server --parallel
```

### 3. Launch `llama-server` (Works With Any OpenAI-Compatible Client)

```bash
LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"low"}' \
./build/bin/llama-server \
  --model /path/to/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj /path/to/mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  --no-mmproj-offload \
  --alias inkling-2-mini-uncensored \
  --host 127.0.0.1 --port 18089 \
  --ctx-size 131072 --parallel 2 --no-kv-unified \
  --batch-size 512 --ubatch-size 256 \
  --device Vulkan0 --n-gpu-layers all \
  --jinja --cache-ram 0 \
  -ctk f16 -ctv f16 \
  --spec-type none --sleep-idle-seconds 600 --no-webui
```

*(For 24 GiB VRAM systems, use `--ctx-size 65536 --parallel 2` for `2×32K` slots at ~19.5 GiB total.)*

### 4. Optional: Adopt KV-Preserving Compaction in Any Client

See [`compaction/README.md`](compaction/README.md) for the universal HTTP JSON payload recipe, the zero-dependency `checkpoint.mjs` engine, and optional reference adapters for Pi and OpenCode.

---

## Reproducing Benchmarks

See [`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md) and [`benchmarks/bench_ab.py`](benchmarks/bench_ab.py) to run paired alternating A/B comparisons on your own hardware.

## License

MIT License — matches upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
