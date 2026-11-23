# Benchmark Methodology & Reproduction Guide

## Hardware & Environment

- **Device**: ASUS ROG Flow Z13 (2025) on AC power
- **APU**: AMD Ryzen AI Max+ 395 (`Strix Halo`), 16 cores / 32 threads
- **GPU**: AMD Radeon 8060S (`gfx1151`, RDNA 3.5, 40 Compute Units, wave64, `KHR_coopmat`, 32 KB LDS)
- **Memory**: 64 GB unified LPDDR5X (~229 GB/s practical peak), 32,843 MiB Vulkan heap (~31,200 MiB free)
- **OS & Backend**: Windows 11 x64, AMD proprietary Vulkan driver, Vulkan SDK 1.4.363.1 (`glslc`)
- **Base `llama.cpp` revision**: `b12628` (commit `8a97fa98a`)

## Model Under Test

- **GGUF**: `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf`
  - **SHA-256**: `36e99f1f56ca33cb356f19d7e3822938f65af1cde978ad13170238928d4c955f`
  - **Architecture**: `inkling2_loop` (26.07B MoE, 8/128 experts, 24 physical layers looping blocks 6–17 × 3 = 48 executed layers; 40 SWA layers with window 1,024 + 8 global layers)
- **Vision projector**: `mmproj-Inkling-2-Mini-Uncensored-F16.gguf` (`--no-mmproj-offload`)

## Fixed Production Flags (Track A A/B)

Both stock `b12628` and the patched build were tested with identical flags:

```text
--ctx-size 65536 --parallel 2 --no-kv-unified
--batch-size 512 --ubatch-size 256
--device Vulkan0 --n-gpu-layers all --no-mmproj-offload
--jinja --cache-ram 0 --spec-type none --no-webui
-ctk f16 -ctv f16
LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"low"}'
```

## Protocol

1. **Single-server isolation**: Only one loaded `llama-server` process at a time.
2. **Paired alternating A/B**: Runs alternate `control → optimized → control → optimized` on a disposable port (`18119`) to neutralize thermal/PPT drift.
3. **Greedy determinism**: `temperature: 0`, `cache_prompt: true` (prime prompt once, then measure cached decode repeats), comparing output token SHA-256 hashes.
4. **End-to-end quality verification**: Full generation on both short (~107 tokens) and long (~27,998 tokens) coding (`merge_intervals` 7-case AST + unit test suite) and code-review (`PASS`/`FAIL` verdict) tasks.
5. **Per-node GPU profiling**: `GGML_VK_PERF_LOGGER=1` on 10 decode tokens to count exact Vulkan dispatches and per-op GPU microseconds.

## Running Your Own A/B Comparison

```bash
python benchmarks/bench_ab.py \
  --control /path/to/stock/llama-server \
  --optimized /path/to/patched/llama-server \
  --model /path/to/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj /path/to/mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  --repeats 3 --tokens 64
```
