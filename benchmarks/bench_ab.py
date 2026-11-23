#!/usr/bin/env python3
"""Minimal reproducible paired A/B benchmark for stock vs. patched llama-server on Inkling-2-Mini."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from urllib.request import Request, urlopen

DEFAULT_PROMPT = (
    "Return only a single Python function `merge_intervals(intervals)` with no prose. "
    "Input is a list of integer `(start, end)` tuples. Return a new sorted list merging "
    "overlapping or endpoint-touching intervals, and raise `ValueError` if `start > end`."
)


def http_json(url: str, path: str, body=None, timeout: int = 300):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(proc: subprocess.Popen, url: str, timeout_s: int = 180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            if http_json(url, "/health", timeout=3).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become ready in time")


def run_arm(label: str, server: Path, args, prompt: str):
    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"] = '{"reasoning_effort":"low"}'
    cmd = [
        str(server),
        "--model", str(args.model),
        "--alias", args.alias,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_per_slot * args.parallel),
        "--parallel", str(args.parallel),
        "--no-kv-unified",
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--device", args.device,
        "--n-gpu-layers", "all",
        "--jinja",
        "--cache-ram", "0",
        "--spec-type", "none",
        "--no-webui",
        "-ctk", "f16",
        "-ctv", "f16",
    ]
    if args.mmproj:
        cmd += ["--mmproj", str(args.mmproj), "--no-mmproj-offload"]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        wait_ready(proc, url)
        # Prime slot 0
        prime = http_json(url, "/v1/chat/completions", {
            "model": args.alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1,
            "cache_prompt": True,
        }, timeout=420)
        pt = prime.get("timings", {})
        print(f"[{label}] prime: {pt.get('prompt_n', 0)} tok @ {pt.get('prompt_per_second', 0):.1f} tok/s", flush=True)

        speeds = []
        hashes = []
        for rep in range(1, args.repeats + 1):
            res = http_json(url, "/v1/chat/completions", {
                "model": args.alias,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.tokens,
                "cache_prompt": True,
            }, timeout=300)
            t = res.get("timings", {})
            tps = float(t.get("predicted_per_second", 0.0))
            msg = res["choices"][0]["message"]
            raw = (msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "")
            sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
            speeds.append(tps)
            hashes.append(sha)
            print(f"[{label}] rep {rep}: {tps:.2f} tok/s ({t.get('predicted_per_token_ms', 0):.2f} ms/tok) sha={sha}", flush=True)
        med = statistics.median(speeds)
        print(f"[{label}] MEDIAN: {med:.2f} tok/s\n", flush=True)
        return {"label": label, "median_tps": med, "speeds": speeds, "hashes": hashes}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True, help="Path to stock llama-server binary")
    ap.add_argument("--optimized", type=Path, required=True, help="Path to patched llama-server binary")
    ap.add_argument("--model", type=Path, required=True, help="Path to Inkling-2-Mini GGUF")
    ap.add_argument("--mmproj", type=Path, default=None, help="Optional path to mmproj GGUF")
    ap.add_argument("--prompt-file", type=Path, default=None, help="Optional text file containing prompt")
    ap.add_argument("--alias", default="inkling-2-mini-uncensored")
    ap.add_argument("--device", default="Vulkan0")
    ap.add_argument("--port", type=int, default=18119)
    ap.add_argument("--ctx-per-slot", type=int, default=32768)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ubatch-size", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2, help="Number of alternating A/B rounds")
    args = ap.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    ctrl_all, opt_all = [], []
    for r in range(1, args.rounds + 1):
        ctrl_all.extend(run_arm(f"control-{r}", args.control, args, prompt)["speeds"])
        opt_all.extend(run_arm(f"optimized-{r}", args.optimized, args, prompt)["speeds"])

    c_med = statistics.median(ctrl_all)
    o_med = statistics.median(opt_all)
    gain = (o_med / c_med - 1.0) * 100.0 if c_med > 0 else 0.0
    print(f"=== SUMMARY ({args.rounds} alternating rounds) ===")
    print(f"Control median:   {c_med:.2f} tok/s")
    print(f"Optimized median: {o_med:.2f} tok/s ({gain:+.1f}%)")


if __name__ == "__main__":
    main()
