"""Warm-up probe: send the same request 5 times in a row to each worker.

Run from the repo root, after write_workers.py:
    python cluster/warm_probe.py

Saves metrics/warmup_<worker>.json. Request 1 vs request 5 is the warm-up
proof for DESIGN.md Part 6.

Per request it records three numbers:
- ttft_s: laptop-measured. Includes the network round trip to Modal.
- engine_ttft_s: vLLM's own TTFT for that request, read as the change in
  vllm:time_to_first_token_seconds_sum. Requests run one at a time, so the
  change belongs to exactly one request. No network in it.
- prefix_cache_hits: prefix-cache hit tokens added by that request.
  Requests 2 to 5 reuse request 1's KV blocks, so part of any speed-up is
  the prefix cache, not warm-up. Read this column before you claim "warm".

Request 1 is only "cold" for the engine if nothing ran on this container
before it. smoke.py warms kernels too, so for a true cold number run this
probe first on a freshly started container.

To split warm-up from prefix caching, run it twice:
    python cluster/warm_probe.py               # fresh container: cold engine, cache miss
    python cluster/warm_probe.py --label warm  # same container: warm engine, cache miss
Request 1 of each run is a cache miss (new session id). Cold minus warm is
the warm-up cost. Warm request 1 minus warm request 5 is the prefix-cache saving.
"""

import argparse
import json
import time
from pathlib import Path

import httpx

from client import (
    METRICS_DIR,
    QUESTION,
    SERVED_MODEL_NAME,
    SYSTEM_PROMPT,
    WORKERS_FILE,
    find_metric,
    load_workers,
    parse_metrics,
    scrape,
    stream_chat,
    wait_healthy,
)

N_REQUESTS = 5
TTFT_SUM = (["vllm:time_to_first_token_seconds_sum"], ["time_to_first_token", "_sum"])
PREFIX_HITS = (["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"], ["prefix_cache_hits"])


def engine_counters(url: str) -> tuple[float, float]:
    """Return (TTFT seconds summed over all requests so far, prefix-cache hit tokens so far)."""
    totals = parse_metrics(scrape(url))
    _, ttft_sum = find_metric(totals, *TTFT_SUM)
    _, hits = find_metric(totals, *PREFIX_HITS)
    return ttft_sum or 0.0, hits or 0.0


def model_root(url: str) -> str:
    """Return the HF model id behind the served name, from /v1/models."""
    models = httpx.get(f"{url}/v1/models", timeout=30).json()["data"]
    return next(m.get("root", m["id"]) for m in models if m["id"] == SERVED_MODEL_NAME)


def probe(name: str, url: str) -> dict:
    wait_healthy(url)
    # A fresh run id per probe run makes request 1 a prefix-cache miss,
    # even if smoke.py or an earlier probe already sent this system prompt.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    system_prompt = f"Session {run_id}.\n{SYSTEM_PROMPT}"

    rows = []
    for i in range(1, N_REQUESTS + 1):
        ttft_before, hits_before = engine_counters(url)
        r = stream_chat(url, system_prompt, QUESTION)
        ttft_after, hits_after = engine_counters(url)
        row = {
            "request": i,
            "ttft_s": round(r["ttft_s"], 4),
            "engine_ttft_s": round(ttft_after - ttft_before, 4),
            "prefix_cache_hits": int(hits_after - hits_before),
            "prompt_tokens": r["prompt_tokens"],
        }
        rows.append(row)
        print(f"  req {i}: TTFT {row['ttft_s'] * 1000:6.0f} ms laptop, "
              f"{row['engine_ttft_s'] * 1000:6.1f} ms engine, "
              f"prefix hits {row['prefix_cache_hits']:5d} of {row['prompt_tokens']} prompt tokens")

    return {
        "worker": name,
        "url": url,
        "model": SERVED_MODEL_NAME,
        "model_root": model_root(url),
        "run_id": run_id,
        "ttft_s": [row["ttft_s"] for row in rows],
        "engine_ttft_s": [row["engine_ttft_s"] for row in rows],
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="",
        help="suffix for the output file, e.g. 'warm' writes warmup_<worker>_warm.json. "
        "Use it for a re-run on a warm engine, so the cold-start file is kept.",
    )
    parser.add_argument("--workers", type=Path, default=WORKERS_FILE, help="JSON {name: url}")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR, help="where to write warmup_*.json")
    args = parser.parse_args()
    suffix = f"_{args.label}" if args.label else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in load_workers(args.workers).items():
        print(f"\n=== {name}  {url}")
        result = probe(name, url)
        result["label"] = args.label or "cold"
        out = args.out_dir / f"warmup_{name}{suffix}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        first, last = result["ttft_s"][0], result["ttft_s"][-1]
        print(f"  request 1 vs {N_REQUESTS}: {first * 1000:.0f} ms -> {last * 1000:.0f} ms (laptop)")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
