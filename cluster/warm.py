"""Warm-up probes: is a worker warm, not just healthy?

    python cluster/warm.py                                   # both workers in cluster/workers.json
    python cluster/warm.py --out metrics/warm_proof.json --label after_cold_start

Sends PROBES (5 requests, one per chapter group) to each worker, one at a time. Each probe
uses the real tutor system prompt and a real retrieved chunk set. A "Probe <run id>." line at
the start of the user message keeps the chunk prefix uncached on every run (the system prompt
can still be reused, as in real traffic), so a second run cannot look warm only because of
the prefix cache.

Engine TTFT per probe = change in vllm:time_to_first_token_seconds_sum / _count around it.
A worker is warm when its last probe's engine TTFT is within 20 % of baseline A's engine
mean (0.15 s), i.e. at most WARM_TTFT_S. The gateway imports probe_worker() and runs it at
start and after a worker restarts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.prompt import build_messages  # noqa: E402
from app.retrieve import retrieve  # noqa: E402
from control.scrape import parse  # noqa: E402

PROBES = [                                   # (chapter group, question): one per group, both workers' homes
    ("00", "What is the difference between TTFT and TPOT?"),
    ("03", "How does PagedAttention store the KV cache in blocks?"),
    ("05", "What does an inference engine do in one scheduler step?"),
    ("07", "How does Little's Law apply to admission control?"),
    ("10", "What is a cold start and how do you warm a replica?"),
]
BASELINE_ENGINE_TTFT_S = 0.15                # baseline A engine TTFT mean, 153 ms (metrics/baseline_a)
TOLERANCE = 0.20                             # warm = within 20 % of the baseline
WARM_TTFT_S = BASELINE_ENGINE_TTFT_S * (1 + TOLERANCE)
MAX_TOKENS = 8                               # probes only need the first token


def _ttft_totals(client: httpx.Client, url: str) -> tuple[float, float]:
    totals, _ = parse(client.get(f"{url}/metrics", timeout=10).text)
    return totals.get("vllm:time_to_first_token_seconds_sum", 0.0), totals.get("vllm:time_to_first_token_seconds_count", 0.0)


def probe_worker(url: str, run_id: str | None = None) -> dict:
    """Run the 5 probes against one worker. Blocking."""
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    url = url.rstrip("/")
    results = []
    with httpx.Client(timeout=120) as client:
        for chapter, question in PROBES:
            messages = build_messages(question, retrieve(question, chapter=chapter))
            messages[1]["content"] = f"Probe {run_id}.\n" + messages[1]["content"]
            payload = {"model": "tutor", "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}
            s0, c0 = _ttft_totals(client, url)
            start = time.perf_counter()
            laptop_ttft, prompt_tokens, status = None, None, None
            with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
                status = r.status_code
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        prompt_tokens = chunk["usage"].get("prompt_tokens")
                    if laptop_ttft is None and any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                        laptop_ttft = time.perf_counter() - start
            s1, c1 = _ttft_totals(client, url)
            engine = (s1 - s0) / (c1 - c0) if c1 > c0 else None
            results.append({"chapter": chapter, "status": status, "prompt_tokens": prompt_tokens,
                            "engine_ttft_s": round(engine, 4) if engine is not None else None,
                            "laptop_ttft_s": round(laptop_ttft, 4) if laptop_ttft is not None else None})
    last = results[-1]["engine_ttft_s"]
    return {"run_id": run_id, "probes": results, "last_engine_ttft_s": last,
            "threshold_s": round(WARM_TTFT_S, 3), "warm": last is not None and last <= WARM_TTFT_S}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=Path, default=ROOT / "cluster" / "workers.json")
    parser.add_argument("--out", type=Path, default=None, help="write the results as JSON")
    parser.add_argument("--label", default="", help="stored with the results, e.g. after_cold_start")
    args = parser.parse_args()
    out = {"label": args.label, "ts": time.time(), "workers": {}}
    for name, url in json.loads(args.workers.read_text()).items():
        res = probe_worker(url)
        out["workers"][name] = res
        ttfts = ", ".join(f"{p['engine_ttft_s'] * 1000:.0f}" if p["engine_ttft_s"] is not None else "n/a"
                          for p in res["probes"])
        print(f"{name}: engine TTFT per probe (ms): {ttfts} -> {'WARM' if res['warm'] else 'NOT WARM'} "
              f"(last {res['last_engine_ttft_s']} s, threshold {res['threshold_s']} s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
