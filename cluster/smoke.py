"""Smoke the engine on both workers, then save a /metrics scrape from each.

Run from the repo root, after write_workers.py:
    python cluster/smoke.py

For each worker:
1. Poll /health until 200 (503 = cold, wait).
2. Send one streamed chat completion (about 400-token system prompt + short question).
   Print time to first token.
3. Save the raw /metrics text to metrics/scrape_<worker>_<timestamp>.txt.

Then write every vllm: metric name to metrics/metric_names.txt and print the
engine numbers the gateway will need later.
"""

import time

from client import (
    METRICS_DIR,
    QUESTION,
    SYSTEM_PROMPT,
    find_metric,
    load_workers,
    parse_metrics,
    scrape,
    stream_chat,
    wait_healthy,
)

# Known names first, newest vLLM first. find_metric falls back to a substring search.
WATCH = {
    "requests running": (["vllm:num_requests_running"], ["running"]),
    "requests waiting": (["vllm:num_requests_waiting"], ["waiting"]),
    "KV cache usage": (["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"], ["cache_usage"]),
    "prefix cache queries": (
        ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
        ["prefix_cache_queries"],
    ),
    "prefix cache hits": (
        ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
        ["prefix_cache_hits"],
    ),
    "preemptions": (["vllm:num_preemptions_total"], ["preempt"]),
}
SUFFIXES = ("_total", "_bucket", "_sum", "_count", "_created")


def metric_types(text: str) -> dict[str, str]:
    """Map each sample name to its Prometheus type, using the # TYPE lines."""
    families = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, family, kind = line.split(" ", 3)
            families[family] = kind
    types = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        family = name
        for suffix in SUFFIXES:
            if family not in families and family.endswith(suffix):
                family = family[: -len(suffix)]
        types[name] = families.get(family, "untyped")
    return types


def main():
    workers = load_workers()
    METRICS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    all_types: dict[str, str] = {}

    for name, url in workers.items():
        print(f"\n=== {name}  {url}")
        waited = wait_healthy(url)
        print(f"healthy after {waited:.1f} s")

        result = stream_chat(url, SYSTEM_PROMPT, QUESTION)
        print(f"TTFT (laptop-measured): {result['ttft_s'] * 1000:.0f} ms")
        print(f"total: {result['total_s']:.2f} s, prompt tokens: {result['prompt_tokens']}, "
              f"completion tokens: {result['completion_tokens']}")
        print(f"answer: {result['text'][:200]}...")

        text = scrape(url)
        out = METRICS_DIR / f"scrape_{name}_{stamp}.txt"
        out.write_text(text)
        print(f"saved {out.relative_to(METRICS_DIR.parent)}")

        all_types.update(metric_types(text))
        totals = parse_metrics(text)
        for label, (candidates, words) in WATCH.items():
            metric, value = find_metric(totals, candidates, words)
            if metric is None:
                print(f"  {label:22} NOT FOUND")
                continue
            if "cache_usage" in metric:
                pct = value * 100 if value <= 1.0 else value  # vLLM reports 0..1
                print(f"  {label:22} {pct:.2f} %   ({metric})")
            else:
                print(f"  {label:22} {value:g}   ({metric})")

    names_file = METRICS_DIR / "metric_names.txt"
    names_file.write_text("".join(f"{n}  {t}\n" for n, t in sorted(all_types.items())))
    print(f"\n{len(all_types)} vllm: metric names (sample name, type):")
    for n, t in sorted(all_types.items()):
        print(f"  {n}  {t}")
    print(f"wrote {names_file.relative_to(METRICS_DIR.parent)}")


if __name__ == "__main__":
    main()
