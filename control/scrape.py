"""Poll /metrics on every worker and append one JSON row per worker per scrape.

    python -m control.scrape --run rehearsal --duration 1800
    python -m control.scrape --run test --out-dir metrics/rehearsal --interval 5

Runs until --duration seconds pass or you press Ctrl-C.
Writes <out-dir>/scrape_<run>.jsonl.

Field names map to the real vLLM metric names listed in
metrics/metric_names.txt. A field whose name is not in that file is
written as null, with a warning at start-up.

Counters (prefix_*, preemptions, request_success, *_sum, *_count) are
cumulative since the engine started. Take differences between rows for rates.
kv_usage is a fraction, 0 to 1, as vLLM reports it.
"""

import argparse
import json
import re
import signal
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKERS = ROOT / "cluster" / "workers.json"
DEFAULT_NAMES = ROOT / "metrics" / "metric_names.txt"

# field -> vLLM sample name (must appear in metric_names.txt)
FIELDS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "request_success": "vllm:request_success_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "ttft_sum": "vllm:time_to_first_token_seconds_sum",
    "ttft_count": "vllm:time_to_first_token_seconds_count",
    "queue_time_sum": "vllm:request_queue_time_seconds_sum",
    "queue_time_count": "vllm:request_queue_time_seconds_count",
    "e2e_sum": "vllm:e2e_request_latency_seconds_sum",
    "e2e_count": "vllm:e2e_request_latency_seconds_count",
}
# also kept per label value, so aborted requests are visible
BY_LABEL = {"request_success": "finished_reason"}

_SAMPLE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)')
_LABEL = re.compile(r'(\w+)="([^"]*)"')


def resolve_fields(names_file: Path) -> dict[str, str | None]:
    """Keep each field's metric name only if metric_names.txt lists it."""
    present = {line.split()[0] for line in names_file.read_text().splitlines() if line.strip()}
    resolved = {}
    for field, name in FIELDS.items():
        resolved[field] = name if name in present else None
        if resolved[field] is None:
            print(f"warning: {name} is not in {names_file}; {field} will be null")
    return resolved


def parse(text: str) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    """Return (totals summed over label sets, per-name {label: {value: sum}})."""
    totals: dict[str, float] = defaultdict(float)
    by_label: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        totals[name] += v
        for key, val in _LABEL.findall(labels or ""):
            by_label[name][key][val] += v
    return totals, by_label


def scrape_once(worker: str, url: str, fields: dict[str, str | None], run: str) -> dict:
    row = {"ts": time.time(), "run": run, "worker": worker, "ok": False, "error": None}
    try:
        r = httpx.get(f"{url.rstrip('/')}/metrics", timeout=4)
        r.raise_for_status()
        totals, by_label = parse(r.text)
        row["ok"] = True
        for field, name in fields.items():
            row[field] = totals.get(name) if name else None
        for field, label in BY_LABEL.items():
            name = fields.get(field)
            row[f"{field}_by_{label}"] = dict(by_label[name][label]) if name else None
    except (httpx.HTTPError, ValueError) as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        for field in fields:
            row[field] = None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=Path, default=DEFAULT_WORKERS, help="JSON {name: url}")
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES, help="metric_names.txt")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between scrapes")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"), help="run name in the file name")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "metrics")
    args = parser.parse_args()

    workers = json.loads(args.workers.read_text())
    fields = resolve_fields(args.names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"scrape_{args.run}.jsonl"
    print(f"scraping {', '.join(workers)} every {args.interval:g} s -> {out}")

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    start = time.monotonic()
    n = 0
    try:
        with out.open("a") as f:
            while not stop["now"]:
                for name, url in workers.items():
                    row = scrape_once(name, url, fields, args.run)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    if not row["ok"]:
                        print(f"{name}: {row['error']}")
                n += 1
                if args.duration is not None and time.monotonic() - start >= args.duration:
                    break
                time.sleep(max(0.0, start + n * args.interval - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"stopped after {n} scrapes; wrote {out}")


if __name__ == "__main__":
    main()
