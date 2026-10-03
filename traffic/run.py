"""Drive one traffic mix against one worker, closed loop.

    python -m traffic.run --mix unique --target <worker url> --duration 480 --users 48
    python -m traffic.run --mix shared_chapter --chapter 07 --target <url> --duration 30
    python -m traffic.run --mix mixed --target <url> --duration 30 --out-dir metrics/rehearsal

Closed loop: each of --users threads sends its next request as soon as its
previous one finishes. 48 users is the default: more than the engine's 32
decode slots, so requests queue. No new request starts after --duration
seconds; requests in flight then finish. Each request goes through the app
path (retrieve -> prompt -> /v1/chat/completions); there is no gateway yet.
The answer budget follows the tenant (512 interactive, 256 batch) unless
--max-tokens is set.

Appends one row per request to <out-dir>/traffic_<mix>.jsonl:
tenant, mix, prompt tokens, TTFT, output tokens, status, and timings.
"""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

from app.retrieve import retrieve
from app.tutor import MAX_TOKENS, call_tutor
from traffic.mixes import MIXES, shared_chapter

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix", choices=sorted(MIXES), required=True)
    parser.add_argument("--target", required=True, help="worker base URL")
    parser.add_argument("--duration", type=float, default=30, help="seconds to keep starting requests")
    parser.add_argument("--users", "--concurrency", dest="users", type=int, default=48,
                        help="closed-loop users (threads); --concurrency is an alias")
    parser.add_argument("--chapter", default="07", help="chapter label for --mix shared_chapter")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"answer budget for every request; default per tenant {MAX_TOKENS}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "metrics")
    args = parser.parse_args()

    if args.mix == "shared_chapter":
        requests = shared_chapter(chapter=args.chapter, seed=args.seed)
    else:
        requests = MIXES[args.mix](seed=args.seed)
    retrieve("warm up the index")  # build the BM25 index once, before the threads start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"traffic_{args.mix}.jsonl"
    lock = threading.Lock()
    deadline = time.monotonic() + args.duration
    counter = {"n": 0}
    rows = []

    def worker(slot: int):
        with out.open("a") as f:
            while time.monotonic() < deadline:
                with lock:
                    req = next(requests)
                    counter["n"] += 1
                    seq = counter["n"]
                record = call_tutor(req["question"], req["tenant"], args.target,
                                    max_tokens=args.max_tokens, chapter=req["chapter"])
                record.update({"mix": req["mix"], "seq": seq, "slot": slot,
                               "users": args.users, "target": args.target})
                with lock:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    rows.append(record)

    print(f"{args.mix}: {args.users} users for {args.duration:g} s -> {args.target}")
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in rows if r["status"] == 200 and r["ttft_s"] is not None]
    ttfts = [r["ttft_s"] * 1000 for r in ok]
    p50, p95 = percentile(ttfts, 50), percentile(ttfts, 95)
    print(f"{len(rows)} requests, {len(ok)} ok, {len(rows) - len(ok)} failed")
    if ok:
        print(f"TTFT p50 {p50:.0f} ms, p95 {p95:.0f} ms (laptop-measured, includes network)")
        print(f"prompt tokens median {statistics.median(r['prompt_tokens'] or 0 for r in ok):.0f}, "
              f"output tokens median {statistics.median(r['output_tokens'] or 0 for r in ok):.0f}")
    for tenant in sorted({r["tenant"] for r in rows}):
        n = sum(r["tenant"] == tenant for r in rows)
        print(f"  tenant {tenant}: {n} requests")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
