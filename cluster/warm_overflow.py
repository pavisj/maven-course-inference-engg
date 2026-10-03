"""Pre-warm the Overflow server (it scales from zero; the cold start was 137.7 s).

    python cluster/warm_overflow.py --out metrics/run_i/overflow_prewarm.json --deadline 18:30

Polls /health until 200, then sends one real tutor request. Exits 1 if Overflow is not up
before --deadline (local time HH:MM), so a run can be skipped instead of overflowing to a cold server.
"""

import argparse
import datetime as dt
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", default=None, help="local HH:MM; give up after this time")
    args = parser.parse_args()

    url = json.loads((ROOT / "cluster" / "overflow.json").read_text())["Overflow"]
    stop = None
    if args.deadline:
        h, m = map(int, args.deadline.split(":"))
        stop = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    t0 = time.time()
    codes = []
    while True:
        try:
            code = httpx.get(f"{url}/health", timeout=30).status_code
        except httpx.HTTPError as e:
            code = type(e).__name__
        codes.append(str(code))
        if code == 200:
            break
        if stop and dt.datetime.now() >= stop:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"up": False, "gave_up_at": args.deadline, "waited_s": round(time.time() - t0, 1)}) + "\n")
            sys.exit(f"Overflow not up by {args.deadline}")
        time.sleep(3)
    q = "Why does prefix caching save prefill time?"
    payload = {"model": "tutor", "messages": build_messages(q, retrieve(q)), "max_tokens": 32, "temperature": 0}
    start = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    result = {"up": True, "url": url, "waited_s": round(time.time() - t0, 1), "health_codes_seen": sorted(set(codes)),
              "probe_status": r.status_code, "probe_total_s": round(time.perf_counter() - start, 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
