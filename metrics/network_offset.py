"""Measure the laptop -> worker network trip that gateway-measured TTFT includes.

    python metrics/network_offset.py

20 GET /health round trips from this laptop straight to each worker (not through the
gateway, so no gateway time is in them). The gateway runs on this laptop, so this is the
same trip a forwarded request makes. Writes metrics/network_offset.json with the median.
"""

import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
N = 20


def main():
    workers = json.loads((ROOT / "cluster" / "workers.json").read_text())
    out = {"n": N, "ts": time.time(), "workers": {}}
    with httpx.Client(timeout=30) as client:
        for name, url in workers.items():
            client.get(f"{url}/health")                      # open the connection first (TLS handshake)
            trips = []
            for _ in range(N):
                t0 = time.perf_counter()
                r = client.get(f"{url}/health")
                trips.append(time.perf_counter() - t0)
                assert r.status_code == 200, f"{name} /health {r.status_code}"
            out["workers"][name] = {"median_ms": round(statistics.median(trips) * 1000, 1),
                                    "min_ms": round(min(trips) * 1000, 1), "max_ms": round(max(trips) * 1000, 1)}
            print(name, out["workers"][name])
    out["offset_s"] = round(statistics.median([w["median_ms"] for w in out["workers"].values()]) / 1000, 4)
    (ROOT / "metrics" / "network_offset.json").write_text(json.dumps(out, indent=2) + "\n")
    print("offset_s", out["offset_s"])


if __name__ == "__main__":
    main()
