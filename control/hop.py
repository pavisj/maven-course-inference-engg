"""Hop records: a request placed away from its chapter's home worker.

On Modal, the two containers do not share KV memory. A hop means the destination
recomputes the prefix from scratch; nothing is copied. The record says where that
recompute happened: {ts, src, dst, tokens, prefix, backend}.
"""

import json
from collections import Counter
from pathlib import Path

BACKEND = "recompute"            # no KV transfer between Modal containers (DESIGN Part 6)


class HopLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.counts: Counter = Counter()      # (src, dst) -> hops
        self._f = path.open("a")

    def record(self, ts: float, src: str, dst: str, tokens: int, prefix: str | None) -> dict:
        row = {"ts": ts, "src": src, "dst": dst, "tokens": tokens, "prefix": prefix, "backend": BACKEND}
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()
        self.counts[(src, dst)] += 1
        return row

    def close(self) -> None:
        self._f.close()
