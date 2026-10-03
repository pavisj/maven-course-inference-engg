"""Live view of the cluster for admission: engine state from /metrics plus gateway counters.

SnapshotSource refreshes each worker's /metrics every REFRESH_S seconds (it reuses
control/scrape.py parsing) and keeps the gateway's own counters in memory.
current() returns a Snapshot: the engine part is up to REFRESH_S old; the
gateway part (queue depth, tenant tokens) is exact at the moment of the call.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from control.scrape import DEFAULT_NAMES, resolve_fields, scrape_once

REFRESH_S = 2.0               # how often the engine part is refreshed
TENANT_WINDOW_S = 60.0        # sliding window for per-tenant tokens
HIT_RATE_WINDOW_S = 30.0      # prefix hit rate = scrape deltas over this window
PROMPT_WINDOW_S = 60.0        # mean prompt tokens = admitted requests over this window
DEFAULT_AVG_UNCACHED = 1200   # before the scrape gives a hit rate (Part 1: 1,173 to 1,920 uncached tokens)


@dataclass(frozen=True)
class WorkerState:
    name: str
    ok: bool
    running: float
    waiting: float
    kv_usage_pct: float
    preemptions: float
    ts: float


@dataclass(frozen=True)
class Snapshot:
    workers: dict[str, WorkerState]
    queue_depth: int                       # requests held in the gateway before forwarding; 0 until Step 5 adds a queue
    tenant_tokens_60s: dict[str, int]      # tokens charged per tenant in the last 60 s
    tenant_events: dict[str, tuple] = field(default_factory=dict)  # (ts, tokens) per tenant, oldest first
    avg_uncached_prompt_tokens: float = DEFAULT_AVG_UNCACHED
    ts: float = 0.0


class TenantWindow:
    """Tokens charged per tenant, over a sliding window."""

    def __init__(self, window_s: float = TENANT_WINDOW_S):
        self.window_s = window_s
        self.events: dict[str, deque] = {}

    def _trim(self, tenant: str, now: float) -> deque:
        q = self.events.setdefault(tenant, deque())
        while q and q[0][0] <= now - self.window_s:
            q.popleft()
        return q

    def charge(self, tenant: str, tokens: int, now: float) -> None:
        self._trim(tenant, now).append((now, tokens))

    def total(self, tenant: str, now: float) -> int:
        return sum(t for _, t in self._trim(tenant, now))

    def snapshot(self, now: float) -> tuple[dict[str, int], dict[str, tuple]]:
        totals, events = {}, {}
        for tenant in list(self.events):
            q = self._trim(tenant, now)
            totals[tenant] = sum(t for _, t in q)
            events[tenant] = tuple(q)
        return totals, events


class SnapshotSource:
    """Holds the latest engine state and the gateway counters; builds Snapshots."""

    def __init__(self, workers: dict[str, str], primary: str = "WorkerA", names_file: Path = DEFAULT_NAMES):
        self.workers = workers
        self.primary = primary
        self.fields = resolve_fields(names_file)
        self.state: dict[str, WorkerState] = {}
        self.tenants = TenantWindow()
        self.queue_depth = 0          # no gateway queue in Step 4
        self.in_flight = 0            # forwarded, not finished (for /metrics only)
        self._prefix = deque()        # (ts, prefix_queries, prefix_hits) of the primary worker
        self._prompts = deque()       # (ts, prompt_tokens) of admitted requests
        self.last_refresh = 0.0

    def refresh(self) -> None:
        """Scrape every worker once. Blocking: run it in a thread."""
        now = time.time()
        for name, url in self.workers.items():
            row = scrape_once(name, url, self.fields, run="gateway")
            if not row["ok"]:
                old = self.state.get(name)
                self.state[name] = WorkerState(name, False, *(
                    (old.running, old.waiting, old.kv_usage_pct, old.preemptions) if old else (0, 0, 0, 0)), now)
                continue
            self.state[name] = WorkerState(
                name, True, row["running"] or 0.0, row["waiting"] or 0.0,
                (row["kv_usage"] or 0.0) * 100, row["preemptions"] or 0.0, now)
            if name == self.primary and row["prefix_queries"] is not None:
                self._prefix.append((now, row["prefix_queries"], row["prefix_hits"]))
                while len(self._prefix) > 2 and self._prefix[0][0] < now - HIT_RATE_WINDOW_S:
                    self._prefix.popleft()
        self.last_refresh = now

    async def run(self) -> None:
        """Background task: refresh every REFRESH_S seconds until cancelled."""
        while True:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(REFRESH_S)

    def record_prompt(self, prompt_tokens: int, now: float) -> None:
        self._prompts.append((now, prompt_tokens))
        while self._prompts and self._prompts[0][0] < now - PROMPT_WINDOW_S:
            self._prompts.popleft()

    def hit_rate(self) -> float | None:
        if len(self._prefix) < 2:
            return None
        (_, q0, h0), (_, q1, h1) = self._prefix[0], self._prefix[-1]
        return (h1 - h0) / (q1 - q0) if q1 > q0 else None

    def avg_uncached(self, now: float) -> float:
        """Mean prompt tokens of recent admitted requests x (1 - prefix hit rate); else the default."""
        hit = self.hit_rate()
        prompts = [p for t, p in self._prompts if t >= now - PROMPT_WINDOW_S]
        if hit is None or not prompts:
            return float(DEFAULT_AVG_UNCACHED)
        return sum(prompts) / len(prompts) * (1 - hit)

    def current(self, now: float | None = None) -> Snapshot:
        now = time.time() if now is None else now
        totals, events = self.tenants.snapshot(now)
        return Snapshot(
            workers=dict(self.state),
            queue_depth=self.queue_depth,
            tenant_tokens_60s=totals,
            tenant_events=events,
            avg_uncached_prompt_tokens=self.avg_uncached(now),
            ts=now,
        )
