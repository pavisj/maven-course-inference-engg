"""Live view of the cluster for admission and placement.

SnapshotSource refreshes each worker's /metrics every REFRESH_S seconds (it reuses
control/scrape.py parsing) and keeps the gateway's own counters in memory:
per-tenant tokens, per-worker queue depth and in-flight count (read from the
gateway queues), per-worker completion rate, and which workers are warm.

current() returns a Snapshot. The engine part is up to REFRESH_S old; the gateway
part (queues, in flight, tenant tokens, completions) is exact at the moment of the call.
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
COMPLETION_WINDOW_S = 30.0    # completion rate = gateway completions per worker over this window
COMPLETION_FLOOR = 0.5        # completions per second, never lower (Step 5a)
COMPLETION_MIN_SPAN_S = 5.0   # right after start, divide by at least this many seconds
DEFAULT_SLOTS = 32            # --max-num-seqs (Part 1); the gateway's in-flight cap per worker


@dataclass(frozen=True)
class WorkerState:
    name: str
    ok: bool
    running: float
    waiting: float
    kv_usage_pct: float
    preemptions: float
    ts: float
    num_gpu_blocks: int | None = None


@dataclass(frozen=True)
class Snapshot:
    workers: dict[str, WorkerState]
    queue_depth: dict[str, int] = field(default_factory=dict)        # gateway queue per worker
    in_flight: dict[str, int] = field(default_factory=dict)          # forwarded, not finished, per worker
    completion_rate: dict[str, float] = field(default_factory=dict)  # completions per second per worker
    warm: dict[str, bool] = field(default_factory=dict)
    slots: int = DEFAULT_SLOTS
    tenant_tokens_60s: dict[str, int] = field(default_factory=dict)
    tenant_events: dict[str, tuple] = field(default_factory=dict)    # (ts, tokens) per tenant, oldest first
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

    def __init__(self, workers: dict[str, str], primary: str = "WorkerA", names_file: Path = DEFAULT_NAMES,
                 slots: int = DEFAULT_SLOTS):
        self.workers = workers
        self.primary = primary
        self.slots = slots
        self.fields = resolve_fields(names_file)
        self.state: dict[str, WorkerState] = {}
        self.tenants = TenantWindow()
        self.queues = {}                       # worker -> control.queue.WorkerQueue, set by the gateway
        self.warm: dict[str, bool] = {w: False for w in workers}
        self.restarted: set[str] = set()       # workers seen down or restarted since they were warm
        self._ttft_count: dict[str, float] = {}
        self._completions = {w: deque() for w in workers}
        self._prefix = deque()                 # (ts, prefix_queries, prefix_hits) summed over workers
        self._prompts = deque()                # (ts, prompt_tokens) of admitted requests
        self.start = time.time()
        self.last_refresh = 0.0

    def refresh(self) -> None:
        """Scrape every worker once. Blocking: run it in a thread."""
        now = time.time()
        queries = hits = 0.0
        all_ok = True
        for name, url in self.workers.items():
            row = scrape_once(name, url, self.fields, run="gateway")
            if not row["ok"]:
                all_ok = False
                if self.warm.get(name):
                    self.restarted.add(name)
                self.warm[name] = False        # a worker we cannot read is not warm
                old = self.state.get(name)
                self.state[name] = WorkerState(name, False, *(
                    (old.running, old.waiting, old.kv_usage_pct, old.preemptions) if old else (0, 0, 0, 0)),
                    now, old.num_gpu_blocks if old else None)
                continue
            count = row.get("ttft_count") or 0.0
            if count < self._ttft_count.get(name, 0.0):   # counters went back: the engine restarted
                self.warm[name] = False
                self.restarted.add(name)
            self._ttft_count[name] = count
            self.state[name] = WorkerState(
                name, True, row["running"] or 0.0, row["waiting"] or 0.0,
                (row["kv_usage"] or 0.0) * 100, row["preemptions"] or 0.0, now, row.get("num_gpu_blocks"))
            queries += row["prefix_queries"] or 0.0
            hits += row["prefix_hits"] or 0.0
        if all_ok:
            self._prefix.append((now, queries, hits))
            while len(self._prefix) > 2 and self._prefix[0][0] < now - HIT_RATE_WINDOW_S:
                self._prefix.popleft()
        self.last_refresh = now

    async def run(self, on_refresh=None) -> None:
        """Background task: refresh every REFRESH_S seconds until cancelled."""
        while True:
            await asyncio.to_thread(self.refresh)
            if on_refresh:
                on_refresh()
            await asyncio.sleep(REFRESH_S)

    def record_prompt(self, prompt_tokens: int, now: float) -> None:
        self._prompts.append((now, prompt_tokens))
        while self._prompts and self._prompts[0][0] < now - PROMPT_WINDOW_S:
            self._prompts.popleft()

    def record_completion(self, worker: str, now: float) -> None:
        q = self._completions.setdefault(worker, deque())
        q.append(now)
        while q and q[0] < now - COMPLETION_WINDOW_S:
            q.popleft()

    def completion_rate(self, worker: str, now: float) -> float:
        q = self._completions.get(worker, deque())
        while q and q[0] < now - COMPLETION_WINDOW_S:
            q.popleft()
        span = max(COMPLETION_MIN_SPAN_S, min(COMPLETION_WINDOW_S, now - self.start))
        return max(COMPLETION_FLOOR, len(q) / span)

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
            queue_depth={w: q.depth() for w, q in self.queues.items()},
            in_flight={w: q.in_flight for w, q in self.queues.items()},
            completion_rate={w: self.completion_rate(w, now) for w in self.workers},
            warm=dict(self.warm),
            slots=self.slots,
            tenant_tokens_60s=totals,
            tenant_events=events,
            avg_uncached_prompt_tokens=self.avg_uncached(now),
            ts=now,
        )
