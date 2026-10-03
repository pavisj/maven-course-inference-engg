"""Gateway queue: one per worker.

At most `max_in_flight` requests are forwarded to a worker at the same time (32 = the
engine's decode slots). Request 33 waits here, not in vLLM's waiting queue. Order:
interactive before batch, then by arrival time.

A waiting request leaves the queue in one of three ways:
- granted:  a seat frees (release()) and it is next in order
- timeout:  its deadline from arrival passes (interactive 6 s, batch 60 s) -> 429 timeout_queue
- gone:     its client disconnects -> dropped, counted as an abort at stage "queued"
"""

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field

PRIORITY = {"interactive": 0, "batch": 1}                 # lower runs first
QUEUE_DEADLINE_S = {"interactive": 6.0, "batch": 60.0}    # from arrival (Step 6; same as admit's deadlines)
POLL_S = 0.25                                            # how often a waiting request checks for a disconnect
DEFAULT_MAX_IN_FLIGHT = 32                               # --max-num-seqs (Part 1)


@dataclass(eq=False)
class Ticket:
    tenant: str
    arrival: float
    deadline: float
    priority: int
    seq: int
    state: str = "waiting"                 # waiting | granted | cancelled
    granted_at: float | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


class WorkerQueue:
    def __init__(self, name: str, max_in_flight: int = DEFAULT_MAX_IN_FLIGHT):
        self.name = name
        self.max_in_flight = max_in_flight
        self.in_flight = 0
        self._heap: list = []
        self._waiting = 0
        self._seq = itertools.count()

    def depth(self) -> int:
        """Requests waiting in this queue (not yet forwarded)."""
        return self._waiting

    def enqueue(self, tenant: str, arrival: float, deadline_s: float | None = None) -> Ticket:
        deadline = arrival + (QUEUE_DEADLINE_S[tenant] if deadline_s is None else deadline_s)
        t = Ticket(tenant, arrival, deadline, PRIORITY[tenant], next(self._seq))
        if self.in_flight < self.max_in_flight and self._waiting == 0:
            self._grant(t)
        else:
            heapq.heappush(self._heap, (t.priority, t.arrival, t.seq, t))
            self._waiting += 1
        return t

    def _grant(self, t: Ticket) -> None:
        t.state = "granted"
        t.granted_at = time.time()
        self.in_flight += 1
        t.event.set()

    def _grant_next(self) -> None:
        while self.in_flight < self.max_in_flight and self._heap:
            *_, t = heapq.heappop(self._heap)
            if t.state == "waiting":
                self._waiting -= 1
                self._grant(t)

    def release(self) -> None:
        """A forwarded request finished: free its seat and grant the next one."""
        self.in_flight -= 1
        self._grant_next()

    def cancel(self, t: Ticket) -> None:
        if t.state == "waiting":
            t.state = "cancelled"
            self._waiting -= 1

    async def wait(self, t: Ticket, is_disconnected) -> str:
        """Wait for a seat. Returns "granted", "timeout" or "gone"."""
        while True:
            if t.state == "granted":
                return "granted"
            remaining = t.deadline - time.time()
            if remaining <= 0:
                self.cancel(t)
                return "timeout"
            try:
                await asyncio.wait_for(t.event.wait(), timeout=min(POLL_S, remaining))
            except asyncio.TimeoutError:
                pass
            if t.state == "granted":
                return "granted"
            if await is_disconnected():
                self.cancel(t)
                return "gone"
