"""Evict: what the gateway forgets when a worker stops being the worker it knew.

On Modal nothing is copied between workers (a hop recomputes the prefix, Part 6), so there is
no KV to evict. What can go stale is the gateway's own state about a worker: its chapter
affinity (the chapters it is home for) and its warm flag. After a container restart, the new
container has an empty prefix cache and cold kernels; if the gateway still sends that
worker's chapters to it as "home" and treats it as warm, that is the ghost.

Affinity.evict(worker, warm, reason) drops both, once per failure, and counts it
(gateway_affinity_evict_total). Triggers:
- the worker fails its warm probe (gateway, after WARM_ROUNDS rounds), or
- the snapshot cannot read the worker, or its counters go back (a restart).
Affinity.restore(worker) gives the chapters back when the worker passes its warm probe.
Place reads the warm flag, so while a worker is evicted its chapters hop to the other worker.
"""

from control.place import CHAPTER_HOME


class Affinity:
    def __init__(self, homes: dict[str, str] | None = None):
        self.homes = dict(CHAPTER_HOME if homes is None else homes)   # chapter -> home worker
        self.dropped: set[str] = set()                                # workers whose affinity is dropped
        self.evictions = 0
        self.log: list[tuple[str, str]] = []                          # (worker, reason)

    def home(self, chapter: str | None) -> str | None:
        """The chapter's home worker, or None while that worker's affinity is dropped."""
        w = self.homes.get(chapter or "")
        return None if w in self.dropped else w

    def evict(self, worker: str, warm: dict[str, bool], reason: str) -> bool:
        """Drop the worker's chapter affinity and its warm flag. Counts once per failure."""
        warm[worker] = False
        if worker in self.dropped:
            return False
        self.dropped.add(worker)
        self.evictions += 1
        self.log.append((worker, reason))
        return True

    def restore(self, worker: str) -> None:
        self.dropped.discard(worker)
