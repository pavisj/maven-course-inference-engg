"""Placement: which worker gets this request?

pick(req, workers, *, policy) -> Placement | Shed

Policies:
- prefix_then_load (default): go to the chapter's home worker (CHAPTER_HOME), so requests on
  the same chapter share a prefix cache. If the home worker has LOAD_BOUND or more requests in
  flight, or is not warm, go to the usable worker with the lowest score instead.
- least_loaded: the usable worker with the lowest score.
- p2c: power of two choices: sample two usable workers, take the lower score.

score = gateway in-flight + gateway queue depth + vLLM waiting (queue depth is a scorer).
Only warm, reachable workers are usable. No usable worker -> Shed(503, "no_warm_worker").
A request is placed once (no bounce). "preferred" is always the chapter's home worker, so
`hopped` means: this request's prefix will be recomputed away from its home.
"""

import random
from dataclasses import dataclass

# Prefix key -> home worker. The key is the file prefix of the top retrieved chunk
# (03a, 03b and 06b are separate files; the manifest's chapter labels are digits only).
CHAPTER_HOME = {
    "00": "WorkerA", "03": "WorkerA", "03a": "WorkerA", "03b": "WorkerA", "05": "WorkerA",
    "06": "WorkerB", "06b": "WorkerB", "07": "WorkerB", "09": "WorkerB", "10": "WorkerB",
}
LOAD_BOUND = 22                  # 70 % of the 32 decode slots (--max-num-seqs, Part 1)
POLICIES = ("prefix_then_load", "least_loaded", "p2c")
DEFAULT_POLICY = "prefix_then_load"


@dataclass(frozen=True)
class WorkerView:
    name: str
    warm: bool
    ok: bool
    in_flight: int
    queue_depth: int
    waiting: float

    @property
    def score(self) -> float:
        return self.in_flight + self.queue_depth + self.waiting


@dataclass(frozen=True)
class PlaceRequest:
    tenant: str
    chapter: str | None


@dataclass(frozen=True)
class Placement:
    preferred: str
    chosen: str
    hopped: bool
    policy: str


@dataclass(frozen=True)
class Shed:
    code: int
    reason: str


def views_from_snapshot(snap) -> dict[str, WorkerView]:
    views = {}
    for name, w in snap.workers.items():
        views[name] = WorkerView(name, snap.warm.get(name, False), w.ok, snap.in_flight.get(name, 0),
                                 snap.queue_depth.get(name, 0), w.waiting)
    return views


def _lowest(views) -> str:
    return min(views, key=lambda v: (v.score, v.name)).name


def pick(req: PlaceRequest, workers: dict[str, WorkerView], *, policy: str = DEFAULT_POLICY,
         rng=random, load_bound: int = LOAD_BOUND) -> Placement | Shed:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; use one of {POLICIES}")
    usable = [w for w in workers.values() if w.ok and w.warm]
    if not usable:
        return Shed(503, "no_warm_worker")
    home = CHAPTER_HOME.get(req.chapter or "")
    preferred = home if home in workers else _lowest(usable)

    if policy == "least_loaded":
        chosen = _lowest(usable)
    elif policy == "p2c":
        chosen = _lowest(rng.sample(usable, min(2, len(usable))))
    else:  # prefix_then_load
        home_view = workers.get(preferred)
        if home_view in usable and home_view.in_flight < load_bound:
            chosen = preferred
        else:
            chosen = _lowest(usable)
    return Placement(preferred, chosen, chosen != preferred, policy)
