"""Evict: a worker that fails its warm probe or restarts loses its chapter affinity and its warm flag."""
import control.snapshot as snapshot
from control.evict import Affinity
from control.snapshot import SnapshotSource


class FakeWorker:
    """Stands in for scrape_once: a worker that is up or down, with a TTFT counter that can go back."""

    def __init__(self):
        self.up, self.count = True, 100.0

    def __call__(self, worker, url, fields, run):
        row = {f: 0.0 for f in fields}
        row.update(ok=self.up, ttft_count=self.count, running=0.0, waiting=0.0, kv_usage=0.0, preemptions=0.0,
                   prefix_queries=0.0, prefix_hits=0.0, num_gpu_blocks=None)
        return row


def test_evict_on_down_and_restart(monkeypatch):
    fake = FakeWorker()
    monkeypatch.setattr(snapshot, "scrape_once", fake)
    src = SnapshotSource({"WorkerB": "http://fake-worker-b"})
    src.refresh()
    src.warm["WorkerB"] = True                         # passed its warm probe
    assert src.affinity.home("07") == "WorkerB"

    fake.up = False                                    # the container goes away
    src.refresh()
    src.refresh()                                      # still down: no second count
    assert src.affinity.evictions == 1 and src.warm["WorkerB"] is False
    assert src.affinity.home("07") is None             # chapter 07 has no home: place hops it

    fake.up = True                                     # back, warm probe passes
    src.refresh()
    src.warm["WorkerB"] = True
    src.affinity.restore("WorkerB")
    assert src.affinity.home("07") == "WorkerB"

    fake.count = 3.0                                   # counters went back: a new container
    src.refresh()
    assert src.affinity.evictions == 2 and src.warm["WorkerB"] is False
    assert [r for _, r in src.affinity.log] == ["unreachable", "restart"]


def test_evict_on_failed_warm_probe():
    aff, warm = Affinity(), {"WorkerA": True, "WorkerB": True}
    assert aff.evict("WorkerA", warm, "warm_probe_failed") is True
    assert aff.evict("WorkerA", warm, "warm_probe_failed") is False   # once per failure
    assert warm == {"WorkerA": False, "WorkerB": True}
    assert aff.home("03a") is None and aff.home("07") == "WorkerB"
    assert aff.evictions == 1
