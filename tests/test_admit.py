import random

import pytest

from control.admit import AdmitRequest, est_ttft, little_ttft, prefill_backlog, should_shed
from control.snapshot import Snapshot, WorkerState

NOW = 10_000.0


class NoJitter:
    """Stands in for random: the jitter factor is always 1.0."""

    @staticmethod
    def uniform(a, b):
        return 1.0


def snap(waiting=0, running=0, kv=0.0, queue=0, in_flight=0, rate=3.0, tokens=None, events=None,
         uncached=1200, blocks=None, worker="WorkerA"):
    w = WorkerState(worker, True, running, waiting, kv, 0, NOW, blocks)
    return Snapshot(
        workers={worker: w},
        queue_depth={worker: queue},
        in_flight={worker: in_flight},
        completion_rate={worker: rate},
        warm={worker: True},
        tenant_tokens_60s=tokens or {},
        tenant_events=events or {},
        avg_uncached_prompt_tokens=uncached,
        ts=NOW,
    )


def interactive(prompt=2_100, max_tokens=512, worker="WorkerA"):
    return AdmitRequest("interactive", prompt, max_tokens, worker)


def batch(prompt=2_016, max_tokens=256, worker="WorkerA"):
    return AdmitRequest("batch", prompt, max_tokens, worker)


def shed(req, s):
    return should_shed(req, s, rng=NoJitter)


# ---- the estimate ----

def test_little_law():
    s = snap(waiting=4, queue=8, rate=3.0)        # 12 ahead / 3 per s + 0.15 s = 4.15 s
    assert little_ttft(s, "WorkerA") == pytest.approx(4.15)
    assert est_ttft(s, "WorkerA") == pytest.approx(4.15)


def test_prefill_backlog_is_a_lower_bound():
    s = snap(waiting=10, rate=100.0, uncached=4096)  # Little: 0.25 s; backlog: 10 x 1 step x 0.85 + 0.4 = 8.9 s
    assert little_ttft(s, "WorkerA") == pytest.approx(0.25)
    assert est_ttft(s, "WorkerA") == pytest.approx(prefill_backlog(s, "WorkerA")) == pytest.approx(8.9)


def test_completion_rate_floor():
    s = snap(queue=1, rate=0.0)                   # floor 0.5 per s: 1 / 0.5 + 0.15 = 2.15 s
    assert little_ttft(s, "WorkerA") == pytest.approx(2.15)


# ---- the four reasons ----

def test_idle_2100_token_interactive_passes():
    assert shed(interactive(), snap()) == (False, 200, "ok", 0.0)


def test_tenant_tokens():
    s = snap(tokens={"batch": 149_000}, events={"batch": ((NOW - 50, 149_000),)})
    assert shed(batch(), s) == (True, 429, "tenant_tokens", 10.0)


def test_timeout_queue():
    s = snap(queue=20, rate=3.0)                  # 20 / 3 + 0.15 = 6.82 s > 6 s
    assert shed(interactive(), s) == (True, 429, "timeout_queue", pytest.approx(1.82, abs=0.01))


def test_timeout_queue_uses_tenant_deadline():
    s = snap(queue=20, rate=3.0, in_flight=32)    # 6.82 s is inside batch's 60 s; but batch over_cap applies
    assert shed(batch(), s)[2] == "over_cap"
    assert shed(batch(), snap(queue=7, rate=0.5))[2] == "ok"  # 14.15 s < 60 s and 7 < 8 waiting


def test_kv_free():
    assert shed(interactive(), snap(kv=84.5)) == (True, 429, "kv_free", 5.0)


def test_kv_free_uses_live_block_count():
    # with 8,000 blocks (a smaller KV), 2,612 tokens = 164 blocks = 2.05 % -> 83.0 + 2.05 > 85
    assert shed(interactive(), snap(kv=83.0, blocks=8_000))[2] == "kv_free"
    assert shed(interactive(), snap(kv=83.0))[2] == "ok"   # 24,876 blocks: 0.66 % -> 83.66 % < 85 %


def test_kv_free_does_not_fire_at_baseline_b_peak():
    assert shed(interactive(), snap(kv=14.5))[0] is False


def test_over_cap_batch_at_8_waiting():
    assert shed(batch(), snap(waiting=3, queue=5))[2] == "over_cap"
    assert shed(batch(), snap(waiting=3, queue=4))[2] == "ok"


def test_over_cap_interactive_needs_full_seats_and_slow_ttft():
    full_slow = snap(queue=6, rate=2.0, in_flight=32)     # 6 / 2 + 0.15 = 3.15 s > 2 s, 32 of 32 seats
    assert shed(interactive(), full_slow) == (True, 429, "over_cap", 1.0)
    assert shed(interactive(), snap(queue=6, rate=2.0, in_flight=31))[2] == "ok"   # a seat is free
    assert shed(interactive(), snap(queue=2, rate=2.0, in_flight=32))[2] == "ok"   # 1.15 s < 2 s


def test_first_match_wins():
    s = snap(queue=30, tokens={"interactive": 399_000}, events={"interactive": ((NOW - 1, 399_000),)})
    assert shed(interactive(), s)[2] == "tenant_tokens"


def test_tenant_window_scales_with_warm_workers():
    from control.admit import tenant_window
    two = Snapshot(workers={}, warm={"WorkerA": True, "WorkerB": True}, ts=NOW)
    one = Snapshot(workers={}, warm={"WorkerA": True, "WorkerB": False}, ts=NOW)
    none = Snapshot(workers={}, warm={}, ts=NOW)
    assert tenant_window(two, "interactive") == 800_000 and tenant_window(two, "batch") == 300_000
    assert tenant_window(one, "interactive") == 400_000 and tenant_window(none, "batch") == 150_000


def test_tenant_tokens_retry_floor():
    # the oldest charge leaves the window in 0.5 s, but a retry is never sooner than 2 s
    s = snap(tokens={"batch": 149_000}, events={"batch": ((NOW - 59.5, 149_000),)})
    assert shed(batch(), s) == (True, 429, "tenant_tokens", 2.0)


def test_tenant_refund_and_completion_seed():
    from control.snapshot import COMPLETION_SEED, SnapshotSource, TenantWindow
    w = TenantWindow(60)
    charge = w.charge("interactive", 2_612, now=100.0)
    TenantWindow.refund(charge, 512 - 170)          # 170 of 512 answer tokens used
    assert w.total("interactive", now=101.0) == 2_270
    src = SnapshotSource({"WorkerA": "http://127.0.0.1:1"})
    assert src.completion_rate("WorkerA", now=src.start + 1) == COMPLETION_SEED   # no completions yet
    for i in range(30):
        src.record_completion("WorkerA", src.start + 1 + i * 0.5)
    assert src.completion_rate("WorkerA", now=src.start + 16) == pytest.approx(30 / 15)   # span from the first completion


def test_completion_rate_ignores_idle_time_before_traffic():
    from control.snapshot import SnapshotSource
    src = SnapshotSource({"WorkerA": "http://127.0.0.1:1"})
    start = src.start + 12.0                       # 12 s of warm-up wait with no traffic (run I)
    for i in range(12):
        src.record_completion("WorkerA", start + i * 0.25)     # 12 completions in 3 s = 4 per second
    rate = src.completion_rate("WorkerA", now=start + 3.0)
    assert rate == pytest.approx(12 / 5.0)         # span floor 5 s; was 12 / 15 = 0.8 per second


def test_retry_after_jitter_range():
    s = snap(kv=90.0)
    rng = random.Random(7)
    retries = {should_shed(interactive(), s, rng=rng)[3] for _ in range(200)}
    assert min(retries) >= 5.0 * 0.7 and max(retries) <= 5.0 * 1.3
    assert len(retries) > 50                      # it really varies


# ---- run J settings: each is off by default, so run I's values stay ----

@pytest.fixture
def run_i_after():
    from control.admit import RUN_I_SETTINGS, apply_settings
    yield apply_settings
    apply_settings(**RUN_I_SETTINGS)


def test_run_j_defaults_are_run_i_values():
    import control.admit as admit
    assert admit.OVER_CAP_TTFT_S == 2.0 and admit.BASE_TTFT_S == 0.4
    assert admit.TENANT_WINDOW_TOKENS == {"interactive": 400_000, "batch": 150_000}


def test_run_j_over_cap_trigger_4s(run_i_after):
    s = snap(queue=6, rate=2.0, in_flight=32)              # seats full, est 3.15 s
    assert shed(interactive(), s)[2] == "over_cap"         # run I: 3.15 s > 2 s
    run_i_after(over_cap_ttft_s=4.0)
    assert shed(interactive(), s)[2] == "ok"               # run J: 3.15 s < 4 s
    assert shed(interactive(), snap(queue=8, rate=2.0, in_flight=32))[2] == "over_cap"   # 4.15 s > 4 s


def test_run_j_tenant_windows_x2(run_i_after):
    from control.admit import tenant_window
    two = snap(tokens={"batch": 299_000}, events={"batch": ((NOW - 50, 299_000),)})
    two.warm["WorkerB"] = True                             # two warm workers: batch window 300,000
    assert shed(batch(), two)[2] == "tenant_tokens"
    info = run_i_after(tenant_window_scale=2.0)
    assert info["tenant_window_tokens"] == {"interactive": 800_000, "batch": 300_000}
    assert tenant_window(two, "batch") == 600_000 and tenant_window(two, "interactive") == 1_600_000
    assert shed(batch(), two)[2] == "ok"
    run_i_after(tenant_window_scale=2.0)                   # a second call does not compound
    assert tenant_window(two, "batch") == 600_000


def test_run_j_base_ttft_floor_removed(run_i_after):
    idle = snap()                                          # nothing waiting, nothing queued
    assert est_ttft(idle, "WorkerA") == pytest.approx(0.4)   # run I: the backlog floor wins
    run_i_after(base_ttft_s=0.0)
    assert prefill_backlog(idle, "WorkerA") == 0.0
    assert est_ttft(idle, "WorkerA") == pytest.approx(0.15)  # run J: Little's law idle TTFT (baseline A 153 ms)
    s = snap(waiting=10, rate=100.0, uncached=4096)        # backlog alone: 10 x 0.85 = 8.5 s
    assert est_ttft(s, "WorkerA") == pytest.approx(8.5)


def test_run_m_interactive_deadline_sets_admit_and_queue(run_i_after):
    from control import queue
    s = snap(queue=10, rate=3.0)                           # 10 / 3 + 0.15 = 3.48 s
    assert shed(interactive(), s)[2] == "ok"               # under the 6 s deadline
    info = run_i_after(interactive_deadline_s=3.0)
    assert info["interactive_deadline_s"] == 3.0 and queue.QUEUE_DEADLINE_S["interactive"] == 3.0
    assert shed(interactive(), s)[2] == "timeout_queue"    # 3.48 s > 3 s
    assert shed(batch(), s)[2] != "timeout_queue"          # batch keeps its 60 s
    run_i_after()
    assert queue.QUEUE_DEADLINE_S["interactive"] == 6.0
