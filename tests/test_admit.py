import pytest

from control.admit import AdmitRequest, est_ttft, should_shed
from control.snapshot import Snapshot, WorkerState

NOW = 10_000.0


def snap(waiting=0, running=0, kv=0.0, queue_depth=0, tokens=None, events=None, uncached=1200):
    worker = WorkerState("WorkerA", True, running, waiting, kv, 0, NOW)
    return Snapshot(
        workers={"WorkerA": worker},
        queue_depth=queue_depth,
        tenant_tokens_60s=tokens or {},
        tenant_events=events or {},
        avg_uncached_prompt_tokens=uncached,
        ts=NOW,
    )


def interactive(prompt=2_100, max_tokens=512):
    return AdmitRequest("interactive", prompt, max_tokens)


def test_idle_2100_token_interactive_passes():
    assert should_shed(interactive(), snap()) == (False, 200, "ok", 0.0)


def test_tenant_tokens():
    # 149,000 batch tokens charged 50 s ago; 2,016 + 256 more would pass the 150,000 window
    s = snap(tokens={"batch": 149_000}, events={"batch": ((NOW - 50, 149_000),)})
    shed, code, reason, retry = should_shed(AdmitRequest("batch", 2_016, 256), s)
    assert (shed, code, reason) == (True, 429, "tenant_tokens")
    assert retry == pytest.approx(10.0)          # the 149,000 tokens leave the window in 10 s


def test_timeout_queue():
    s = snap(waiting=30)                          # 30 x 1200 / 4096 x 0.85 + 0.4 = 7.87 s > 6 s
    assert est_ttft(s) == pytest.approx(7.87, abs=0.01)
    shed, code, reason, retry = should_shed(interactive(), s)
    assert (shed, code, reason) == (True, 429, "timeout_queue")
    assert retry == pytest.approx(2.87, abs=0.01)  # min(10, 7.87 - 6 + 1)


def test_timeout_queue_uses_tenant_deadline():
    # the same queue is fine for batch: 7.87 s is well inside its 60 s deadline
    assert should_shed(AdmitRequest("batch", 2_016, 256), snap(waiting=30))[2] == "ok"


def test_kv_free():
    s = snap(kv=84.5)                             # 84.5 % + ceil(2612 / 16) / 24,876 = 85.16 % > 85 %
    assert should_shed(interactive(), s) == (True, 429, "kv_free", 5.0)


def test_kv_free_does_not_fire_at_baseline_b_peak():
    assert should_shed(interactive(), snap(kv=14.5))[0] is False


def test_over_cap():
    s = snap(running=32, waiting=13, uncached=622)  # est 13 x 622 / 4096 x 0.85 + 0.4 = 2.08 s > 2 s
    shed, code, reason, retry = should_shed(interactive(), s)
    assert (shed, code, reason) == (True, 429, "over_cap")
    assert retry == 1.0                           # 2.08 - 6 + 1 is negative, so the 1 s floor applies


def test_over_cap_needs_both_conditions():
    assert should_shed(interactive(), snap(running=32, waiting=5, uncached=622))[0] is False   # est below 2 s
    assert should_shed(interactive(), snap(running=20, waiting=13, uncached=622))[0] is False  # slots not full


def test_first_match_wins():
    # over the tenant window and past the deadline: tenant_tokens is tested first
    s = snap(waiting=30, tokens={"interactive": 399_000}, events={"interactive": ((NOW - 1, 399_000),)})
    assert should_shed(interactive(), s)[2] == "tenant_tokens"
