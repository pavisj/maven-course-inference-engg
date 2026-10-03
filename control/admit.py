"""Admission: should this request enter the chosen worker now?

should_shed(req, snap, rng) -> (shed, code, reason, retry_after_seconds)

Four tests, in this order; the first match wins:
1. tenant_tokens  one tenant's tokens in the last 60 s would pass its window
                  (per-worker window x warm workers; unused answer budget is refunded on completion)
2. timeout_queue  the estimated TTFT is already past the tenant's deadline
3. kv_free        the request's KV blocks would push KV usage past 85 %
4. over_cap       batch: 8 or more requests already wait for this worker;
                  interactive: every seat is taken and the estimated TTFT is above 2 s

Every shed is a 429: "try here again later". A 429 never overflows to another API.
Every retry_after gets jitter: x a random factor in [0.7, 1.3], so shed clients do not
come back together. Placement runs first (control/place.py), so admit checks the
worker the request would go to.
"""

import math
import random
from dataclasses import dataclass

from control.snapshot import COMPLETION_FLOOR, Snapshot

# tenant_tokens
TENANT_WINDOW_TOKENS = {"interactive": 400_000, "batch": 150_000}   # PER WARM WORKER (since Step 5, run D)
#   Baseline B (one worker), mixed mix, charged prompt + max_tokens per minute: interactive 337,306, batch 290,105.
#   400,000 gives interactive about 19 % headroom; 150,000 holds batch to about half its baseline-B demand.
#   Run D showed one-worker windows on two workers shed 75 % of all requests, so the window is
#   multiplied by the number of warm workers: two warm workers -> 800,000 and 300,000.
TENANT_WINDOW_S = 60.0           # sliding window length (the brief's "tenant window")
MIN_TENANT_RETRY_S = 2.0         # run D: sub-second retries made 8,027 requests for 2,005 admitted

# timeout_queue: est_ttft = max(Little's law, prefill backlog)
DEADLINE_S = {"interactive": 6.0, "batch": 60.0}
#   interactive: above baseline B's worst laptop-measured TTFT p95 (5,391 ms, unique mix); batch: practice questions can wait a minute.
IDLE_TTFT_S = 0.15               # Little's law floor: baseline A engine TTFT mean, 153 ms (Step 5a)
STEP_TOKENS = 4096               # prefill backlog: --max-num-batched-tokens, prefill tokens per engine step (Part 1)
STEP_SECONDS = 0.85              # prefill backlog: seconds per full step (Step 4 assumption; now only a lower bound)
BASE_TTFT_S = 0.4                # prefill backlog: TTFT with nothing waiting (Step 4); run J: 0 (gateway --base-ttft)

# kv_free
BLOCK_TOKENS = 16                # vLLM block_size (vllm:cache_config_info)
TOTAL_BLOCKS = 24_876            # fallback for vLLM num_gpu_blocks: 398,016 KV tokens / 16 (Part 1); the scrape gives the live value
KV_SHED_PCT = 85.0               # keep 15 % of KV free for decode growth (baseline B peaked at 14.5 %)

# over_cap
BATCH_MAX_WAITING = 8            # batch: shed when vLLM waiting + gateway queue for the worker reaches 8 (Step 5a)
OVER_CAP_TTFT_S = 2.0            # interactive: shed only when seats are full and est_ttft is above 2 s; run J: 4 s (--over-cap-ttft)

# retry_after
KV_RETRY_S = 5.0                 # kv_free: KV frees as running requests finish
MAX_RETRY_S = 10.0               # cap for timeout_queue and over_cap
MIN_RETRY_S = 1.0                # floor: over_cap's formula goes negative when est_ttft is below the deadline
JITTER = (0.7, 1.3)              # every retry_after x a random factor in this range (Step 5a)

TARGET = "WorkerA"               # default worker when a caller does not name one

# Capacity overflow (Step 12; instructor's AMA whiteboard, ~01:18: "any time there is a request
# overflow on the gateway itself, this goes directly to the Overflow"). These (tenant, reason)
# sheds go to the overflow gate instead of a 429; the gate's limiter decides, and over the
# limit the request still gets the 429. Batch over_cap is unchanged.
CAPACITY_OVERFLOW = {("interactive", "over_cap")}


# Run J settings (Pavithra's three open decisions after run I). The constants above stay run I's
# values; the gateway applies these only when its flags ask for it.
RUN_I_SETTINGS = {"over_cap_ttft_s": OVER_CAP_TTFT_S, "tenant_window_scale": 1.0, "base_ttft_s": BASE_TTFT_S,
                  "interactive_deadline_s": DEADLINE_S["interactive"]}
RUN_J_SETTINGS = {"over_cap_ttft_s": 4.0, "tenant_window_scale": 2.0, "base_ttft_s": 0.0}
_BASE_WINDOWS = dict(TENANT_WINDOW_TOKENS)


def apply_settings(over_cap_ttft_s: float = OVER_CAP_TTFT_S, tenant_window_scale: float = 1.0,
                   base_ttft_s: float = BASE_TTFT_S, interactive_deadline_s: float = 6.0) -> dict:
    """Set the interactive over_cap trigger, the tenant-window scale and the backlog floor.

    The scale multiplies run I's per-worker windows (400,000 and 150,000), not the current ones,
    so a second call does not compound. The interactive deadline is set in admit and in the queue
    together (run M), so the two never disagree. Returns the active values, for the run log.
    """
    from control import queue
    global OVER_CAP_TTFT_S, TENANT_WINDOW_TOKENS, BASE_TTFT_S
    OVER_CAP_TTFT_S = float(over_cap_ttft_s)
    TENANT_WINDOW_TOKENS = {t: int(v * tenant_window_scale) for t, v in _BASE_WINDOWS.items()}
    BASE_TTFT_S = float(base_ttft_s)
    DEADLINE_S["interactive"] = queue.QUEUE_DEADLINE_S["interactive"] = float(interactive_deadline_s)
    return {"interactive_deadline_s": DEADLINE_S["interactive"], "over_cap_ttft_s": OVER_CAP_TTFT_S, "tenant_window_scale": float(tenant_window_scale),
            "tenant_window_tokens": dict(TENANT_WINDOW_TOKENS), "base_ttft_s": BASE_TTFT_S}


def overflow_instead(tenant: str, reason: str) -> bool:
    """True if this shed should go to the Overflow server instead of a 429."""
    return (tenant, reason) in CAPACITY_OVERFLOW


@dataclass(frozen=True)
class AdmitRequest:
    tenant: str
    prompt_tokens: int
    max_tokens: int
    worker: str = TARGET


def tenant_window(snap: Snapshot, tenant: str) -> int:
    """The tenant's window: per-worker tokens x the number of warm workers (at least 1)."""
    warm = sum(1 for v in snap.warm.values() if v)
    return TENANT_WINDOW_TOKENS[tenant] * max(1, warm)


def little_ttft(snap: Snapshot, worker: str) -> float:
    """Little's law: requests ahead / completions per second + idle TTFT."""
    w = snap.workers.get(worker)
    ahead = (w.waiting if w else 0.0) + snap.queue_depth.get(worker, 0)
    rate = max(COMPLETION_FLOOR, snap.completion_rate.get(worker, COMPLETION_FLOOR))
    return ahead / rate + IDLE_TTFT_S


def prefill_backlog(snap: Snapshot, worker: str) -> float:
    """Step 4 formula: the prefill tokens ahead, at 4,096 tokens per 0.85 s step."""
    w = snap.workers.get(worker)
    ahead = (w.waiting if w else 0.0) + snap.queue_depth.get(worker, 0)
    return ahead * snap.avg_uncached_prompt_tokens / STEP_TOKENS * STEP_SECONDS + BASE_TTFT_S


def est_ttft(snap: Snapshot, worker: str = TARGET) -> float:
    """Little's law, with the prefill backlog as a lower bound."""
    return max(little_ttft(snap, worker), prefill_backlog(snap, worker))


def _jitter(seconds: float, rng) -> float:
    return round(seconds * rng.uniform(*JITTER), 2)


def _clamp_retry(seconds: float) -> float:
    return min(MAX_RETRY_S, max(MIN_RETRY_S, seconds))


def tenant_retry_after(events, used: int, need: int, window_tokens: int, now: float,
                       window_s: float = TENANT_WINDOW_S) -> float:
    """Seconds until enough old charges leave the window for this request to fit."""
    excess = used + need - window_tokens
    freed = 0
    for ts, tokens in events:          # oldest first
        freed += tokens
        if freed >= excess:
            return max(0.0, ts + window_s - now)
    return window_s                    # unknown history, or the request alone is larger than the window


def should_shed(req: AdmitRequest, snap: Snapshot, rng=random) -> tuple[bool, int, str, float]:
    need = req.prompt_tokens + req.max_tokens
    w = snap.workers.get(req.worker)

    # 1. tenant_tokens
    used = snap.tenant_tokens_60s.get(req.tenant, 0)
    window = tenant_window(snap, req.tenant)
    if used + need > window:
        events = snap.tenant_events.get(req.tenant, ())
        retry = _jitter(tenant_retry_after(events, used, need, window, snap.ts), rng)
        return True, 429, "tenant_tokens", max(MIN_TENANT_RETRY_S, retry)

    # 2. timeout_queue
    ttft = est_ttft(snap, req.worker)
    deadline = DEADLINE_S[req.tenant]
    if ttft > deadline:
        return True, 429, "timeout_queue", _jitter(_clamp_retry(ttft - deadline + 1), rng)

    # 3. kv_free
    kv_pct = w.kv_usage_pct if w else 0.0
    total_blocks = (w.num_gpu_blocks if w and w.num_gpu_blocks else None) or TOTAL_BLOCKS
    blocks_needed = math.ceil(need / BLOCK_TOKENS)
    if kv_pct + blocks_needed / total_blocks * 100 > KV_SHED_PCT:
        return True, 429, "kv_free", _jitter(KV_RETRY_S, rng)

    # 4. over_cap, by tenant
    waiting = w.waiting if w else 0.0
    queue_depth = snap.queue_depth.get(req.worker, 0)
    if req.tenant == "batch" and waiting + queue_depth >= BATCH_MAX_WAITING:
        return True, 429, "over_cap", _jitter(_clamp_retry(ttft - deadline + 1), rng)
    seats_full = snap.in_flight.get(req.worker, 0) >= snap.slots
    if req.tenant == "interactive" and seats_full and ttft > OVER_CAP_TTFT_S:
        return True, 429, "over_cap", _jitter(_clamp_retry(ttft - deadline + 1), rng)

    return False, 200, "ok", 0.0
