"""Admission: should this request enter the engine now?

should_shed(req, snap) -> (shed, code, reason, retry_after_seconds)

Four tests, in this order; the first match wins:
1. tenant_tokens  one tenant's tokens in the last 60 s would pass its window
2. timeout_queue  the estimated TTFT is already past the tenant's deadline
3. kv_free        the request's KV blocks would push KV usage past 85 %
4. over_cap       the engine's decode slots are full and the estimated TTFT is above 2 s

Every shed is a 429: "try here again later". A 429 never overflows to another API.
A worker with no scrape yet counts as idle (fail-open).
"""

import math
from dataclasses import dataclass

from control.snapshot import Snapshot

# tenant_tokens
TENANT_WINDOW_TOKENS = {"interactive": 400_000, "batch": 150_000}
#   Baseline B, mixed mix, charged prompt + max_tokens per minute: interactive 337,306, batch 290,105.
#   400,000 gives interactive about 19 % headroom; 150,000 holds batch to about half its baseline-B demand.
TENANT_WINDOW_S = 60.0           # sliding window length (the brief's "tenant window")

# timeout_queue
DEADLINE_S = {"interactive": 6.0, "batch": 60.0}
#   interactive: above baseline B's worst laptop-measured TTFT p95 (5,391 ms, unique mix); batch: practice questions can wait a minute.
STEP_TOKENS = 4096               # --max-num-batched-tokens: prefill tokens per engine step (cluster/modal_vllm.py, Part 1)
STEP_SECONDS = 0.85              # seconds per full engine step under load (assumed; baseline B implies 1.0 to 2.7 s, see DESIGN Part 3)
BASE_TTFT_S = 0.4                # TTFT with nothing waiting: baseline A engine TTFT mean 120 to 216 ms, plus margin

# kv_free
BLOCK_TOKENS = 16                # vLLM block_size (vllm:cache_config_info)
TOTAL_BLOCKS = 24_876            # vLLM num_gpu_blocks: 398,016 KV tokens / 16 (Part 1, measured)
KV_SHED_PCT = 85.0               # keep 15 % of KV free for decode growth (baseline B peaked at 14.5 %)

# over_cap
SLOTS = 32                       # --max-num-seqs (Part 1)
OVER_CAP_TTFT_S = 2.0            # baseline B mixed: est_ttft passes 2 s at about 13 waiting requests

# retry_after
KV_RETRY_S = 5.0                 # kv_free: KV frees as running requests finish
MAX_RETRY_S = 10.0               # cap for timeout_queue and over_cap
MIN_RETRY_S = 1.0                # floor: over_cap's formula goes negative when est_ttft is below the deadline

TARGET = "WorkerA"               # Step 4 forwards everything to worker A


@dataclass(frozen=True)
class AdmitRequest:
    tenant: str
    prompt_tokens: int
    max_tokens: int
    worker: str = TARGET


def est_ttft(snap: Snapshot, worker: str = TARGET) -> float:
    """(waiting + queue_depth) x avg uncached prompt tokens / 4096 x 0.85 s + 0.4 s."""
    w = snap.workers.get(worker)
    waiting = w.waiting if w else 0.0
    return (waiting + snap.queue_depth) * snap.avg_uncached_prompt_tokens / STEP_TOKENS * STEP_SECONDS + BASE_TTFT_S


def _clamp_retry(seconds: float) -> float:
    return round(min(MAX_RETRY_S, max(MIN_RETRY_S, seconds)), 2)


def tenant_retry_after(events, used: int, need: int, window_tokens: int, now: float,
                       window_s: float = TENANT_WINDOW_S) -> float:
    """Seconds until enough old charges leave the window for this request to fit."""
    excess = used + need - window_tokens
    freed = 0
    for ts, tokens in events:          # oldest first
        freed += tokens
        if freed >= excess:
            return round(max(0.0, ts + window_s - now), 2)
    return window_s                    # unknown history, or the request alone is larger than the window


def should_shed(req: AdmitRequest, snap: Snapshot) -> tuple[bool, int, str, float]:
    need = req.prompt_tokens + req.max_tokens

    # 1. tenant_tokens
    used = snap.tenant_tokens_60s.get(req.tenant, 0)
    window = TENANT_WINDOW_TOKENS[req.tenant]
    if used + need > window:
        events = snap.tenant_events.get(req.tenant, ())
        return True, 429, "tenant_tokens", tenant_retry_after(events, used, need, window, snap.ts)

    # 2. timeout_queue
    ttft = est_ttft(snap, req.worker)
    deadline = DEADLINE_S[req.tenant]
    if ttft > deadline:
        return True, 429, "timeout_queue", _clamp_retry(ttft - deadline + 1)

    # 3. kv_free
    w = snap.workers.get(req.worker)
    kv_pct = w.kv_usage_pct if w else 0.0
    blocks_needed = math.ceil(need / BLOCK_TOKENS)
    if kv_pct + blocks_needed / TOTAL_BLOCKS * 100 > KV_SHED_PCT:
        return True, 429, "kv_free", KV_RETRY_S

    # 4. over_cap
    running = w.running if w else 0.0
    if running + snap.queue_depth >= SLOTS and ttft > OVER_CAP_TTFT_S:
        return True, 429, "over_cap", _clamp_retry(ttft - deadline + 1)

    return False, 200, "ok", 0.0
