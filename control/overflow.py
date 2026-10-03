"""Overflow gate: two kinds of overflow share this module and its limiter.

- Failure overflow (Step 8): an engine answered 503/529 -> the other worker once, then Overflow.
- Capacity overflow (Step 12): admit says the gateway itself is over capacity for an interactive
  request (over_cap) -> Overflow instead of a 429, under the same limiter, with
  X-Overflow-Reason: over_cap. Decided by the gateway, before any engine sees the request.

Failure overflow: after a worker answers, does the request stay or leave?

classify(status, body) -> "ok" | "stay" | "leave" | "other"
- 200                                  -> ok:    stream it to the client
- 429, 500, or an out-of-memory text   -> stay:  return it to the client with Retry-After
- 503, 529                             -> leave: try the other worker once (still inside the
                                          cluster), then the Overflow server
- anything else (for example a 400)    -> other: pass it through unchanged

Leaving goes to the Overflow server (Qwen2.5-1.5B on its own A10, scales from zero).
At most OVERFLOW_PER_MIN requests leave per minute; above that the client gets the 503.
Every overflow response carries the header X-Via: overflow:qwen2.5-1.5b.
"""

import time
from collections import deque

STAY_CODES = {429, 500}
LEAVE_CODES = {503, 529}
OOM_MARKERS = ("out of memory", "outofmemoryerror", "slice_oom")   # matched in lower-case response text
STAY_RETRY_S = 2.0                     # Retry-After for a "stay" response that has none of its own
VIA_HEADER = "X-Via"
VIA_OVERFLOW = "overflow:qwen2.5-1.5b"
VIA_NONE = "overflow:none"             # no leave target is available
OVERFLOW_REASON_HEADER = "X-Overflow-Reason"   # capacity overflow (Step 12): why the gateway sent it away
OVERFLOW_PER_MIN = 10                  # limiter on requests sent to Overflow (Step 8)
OVERFLOW_WINDOW_S = 60.0


def classify(status: int, body: bytes | str = b"") -> str:
    if status == 200:
        return "ok"
    text = body.decode(errors="replace") if isinstance(body, bytes) else body
    if any(m in text.lower() for m in OOM_MARKERS) or status in STAY_CODES:
        return "stay"
    if status in LEAVE_CODES:
        return "leave"
    return "other"


class OverflowLimiter:
    """Sliding window: at most `per_window` requests may leave per `window_s` seconds."""

    def __init__(self, per_window: int = OVERFLOW_PER_MIN, window_s: float = OVERFLOW_WINDOW_S):
        self.per_window = per_window
        self.window_s = window_s
        self._sent = deque()

    def allow(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        while self._sent and self._sent[0] <= now - self.window_s:
            self._sent.popleft()
        if len(self._sent) >= self.per_window:
            return False
        self._sent.append(now)
        return True
