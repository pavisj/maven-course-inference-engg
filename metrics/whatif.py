"""What-if replay over logged gateway decisions (gateway_requests.jsonl).

    python metrics/whatif.py

For each knob, replay the logged requests in order and count the ones whose decision would
change bucket, plus a first-order goodput change.

FIRST-ORDER ONLY: a changed decision changes the later queue state (more admitted requests
make later ones slower), and the replay keeps the logged state, so each number is a bound,
not a measurement.

Knobs:
1. LOAD_BOUND 22 -> 28: hopped requests whose home had fewer than 28 in flight (nearest
   gateway scrape) would stay home. Goodput: a late hopped request counts as rescued if its
   TTFT minus the run's hop penalty (median TTFT hopped - not hopped) meets the deadline.
2. Interactive over_cap trigger 2 s -> 4 s: interactive over_cap requests with est_ttft <= 4 s
   would be admitted. Goodput: rescued if est_ttft <= the 6 s deadline.
3. Overflow limiter 10 -> 30 per minute: re-run the limiter over run I's capacity-overflow
   attempts. Goodput: extra sends x the share of run I's overflow answers within deadline.
4. Tenant windows x1 -> x2: re-run the 60 s window with twice the tokens; tenant_tokens sheds
   that now fit are admitted. Goodput: x that tenant's in-run share of admitted answers within deadline.
Writes metrics/whatif.json.
"""

import json
import statistics
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEADLINE_S = {"interactive": 6.0, "batch": 60.0}


def load(folder: str, pattern: str) -> list[dict]:
    return [json.loads(l) for p in sorted((ROOT / "metrics" / folder).glob(pattern)) for l in p.read_text().splitlines() if l.strip()]


def minutes_of(rows):
    return max((max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 60, 1e-9)


def on_time(r):
    return r.get("ttft_measured") is not None and r["ttft_measured"] <= DEADLINE_S.get(r["tenant"], 6.0)


def load_bound(folder: str, old: int = 22, new: int = 28) -> dict | None:
    gw = load(folder, "gateway_requests.jsonl")
    ok = [r for r in gw if r.get("decision") == "ok" and r.get("ttft_measured") is not None]
    hopped = [r for r in ok if r.get("hopped")]
    if not hopped or ok[0].get("policy", "prefix_then_load") != "prefix_then_load" or ok[0].get("load_bound", old) != old:
        return None
    scrape = [r for r in load(folder, "scrape_*.jsonl") if r.get("worker") == "gateway" and r.get("gateway")]
    def in_flight(ts, worker):
        prior = [s for s in scrape if s["ts"] <= ts]
        row = prior[-1] if prior else (scrape[0] if scrape else None)
        return row["gateway"].get(f'gateway_in_flight{{worker="{worker}"}}', 0) if row else 0
    stay = [r for r in hopped if in_flight(r["ts"], r["preferred"]) < new]
    penalty = statistics.median(r["ttft_measured"] for r in hopped) - statistics.median(
        r["ttft_measured"] for r in ok if not r.get("hopped"))
    rescued = sum(1 for r in stay if not on_time(r) and r["ttft_measured"] - penalty <= DEADLINE_S[r["tenant"]])
    return {"knob": "LOAD_BOUND 22 -> 28", "run": folder, "hops_logged": len(hopped), "change_bucket": len(stay),
            "hop_penalty_ms": round(penalty * 1000), "goodput_change_per_min": round(rescued / minutes_of(gw), 2)}


def over_cap_trigger(folder: str, new: float = 4.0) -> dict | None:
    gw = load(folder, "gateway_requests.jsonl")
    ic = [r for r in gw if r.get("tenant") == "interactive" and r.get("reason") == "over_cap" and r.get("decision") in ("shed", "overflow")]
    if not ic:
        return None
    admit = [r for r in ic if r.get("est_ttft") is not None and r["est_ttft"] <= new]
    rescued = sum(1 for r in admit if r["decision"] == "shed" and r["est_ttft"] <= DEADLINE_S["interactive"])
    return {"knob": "interactive over_cap trigger 2 s -> 4 s", "run": folder, "over_cap_logged": len(ic),
            "change_bucket": len(admit), "goodput_change_per_min": round(rescued / minutes_of(gw), 2)}


def overflow_limiter(folder: str, old: int = 10, new: int = 30) -> dict | None:
    gw = load(folder, "gateway_requests.jsonl")
    tries = sorted((r for r in gw if r.get("reason") == "over_cap" and r.get("tenant") == "interactive"
                    and (r.get("decision") == "overflow" or r.get("overflow_outcome") in ("limited", "failed"))),
                   key=lambda r: r["ts"])
    if not tries:
        return None
    def sends(limit):
        q, n = deque(), 0
        for r in tries:
            while q and q[0] <= r["ts"] - 60:
                q.popleft()
            if len(q) < limit:
                q.append(r["ts"])
                n += 1
        return n
    sent_ok = [r for r in gw if r.get("decision") == "overflow" and r.get("status") == 200]
    share = sum(1 for r in sent_ok if on_time(r)) / len(sent_ok) if sent_ok else 0
    extra = sends(new) - sends(old)
    return {"knob": "overflow limiter 10 -> 30 per minute", "run": folder, "overflow_attempts": len(tries),
            "change_bucket": extra, "overflow_on_time_share": round(share, 3),
            "goodput_change_per_min": round(extra * share / minutes_of(gw), 2)}


def tenant_windows(folder: str, windows: dict, factor: float = 2.0) -> dict | None:
    gw = sorted(load(folder, "gateway_requests.jsonl"), key=lambda r: r["ts"])
    shed = [r for r in gw if r.get("reason") == "tenant_tokens"]
    if not shed:
        return None
    q = {t: deque() for t in windows}
    used = {t: 0 for t in windows}
    changed, rescued = 0, 0
    share = {}
    for t in windows:
        adm = [r for r in gw if r.get("tenant") == t and r.get("decision") == "ok"]
        share[t] = sum(1 for r in adm if on_time(r)) / len(adm) if adm else 0
    for r in gw:
        t = r.get("tenant")
        if t not in windows or r.get("prompt_tokens") is None:
            continue
        while q[t] and q[t][0][0] <= r["ts"] - 60:
            used[t] -= q[t].popleft()[1]
        need = r["prompt_tokens"] + (r.get("max_tokens") or 0)
        admitted = r.get("decision") in ("ok", "overflow")
        if r.get("reason") == "tenant_tokens" and used[t] + need <= windows[t] * factor:
            admitted = True
            changed += 1
            rescued += share[t]
        if admitted:
            q[t].append((r["ts"], need))
            used[t] += need
    return {"knob": "tenant windows x1 -> x2", "run": folder, "tenant_tokens_logged": len(shed),
            "change_bucket": changed, "goodput_change_per_min": round(rescued / minutes_of(gw), 2)}


def main():
    one = {"interactive": 400_000, "batch": 150_000}      # runs D, E: one-worker windows
    two = {"interactive": 800_000, "batch": 300_000}      # runs G, I: windows x 2 warm workers
    rows = []
    for folder in ("run_d", "run_e", "run_g", "run_f2", "run_i"):
        rows.append(load_bound(folder))
    for folder in ("run_g", "run_i"):
        rows.append(over_cap_trigger(folder))
    rows.append(overflow_limiter("run_i"))
    for folder, w in (("run_d", one), ("run_e", one), ("run_g", two), ("run_i", two)):
        rows.append(tenant_windows(folder, w))
    rows = [r for r in rows if r]
    (ROOT / "metrics" / "whatif.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
