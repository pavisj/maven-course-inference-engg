"""Compare two gateway runs on the same load (run J vs run I), and check the what-if predictions.

    python metrics/compare_runs.py --base run_i --new run_j --also run_m run_m2

Reads metrics/<run>/summary.json, metrics/<run>/gateway_requests.jsonl, metrics/goodput.json and
metrics/whatif.json (run summarise_run.py and goodput.py first). Writes metrics/<new>/compare_<base>.json.

1. The table: goodput per minute (total, by tenant), sheds by reason and tenant, gateway TTFT p50 / p95,
   overflow sent / limited / failed, est-vs-measured within 50 % (raw and with the network trip removed).
2. The estimator check: timeout_queue sheds in the first 15 s after the first request.
3. The what-if check: each first-order prediction for the base run against what the new run measured.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EARLY_S = 15.0


def load_rows(run: str) -> list[dict]:
    p = ROOT / "metrics" / run / "gateway_requests.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def within_50(rows: list[dict], offset: float) -> float:
    pairs = [(r["est_ttft"], max(0.0, r["ttft_measured"] - offset)) for r in rows
             if r.get("decision") == "ok" and r.get("est_ttft") is not None and r.get("ttft_measured") is not None]
    return round(sum(1 for e, x in pairs if abs(x - e) <= 0.5 * max(e, x)) / len(pairs), 3) if pairs else None


def early(rows: list[dict]) -> dict:
    t0 = min(r["ts"] for r in rows)
    first = [r for r in rows if r["ts"] - t0 <= EARLY_S]
    tq = [r for r in first if r.get("reason") == "timeout_queue"]
    return {"window_s": EARLY_S, "requests": len(first), "timeout_queue": len(tq),
            "timeout_queue_by_tenant": dict(Counter(r["tenant"] for r in tq)),
            "timeout_queue_est_max_s": max((r["est_ttft"] for r in tq), default=None),
            "timeout_queue_whole_run": sum(1 for r in rows if r.get("reason") == "timeout_queue")}


def est_floor(rows: list[dict]) -> dict:
    est = sorted(r["est_ttft"] for r in rows if r.get("est_ttft") is not None)
    return {"est_min_s": est[0], "est_p50_s": est[len(est) // 2],
            "share_below_0_4_s": round(sum(1 for e in est if e < 0.4) / len(est), 3)}


def run_view(run: str, goodput: dict) -> dict:
    s = json.loads((ROOT / "metrics" / run / "summary.json").read_text())
    g = next(x for x in goodput if x["folder"] == run)
    rows = load_rows(run)
    ev = s["est_vs_measured"]
    trip_own = ev.get("network_offset_s")
    return {
        "goodput_per_min": g["goodput_per_min"],
        "goodput_interactive": g["by_tenant"]["interactive"]["goodput_per_min"],
        "goodput_batch": g["by_tenant"]["batch"]["goodput_per_min"],
        "goodput_by_worker": g["by_worker_goodput_per_min"],
        "offered_per_min": g["offered_per_min"], "goodput_share": g["goodput_share"], "late": g["late_answers"],
        "requests_sent": s["requests_sent"], "admitted_by_tenant": s["admitted_by_tenant"],
        "sheds": s["sheds_by_reason_tenant"],
        "ttft_ms": s["gateway_ttft_ms"],
        "overflow": {k: s.get("overflow", {}).get(k, 0) for k in ("sent", "limited", "failed")},
        "overflow_ttft_ms": s.get("overflow", {}).get("overflow_ttft_ms"),
        "est_p50_s": ev["est_p50_s"], "measured_p50_s": ev["measured_p50_s"],
        "within_50_raw": ev["within_50pct"], "trip_s": trip_own,
        "within_50_minus_trip": ev.get("within_50pct_minus_trip"),
        "measured_minus_trip_p50_s": ev.get("measured_minus_trip_p50_s"),
        "within_50_minus_348ms": within_50(rows, 0.3478),
        "est": est_floor(rows),
        "early": early(rows),
        "workers": {w: {k: v[k] for k in ("prefix_hit_pct", "engine_ttft_mean_ms", "max_kv_pct", "preemptions",
                                          "gateway_queue_max", "gateway_in_flight_mean")}
                    for w, v in s["workers"].items()},
        "hops": s["hops"]["total"], "hop_share": s["hops"]["share_of_admitted"],
    }


def whatif_check(base: str, b: dict, n: dict) -> list[dict]:
    rows = [r for r in json.loads((ROOT / "metrics" / "whatif.json").read_text()) if r["run"] == base]
    out = []
    for r in rows:
        k = r["knob"]
        if k.startswith("interactive over_cap"):
            out.append({"knob": k, "tenant": "interactive",
                        "predicted_admitted_more": r["change_bucket"],
                        "measured_shed_change": n["sheds"].get("interactive/over_cap", 0) + n["overflow"]["sent"]
                        - (b["sheds"].get("interactive/over_cap", 0) + b["overflow"]["sent"]),
                        "measured_admitted_change": n["admitted_by_tenant"]["interactive"] - b["admitted_by_tenant"]["interactive"],
                        "predicted_goodput_change": r["goodput_change_per_min"],
                        "measured_goodput_change": round(n["goodput_interactive"] - b["goodput_interactive"], 1)})
        elif k.startswith("tenant windows"):
            out.append({"knob": k, "tenant": "batch (all of the base run's tenant_tokens sheds)",
                        "predicted_admitted_more": r["change_bucket"],
                        "measured_shed_change": sum(v for kk, v in n["sheds"].items() if kk.endswith("tenant_tokens"))
                        - sum(v for kk, v in b["sheds"].items() if kk.endswith("tenant_tokens")),
                        "measured_admitted_change": n["admitted_by_tenant"]["batch"] - b["admitted_by_tenant"]["batch"],
                        "predicted_goodput_change": r["goodput_change_per_min"],
                        "measured_goodput_change": round(n["goodput_batch"] - b["goodput_batch"], 1)})
    both = [o for o in out]
    if len(both) == 2:
        out.append({"knob": "both together (sum of the two first-order predictions)", "tenant": "all",
                    "predicted_goodput_change": round(sum(o["predicted_goodput_change"] for o in both), 1),
                    "measured_goodput_change": round(n["goodput_per_min"] - b["goodput_per_min"], 1)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="run_i")
    parser.add_argument("--new", default="run_j")
    parser.add_argument("--also", nargs="*", default=[], help="more runs to add as columns, e.g. run_m run_m2")
    args = parser.parse_args()
    goodput = json.loads((ROOT / "metrics" / "goodput.json").read_text())
    b, n = run_view(args.base, goodput), run_view(args.new, goodput)
    out = {"base": args.base, "new": args.new, args.base: b, args.new: n, "whatif": whatif_check(args.base, b, n)}
    out.update({run: run_view(run, goodput) for run in args.also})
    path = ROOT / "metrics" / args.new / f"compare_{args.base}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
