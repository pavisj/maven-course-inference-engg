"""Summarise one gateway run folder into <run-dir>/summary.json.

    python metrics/summarise_run.py metrics/run_d

Reads gateway_requests.jsonl, hops.jsonl, scrape_*.jsonl, traffic_*.jsonl (and warm_proof.json
if present). Reports sheds by reason and tenant, hops, placement, est vs measured TTFT,
per-worker gateway queue and in-flight, max waiting, max KV %, preemptions, prefix hit rate
per worker, and TTFT per mix.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(folder: Path, pattern: str) -> list[dict]:
    return [json.loads(l) for p in sorted(folder.glob(pattern)) for l in p.read_text().splitlines() if l.strip()]


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))] if values else None


def ms(v):
    return round(v * 1000) if v is not None else None


def main(folder: Path) -> dict:
    gw = load(folder, "gateway_requests.jsonl")
    hops = load(folder, "hops.jsonl")
    scrape = load(folder, "scrape_*.jsonl")
    traffic = load(folder, "traffic_*.jsonl")
    out = {"run": folder.name, "requests_sent": len(gw)}

    out["decisions"] = dict(Counter(r["decision"] for r in gw))
    out["sheds_by_reason_tenant"] = {f"{t}/{r}": n for (t, r), n in sorted(Counter(
        (r["tenant"], r["reason"]) for r in gw if r["decision"] == "shed").items())}
    out["sheds_by_stage"] = dict(Counter(r.get("stage") for r in gw if r["decision"] == "shed"))
    out["guard_rejects"] = dict(Counter(r["reason"] for r in gw if r["decision"] == "guard"))
    out["leave"] = dict(Counter((r.get("outcome") or r.get("reason")) for r in gw if r["decision"] == "leave"))
    out["aborts"] = {"queued": sum(1 for r in gw if r["decision"] in ("abort", "aborted") and r.get("stage") == "queue"),
                     "in_flight": sum(1 for r in gw if r.get("outcome") == "client_gone")}
    ok = [r for r in gw if r["decision"] == "ok"]
    out["admitted_by_tenant"] = dict(Counter(r["tenant"] for r in ok))
    out["placed_by_worker"] = dict(Counter(r.get("worker") for r in ok))
    out["preferred_by_worker"] = dict(Counter(r.get("preferred") for r in ok))
    out["hops"] = {"total": len(hops), "by_route": {f"{s}->{d}": n for (s, d), n in Counter((h["src"], h["dst"]) for h in hops).items()},
                   "tokens_recomputed": sum(h["tokens"] for h in hops),
                   "share_of_admitted": round(len(hops) / len(ok), 3) if ok else None}

    # TTFT at the gateway (from arrival, includes queue wait), est vs measured
    m = [r for r in ok if r.get("ttft_measured") is not None]
    out["gateway_ttft_ms"] = {"all": {"p50": ms(pct([r["ttft_measured"] for r in m], 50)), "p95": ms(pct([r["ttft_measured"] for r in m], 95))}}
    for t in ("interactive", "batch"):
        v = [r["ttft_measured"] for r in m if r["tenant"] == t]
        if v:
            out["gateway_ttft_ms"][t] = {"p50": ms(pct(v, 50)), "p95": ms(pct(v, 95))}
    pairs = [(r["est_ttft"], r["ttft_measured"]) for r in m if r.get("est_ttft") is not None]
    offset_file = folder / "network_offset.json"           # the run's own trip (run J), else the shared one
    offset_file = offset_file if offset_file.exists() else folder.parent / "network_offset.json"
    offset = json.loads(offset_file.read_text())["offset_s"] if offset_file.exists() else None
    if pairs:
        ratio = [mt / es for es, mt in pairs if es > 0]
        out["est_vs_measured"] = {
            "n": len(pairs), "est_p50_s": round(pct([e for e, _ in pairs], 50), 3),
            "measured_p50_s": round(pct([x for _, x in pairs], 50), 3),
            "measured_over_est_p50": round(pct(ratio, 50), 2),
            "within_50pct": round(sum(1 for e, x in pairs if abs(x - e) <= 0.5 * max(e, x)) / len(pairs), 3),
            "queue_wait_p95_s": round(pct([r.get("queue_wait_s") for r in ok], 95) or 0, 3),
        }
        if offset is not None:     # the estimate predicts engine TTFT; take the laptop -> worker trip out
            net = [(e, max(0.0, x - offset)) for e, x in pairs]
            out["est_vs_measured"].update(
                network_offset_s=offset,
                measured_minus_trip_p50_s=round(pct([x for _, x in net], 50), 3),
                within_50pct_minus_trip=round(sum(1 for e, x in net if abs(x - e) <= 0.5 * max(e, x)) / len(net), 3))

    # overflow (Step 12 capacity overflow and Step 8 failure overflow)
    ov = [r for r in gw if r["decision"] == "overflow"]
    if ov or any(r.get("overflow_outcome") for r in gw):
        sent = [r for r in ov if r.get("status") == 200]
        ot = [r["ttft_measured"] for r in sent if r.get("ttft_measured") is not None]
        ct = [r["ttft_measured"] for r in m if r["tenant"] == "interactive"]
        out["overflow"] = {
            "sent": len(sent),
            "failed": sum(1 for r in gw if r.get("overflow_outcome") == "failed"),
            "limited": sum(1 for r in gw if r.get("overflow_outcome") == "limited"),
            "interactive_over_cap_429": sum(1 for r in gw if r["decision"] == "shed" and r["tenant"] == "interactive"
                                            and r["reason"] == "over_cap"),
            "overflow_ttft_ms": {"p50": ms(pct(ot, 50)), "p95": ms(pct(ot, 95))},
            "cluster_interactive_ttft_ms": {"p50": ms(pct(ct, 50)), "p95": ms(pct(ct, 95))},
            "overflow_prompt_tokens_median": statistics.median([r["prompt_tokens"] for r in sent]) if sent else None,
        }

    # per mix (by time window of the traffic rows)
    mixes = {}
    for mix in sorted({r["mix"] for r in traffic}):
        rows = [r for r in traffic if r["mix"] == mix]
        lo, hi = min(r["ts"] for r in rows), max(r["ts"] + (r["total_s"] or 0) for r in rows)
        g = [r for r in gw if lo - 1 <= r["ts"] <= hi]
        gm = [r["ttft_measured"] for r in g if r["decision"] == "ok" and r.get("ttft_measured") is not None]
        mixes[mix] = {"sent": len(g), "admitted": sum(1 for r in g if r["decision"] == "ok"),
                      "shed": dict(Counter(r["reason"] for r in g if r["decision"] == "shed")),
                      "hopped": sum(1 for r in g if r["decision"] == "ok" and r.get("hopped")),
                      "gateway_ttft_p50_ms": ms(pct(gm, 50)), "gateway_ttft_p95_ms": ms(pct(gm, 95)),
                      "laptop_ttft_p50_ms": ms(pct([r["ttft_s"] for r in rows if r["status"] == 200], 50)),
                      "laptop_ttft_p95_ms": ms(pct([r["ttft_s"] for r in rows if r["status"] == 200], 95))}
    out["by_mix"] = mixes

    # per worker, from the scrape
    workers = {}
    for w in ("WorkerA", "WorkerB"):
        rows = [r for r in scrape if r.get("worker") == w and r.get("ok")]
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        d = lambda f: (last[f] or 0) - (first[f] or 0)
        dq = d("prefix_queries")
        workers[w] = {
            "max_running": max(r["running"] for r in rows), "max_waiting": max(r["waiting"] for r in rows),
            "max_kv_pct": round(max(r["kv_usage"] for r in rows) * 100, 1), "preemptions": d("preemptions"),
            "prefix_hit_pct": round(100 * d("prefix_hits") / dq, 1) if dq > 0 else None,
            "engine_ttft_mean_ms": ms(d("ttft_sum") / d("ttft_count")) if d("ttft_count") > 0 else None,
            "requests_seen": int(d("ttft_count")),
        }
    g_rows = [r["gateway"] for r in scrape if r.get("worker") == "gateway" and r.get("ok") and r.get("gateway")]
    for w in workers:
        qd = [g.get(f'gateway_queue_depth{{worker="{w}"}}', 0) for g in g_rows]
        fl = [g.get(f'gateway_in_flight{{worker="{w}"}}', 0) for g in g_rows]
        if qd:
            workers[w].update(gateway_queue_max=max(qd), gateway_queue_mean=round(statistics.mean(qd), 1),
                              gateway_in_flight_max=max(fl), gateway_in_flight_mean=round(statistics.mean(fl), 1))
    out["workers"] = workers
    proof = folder / "warm_proof.json"
    if proof.exists():
        out["warm_proof"] = {w: {"probes_ms": [ms(p["engine_ttft_s"]) for p in v["probes"]], "warm": v["warm"]}
                             for w, v in json.loads(proof.read_text())["workers"].items()}
    (folder / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
