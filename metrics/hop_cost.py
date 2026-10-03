"""Hop cost: TTFT for hopped vs not hopped requests, per placement policy, and the tokens each hop recomputes.

    python metrics/hop_cost.py

Reads metrics/<run>/gateway_requests.jsonl and hops.jsonl for runs D to L. Runs D to G have no
"policy" field: the gateway then always used prefix_then_load (LOAD_BOUND 22).
Tokens recomputed per hop = prompt tokens - 431 shared system-prompt tokens (Part 1): on Modal
nothing is copied, so the destination prefills everything but the system prompt, which every
worker already holds in its prefix cache.
Writes metrics/hops.json (summary + one record per hop) and plots/07_hops.png.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHARED_TOKENS = 431
RUNS = [("run_d", "D"), ("run_e", "E"), ("run_f", "F"), ("run_f2", "F2"), ("run_g", "G"), ("run_h1", "H1"),
        ("run_h2", "H2"), ("run_i", "I"), ("run_j", "J"), ("run_l_abort", "L")]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def stats(v: list[float]) -> dict:
    return {"n": len(v), "ttft_p50_ms": round(pct(v, 50) * 1000) if v else None,
            "ttft_p95_ms": round(pct(v, 95) * 1000) if v else None}


def main():
    by_policy: dict[str, dict[str, list[float]]] = {}
    by_run, records = {}, []
    for folder, label in RUNS:
        rows = load(ROOT / "metrics" / folder / "gateway_requests.jsonl")
        ok = [r for r in rows if r.get("decision") == "ok" and r.get("ttft_measured") is not None
              and r.get("worker") in ("WorkerA", "WorkerB")]
        policy = next((r["policy"] for r in rows if r.get("policy")), None) or "prefix_then_load"
        hop = [r["ttft_measured"] for r in ok if r.get("hopped")]
        home = [r["ttft_measured"] for r in ok if not r.get("hopped")]
        d = by_policy.setdefault(policy, {"hopped": [], "home": []})
        d["hopped"] += hop
        d["home"] += home
        hops = load(ROOT / "metrics" / folder / "hops.jsonl")
        for h in hops:
            records.append({"run": label, "src": h["src"], "dst": h["dst"], "tokens": max(0, h["tokens"] - SHARED_TOKENS),
                            "prompt_tokens": h["tokens"], "prefix": h.get("prefix"), "backend": h.get("backend", "recompute")})
        rec = [max(0, h["tokens"] - SHARED_TOKENS) for h in hops]
        by_run[label] = {"policy": policy, "hopped": stats(hop), "home": stats(home), "hops": len(hops),
                         "tokens_recomputed_total": sum(rec),
                         "tokens_recomputed_per_hop_p50": round(statistics.median(rec)) if rec else None}
    summary = {p: {"hopped": stats(v["hopped"]), "home": stats(v["home"])} for p, v in by_policy.items()}
    out = {"method": __doc__.strip().splitlines()[0], "shared_system_tokens": SHARED_TOKENS,
           "by_policy": summary, "by_run": by_run, "hops_total": len(records),
           "tokens_recomputed_total": sum(r["tokens"] for r in records),
           "tokens_recomputed_per_hop_p50": round(statistics.median(r["tokens"] for r in records)), "hops": records}
    (ROOT / "metrics" / "hops.json").write_text(json.dumps(out, indent=1) + "\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
    pols = list(summary)
    x = range(len(pols))
    for k, (kind, color) in enumerate((("home", "#2c5bcb"), ("hopped", "#8a5a00"))):
        p50 = [summary[p][kind]["ttft_p50_ms"] for p in pols]
        p95 = [summary[p][kind]["ttft_p95_ms"] for p in pols]
        xs = [i + (k - 0.5) * 0.38 for i in x]
        a1.bar(xs, p95, 0.36, color=color, alpha=0.35, label=f"{kind} p95")
        a1.bar(xs, p50, 0.36, color=color, label=f"{kind} p50")
        for xi, v in zip(xs, p95):
            a1.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(list(x), [f"{p}\n(n hop {summary[p]['hopped']['n']:,})" for p in pols], fontsize=8)
    a1.set_ylabel("gateway TTFT (ms)")
    a1.set_title("All runs, by policy")
    a1.legend(fontsize=8)
    labels = [l for _, l in RUNS if by_run[l]["hopped"]["n"] or by_run[l]["home"]["n"]]
    xr = range(len(labels))
    for k, (kind, color) in enumerate((("home", "#2c5bcb"), ("hopped", "#8a5a00"))):
        xs = [i + (k - 0.5) * 0.38 for i in xr]
        a2.bar(xs, [by_run[l][kind]["ttft_p50_ms"] or 0 for l in labels], 0.36, color=color, label=f"{kind} p50")
        a2.scatter(xs, [by_run[l][kind]["ttft_p95_ms"] or 0 for l in labels], color=color, marker="_", s=180, label=f"{kind} p95")
    a2.set_xticks(list(xr), [f"{l}\n{by_run[l]['hops']} hops" for l in labels], fontsize=8)
    a2.set_ylabel("gateway TTFT (ms)")
    a2.set_title("Per run (H1 = least_loaded; the rest prefix_then_load)")
    a2.legend(fontsize=8, ncol=2)
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"07 · Hops: {len(records):,} hops, {out['tokens_recomputed_per_hop_p50']:,} tokens recomputed per hop (median)")
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "07_hops.png", dpi=120)
    print(json.dumps({k: v for k, v in out.items() if k != "hops"}, indent=1))


if __name__ == "__main__":
    main()
