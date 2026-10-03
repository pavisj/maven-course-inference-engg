"""Goodput across runs: requests answered within deadline per minute.

    python metrics/goodput.py

good = status 200 and first token within the deadline from arrival (interactive 6 s,
batch 60 s; the same deadlines admit and the queue use). Shed, failed or late = 0.
Arrival is when the client sent the request, so the client-side TTFT counts (it includes
the gateway, any queue wait and the network). Per run, per tenant and per worker.
Writes metrics/goodput.json and plots/goodput_vs_offered.png.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEADLINE_S = {"interactive": 6.0, "batch": 60.0}
RUNS = [  # (folder, label, users, through the gateway)
    ("baseline_a", "A", 8, False), ("baseline_b", "B", 48, False), ("run_c", "C", 48, True),
    ("run_d", "D", 48, True), ("run_e", "E", 48, True), ("run_f", "F", 64, True), ("run_g", "G", 96, True),
    ("run_f2", "F2", 128, True), ("run_h1", "H1", 48, True), ("run_h2", "H2", 48, True), ("run_i", "I", 96, True),
    ("run_j", "J", 96, True), ("run_m", "M", 64, True), ("run_m2", "M2", 96, True),
]


def load(folder: Path, pattern: str) -> list[dict]:
    return [json.loads(l) for p in sorted(folder.glob(pattern)) for l in p.read_text().splitlines() if l.strip()]


def good(row: dict) -> bool:
    return row.get("status") == 200 and row.get("ttft_s") is not None and row["ttft_s"] <= DEADLINE_S[row["tenant"]]


def run_goodput(folder: Path, label: str, users: int, gateway: bool) -> dict | None:
    traffic = load(folder, "traffic_*.jsonl")
    if not traffic:
        return None
    minutes = (max(r["ts"] + (r["total_s"] or 0) for r in traffic) - min(r["ts"] for r in traffic)) / 60
    out = {"run": label, "folder": folder.name, "users": users, "through_gateway": gateway,
           "minutes": round(minutes, 2), "offered_per_min": round(len(traffic) / minutes, 1),
           "answered_per_min": round(sum(1 for r in traffic if r["status"] == 200) / minutes, 1),
           "goodput_per_min": round(sum(1 for r in traffic if good(r)) / minutes, 1)}
    out["goodput_share"] = round(out["goodput_per_min"] / out["offered_per_min"], 3) if out["offered_per_min"] else None
    out["by_tenant"] = {}
    for t in sorted({r["tenant"] for r in traffic}):
        rows = [r for r in traffic if r["tenant"] == t]
        out["by_tenant"][t] = {"offered_per_min": round(len(rows) / minutes, 1),
                               "goodput_per_min": round(sum(1 for r in rows if good(r)) / minutes, 1)}
    # per worker: from the gateway log (where each admitted request went); baselines used WorkerA only
    gw = load(folder, "gateway_requests.jsonl")
    if gw:
        per = {}
        for r in gw:
            if r.get("decision") in ("ok", "overflow") and r.get("status") == 200:
                w = r.get("worker") or "WorkerA"
                ok = r.get("ttft_measured") is not None and r["ttft_measured"] <= DEADLINE_S.get(r["tenant"], 6.0)
                per[w] = per.get(w, 0) + int(ok)
        out["by_worker_goodput_per_min"] = {w: round(n / minutes, 1) for w, n in sorted(per.items())}
    else:
        out["by_worker_goodput_per_min"] = {"WorkerA": out["goodput_per_min"]}
    ttft = [r["ttft_s"] for r in traffic if r["status"] == 200 and r.get("ttft_s") is not None]
    out["late_answers"] = sum(1 for r in traffic if r["status"] == 200 and r.get("ttft_s") is not None
                              and r["ttft_s"] > DEADLINE_S[r["tenant"]])
    out["ttft_p50_ms"] = round(statistics.median(ttft) * 1000) if ttft else None
    out.update(wasted_engine_seconds(folder, gw))
    return out


def wasted_engine_seconds(folder: Path, gw: list[dict]) -> dict:
    """E2: engine time spent on answers whose first token came after the deadline (the user gave up).

    From the gateway log: status 200 and ttft_measured (from arrival) past the run's deadline.
    Engine-slot seconds = total_s - queue_wait_s - the network trip. The deadline is the run's own
    (admit_settings.json: run M and M2 used 3 s for interactive). Baselines have no gateway log.
    """
    if not gw:
        return {"late_gateway": None, "wasted_engine_seconds": None}
    deadline = dict(DEADLINE_S)
    settings = folder / "admit_settings.json"
    if settings.exists():
        deadline["interactive"] = json.loads(settings.read_text()).get("interactive_deadline_s", deadline["interactive"])
    trip_file = folder / "network_offset.json"
    trip_file = trip_file if trip_file.exists() else folder.parent / "network_offset.json"
    trip = json.loads(trip_file.read_text())["offset_s"] if trip_file.exists() else 0.0
    late = [r for r in gw if r.get("status") == 200 and r.get("ttft_measured") is not None and r.get("total_s") is not None
            and r.get("tenant") in deadline and r["ttft_measured"] > deadline[r["tenant"]]]
    secs = sum(max(0.0, r["total_s"] - (r.get("queue_wait_s") or 0.0) - trip) for r in late)
    return {"late_gateway": len(late), "wasted_engine_seconds": round(secs, 1)}


def main():
    rows = [r for r in (run_goodput(ROOT / "metrics" / f, l, u, g) for f, l, u, g in RUNS) if r]
    (ROOT / "metrics" / "goodput.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for r in rows:
        color = "#2457c5" if r["through_gateway"] else "#9a5b00"
        ax.scatter(r["offered_per_min"], r["goodput_per_min"], s=50, color=color)
        ax.annotate(f"{r['run']} ({r['users']} users)", (r["offered_per_min"], r["goodput_per_min"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    top = max(r["offered_per_min"] for r in rows)
    ax.plot([0, top], [0, top], linestyle="--", color="#888888", linewidth=1, label="goodput = offered")
    ax.scatter([], [], color="#9a5b00", label="direct to WorkerA (baselines)")
    ax.scatter([], [], color="#2457c5", label="through the gateway")
    ax.set_xlabel("offered load (requests per minute, retries included)")
    ax.set_ylabel("goodput (answered within deadline per minute)")
    ax.set_title("Goodput vs offered load, runs A to I")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "plots" / "goodput_vs_offered.png", dpi=120)
    print(f"{'run':4} {'users':>5} {'offered/min':>11} {'answered/min':>12} {'goodput/min':>11} {'share':>6} {'late':>5}")
    for r in rows:
        print(f"{r['run']:4} {r['users']:5} {r['offered_per_min']:11} {r['answered_per_min']:12} {r['goodput_per_min']:11} "
              f"{r['goodput_share']:6} {r['late_answers']:5}")


if __name__ == "__main__":
    main()
