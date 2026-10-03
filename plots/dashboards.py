"""Four dashboard PNGs for the gateway runs C to M2.

    .venv/bin/python plots/dashboards.py

Reads metrics/run_<x>/summary.json and scrape_run_<x>.jsonl. Writes to plots/:

- 02_success_failures.png   stacked bars of gateway decisions per run
- 03_sheds_by_reason.png    stacked bars of sheds by reason per run
- 04_router.png             placement, hop share and prefix hit rate per run
- 06_vllm.png               vLLM running, waiting and KV usage per worker (run J and run F2)

Run C has no summary.json. Its counts come from gateway_requests.jsonl and its prefix hit
rate from the scrape. Run C has no router data (no placement, no hops).
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "metrics"
OUT = ROOT / "plots"

RUNS = ["c", "d", "e", "f", "f2", "g", "h1", "h2", "i", "j", "m", "m2"]
LABELS = ["C", "D", "E", "F", "F2", "G", "H1", "H2", "I", "J", "M", "M2"]

BLUE = "#2c5bcb"  # WorkerA / interactive
AMBER = "#8a5a00"  # WorkerB / batch
RED = "#b5483f"  # muted red, sheds and hops
ROSE = "#dfaaa3"
SLATE = "#7b8794"
CHARCOAL = "#474d55"
LIGHT = "#c7ccd2"
INK = "#222222"
MUTED = "#5f6670"
GRID = "#e3e6ea"

DPI = 120
REASONS = ["tenant_tokens", "over_cap", "timeout_queue", "kv_free"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#9aa1a9",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
})


# ---------------------------------------------------------------- loading

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scrape_rows(run):
    return load_jsonl(METRICS / f"run_{run}" / f"scrape_run_{run}.jsonl")


def prefix_hit_pct(rows, worker):
    """Same method as metrics/summarise_run.py: last minus first, hits over queries."""
    w = [r for r in rows if r.get("worker") == worker and r.get("ok")]
    if len(w) < 2:
        return None
    dq = (w[-1]["prefix_queries"] or 0) - (w[0]["prefix_queries"] or 0)
    dh = (w[-1]["prefix_hits"] or 0) - (w[0]["prefix_hits"] or 0)
    return round(100 * dh / dq, 1) if dq > 0 else None


def load_run(run):
    """Return one dict per run with the fields the plots need. Missing data is None."""
    folder = METRICS / f"run_{run}"
    path = folder / "summary.json"
    notes = []
    if path.exists():
        s = json.loads(path.read_text())
        sent = s["requests_sent"]
        decisions = dict(s["decisions"])
        reasons = Counter()
        for key, n in s["sheds_by_reason_tenant"].items():
            reasons[key.split("/", 1)[1]] += n
        placed = s.get("placed_by_worker")
        hops = (s.get("hops") or {}).get("share_of_admitted")
        prefix = {w: v.get("prefix_hit_pct") for w, v in (s.get("workers") or {}).items()}
    else:
        notes.append("no summary.json: counts derived from gateway_requests.jsonl")
        reqs = load_jsonl(folder / "gateway_requests.jsonl")
        sent = len(reqs)
        decisions = dict(Counter(r["decision"] for r in reqs))
        reasons = Counter(r["reason"] for r in reqs if r["decision"] == "shed")
        hops = None  # no hops.jsonl for run C
        rows = scrape_rows(run)
        # no worker field in the run C gateway log: use the vLLM request_success delta per worker
        placed = {}
        for w in ("WorkerA", "WorkerB"):
            ws = [r for r in rows if r.get("worker") == w and r.get("ok")]
            placed[w] = int(ws[-1]["request_success"] - ws[0]["request_success"])
        prefix = {w: prefix_hit_pct(rows, w) for w in ("WorkerA", "WorkerB")}
    other = sent - decisions.get("ok", 0) - decisions.get("shed", 0) - decisions.get("overflow", 0)
    return {
        "sent": sent,
        "ok": decisions.get("ok", 0),
        "shed": decisions.get("shed", 0),
        "overflow": decisions.get("overflow", 0),
        "other": max(other, 0),
        "reasons": reasons,
        "placed": placed,
        "hop_pct": None if hops is None else 100 * hops,
        "prefix": prefix,
        "notes": notes,
    }


def style_axes(ax, ygrid=True):
    ax.tick_params(length=3)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def stacked(ax, x, series, colors, hatches=None, width=0.66):
    """Stacked bars with a thin white gap between segments."""
    bottom = np.zeros(len(x))
    handles = []
    for i, (name, values) in enumerate(series.items()):
        v = np.array(values, dtype=float)
        h = ax.bar(x, v, width, bottom=bottom, color=colors[i], edgecolor="white", linewidth=1.0,
                   label=name, hatch=(hatches[i] if hatches else None))
        handles.append(h)
        bottom += v
    return bottom


def label_totals(ax, x, totals, fmt="{:,.0f}", pad_frac=0.012):
    top = max(totals) if len(totals) else 1
    for xi, t in zip(x, totals):
        if t == 0:
            continue
        ax.text(xi, t + top * pad_frac, fmt.format(t), ha="center", va="bottom", fontsize=9, color=INK)


# ---------------------------------------------------------------- plots

def plot_success_failures(data):
    x = np.arange(len(RUNS))
    series = {
        "Answered (ok)": [d["ok"] for d in data],
        "Shed 429 (all reasons)": [d["shed"] for d in data],
        "Overflow sent": [d["overflow"] for d in data],
        "Other": [d["other"] for d in data],
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    tops = stacked(ax, x, series, [SLATE, RED, ROSE, LIGHT], hatches=[None, None, None, "///"])
    label_totals(ax, x, tops)
    ax.set_xticks(x, [l + ("*" if d["notes"] else "") for l, d in zip(LABELS, data)])
    ax.set_xlabel("Run (* = no summary.json, counts from the gateway request log)")
    ax.set_ylabel("Requests sent through the gateway (count)")
    ax.set_title("Success and failures per run", loc="left", fontsize=14, color=INK)
    ax.set_ylim(0, max(tops) * 1.08)
    style_axes(ax)
    ax.legend(loc="upper right", ncol=1, title="Gateway decision", title_fontsize=10)
    ovf = ", ".join(f"run {l} {d['overflow']}" for l, d in zip(LABELS, data) if d["overflow"])
    fig.text(0.01, 0.01, f"Number above each bar = requests sent. Run F and run M had no sheds. "
             f"Overflow sent (too thin to see): {ovf}. Other (zero in every run) = sent minus the rest.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "02_success_failures.png", dpi=DPI)
    plt.close(fig)


def plot_sheds_by_reason(data):
    x = np.arange(len(RUNS))
    series = {r: [d["reasons"].get(r, 0) for d in data] for r in REASONS}
    extra = [d["reasons"].keys() - set(REASONS) for d in data]
    unknown = set().union(*extra)
    if unknown:
        raise SystemExit(f"unexpected shed reasons: {unknown}")
    fig, ax = plt.subplots(figsize=(11, 5))
    tops = stacked(ax, x, series, [SLATE, RED, ROSE, CHARCOAL])
    label_totals(ax, x, tops)
    for xi, t in zip(x, tops):
        if t == 0:
            ax.text(xi, max(tops) * 0.012, "none", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(x, [l + ("*" if d["notes"] else "") for l, d in zip(LABELS, data)])
    ax.set_xlabel("Run (* = no summary.json, counts from the gateway request log)")
    ax.set_ylabel("Shed requests, 429 (count, interactive + batch)")
    ax.set_title("Sheds by reason per run", loc="left", fontsize=14, color=INK)
    ax.set_ylim(0, max(tops) * 1.08)
    style_axes(ax)
    ax.legend(loc="upper right", title="Shed reason", title_fontsize=10)
    fig.text(0.01, 0.01, "Reasons are summed over the interactive and batch tenants. "
             "Number above each bar = total sheds in that run.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "03_sheds_by_reason.png", dpi=DPI)
    plt.close(fig)


def plot_router(data):
    x = np.arange(len(RUNS))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    # top: placement
    w = 0.36
    pa = [d["placed"]["WorkerA"] if d["placed"] else np.nan for d in data]
    pb = [d["placed"]["WorkerB"] if d["placed"] else np.nan for d in data]
    ax1.bar(x - w / 2, pa, w, color=BLUE, edgecolor="white", linewidth=1.0, label="Placed on WorkerA")
    ax1.bar(x + w / 2, pb, w, color=AMBER, edgecolor="white", linewidth=1.0, label="Placed on WorkerB")
    top = np.nanmax(pa + pb)
    for xi, a, b in zip(x, pa, pb):
        ax1.text(xi - w / 2, a + top * 0.01, f"{a:,.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax1.text(xi + w / 2, b + top * 0.01, f"{b:,.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        if min(a, b) == 0:
            ax1.text(xi - w / 2, max(a, b) + top * 0.07, "WorkerA only", ha="center", va="bottom",
                     fontsize=8.5, color=MUTED)
    ax1.set_ylim(0, top * 1.12)
    ax1.set_ylabel("Admitted requests placed (count)")
    ax1.set_title("Router: placement, hops and prefix hits", loc="left", fontsize=14, color=INK)
    style_axes(ax1)
    ax1.legend(loc="upper right", ncol=2)

    # bottom: hop share and prefix hit rate, one axis (percent)
    hop = [np.nan if d["hop_pct"] is None else d["hop_pct"] for d in data]
    pha = [np.nan if d["prefix"].get("WorkerA") is None else d["prefix"]["WorkerA"] for d in data]
    phb = [np.nan if d["prefix"].get("WorkerB") is None else d["prefix"]["WorkerB"] for d in data]
    d_off = 0.12
    ax2.plot(x, hop, color=RED, marker="D", markersize=8, linewidth=1.4, alpha=0.95,
             markeredgecolor="white", markeredgewidth=1.0, label="Hop share of admitted requests")
    ax2.plot(x - d_off, pha, color=BLUE, marker="o", markersize=8, linewidth=1.4,
             markeredgecolor="white", markeredgewidth=1.0, label="Prefix hit rate, WorkerA")
    ax2.plot(x + d_off, phb, color=AMBER, marker="s", markersize=8, linewidth=1.4,
             markeredgecolor="white", markeredgewidth=1.0, label="Prefix hit rate, WorkerB")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Share (percent)")
    ax2.set_xlabel("Run (run C used WorkerA only: placement counted from vLLM, no hops, no WorkerB hit rate)")
    ax2.set_xticks(x, LABELS)
    style_axes(ax2)
    ax2.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.09)
    fig.savefig(OUT / "04_router.png", dpi=DPI)
    plt.close(fig)


def plot_vllm():
    cols = [("j", "run J (normal engine, 96 users)"), ("f2", "run F2 (pressure2 engine, 128 users)")]
    rows_spec = [("running", "Running requests (count)", 1.0), ("waiting", "Waiting requests (count)", 1.0),
                 ("kv_usage", "KV cache usage (percent)", 100.0)]
    colors = {"WorkerA": BLUE, "WorkerB": AMBER}
    series = {}  # (field, run, worker) -> (seconds, values)
    for run, _ in cols:
        rows = [r for r in scrape_rows(run) if r.get("worker") in colors and r.get("ok")]
        t0 = min(r["ts"] for r in rows)
        for field, _, scale in rows_spec:
            for worker in colors:
                pts = sorted((r["ts"] - t0, r[field] * scale) for r in rows
                             if r["worker"] == worker and r.get(field) is not None)
                series[(field, run, worker)] = ([p[0] for p in pts], [p[1] for p in pts])
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), sharex="col", sharey="row")
    for ri, (field, ylabel, _) in enumerate(rows_spec):
        ymax = max(max(series[(field, run, w)][1]) for run, _ in cols for w in colors)
        for c, (run, title) in enumerate(cols):
            ax = axes[ri][c]
            for worker, color in colors.items():
                xs, ys = series[(field, run, worker)]
                ax.plot(xs, ys, color=color, linewidth=1.6, label=worker)
            style_axes(ax)
            if c == 0:
                ax.set_ylabel(ylabel)
            if ri == 0:
                ax.set_title(title, fontsize=11, color=INK)
            if field == "kv_usage":
                ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
                ax.set_ylim(0, 108)
                if c == 1:
                    ax.text(2, 101.5, "KV full (100 percent)", fontsize=8.5, color=MUTED, va="bottom")
            else:
                ax.set_ylim(0, ymax * 1.08)
            if ri == 2:
                ax.set_xlabel("Seconds since first scrape (s)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("vLLM: running, waiting and KV per worker", x=0.01, y=0.995, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.005, "Each row shares one y axis across both columns, so the two engines compare directly.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(OUT / "06_vllm.png", dpi=DPI)
    plt.close(fig)


def main():
    data = [load_run(r) for r in RUNS]
    for lab, d in zip(LABELS, data):
        print(lab, "sent", d["sent"], "ok", d["ok"], "shed", d["shed"], "overflow", d["overflow"],
              "other", d["other"], "placed", d["placed"], "hop%", d["hop_pct"], "prefix", d["prefix"],
              d["notes"] or "")
    plot_success_failures(data)
    plot_sheds_by_reason(data)
    plot_router(data)
    plot_vllm()
    from matplotlib.image import imread
    for name in ["02_success_failures", "03_sheds_by_reason", "04_router", "06_vllm"]:
        p = OUT / f"{name}.png"
        img = imread(p)
        print(f"{p}: {p.stat().st_size} bytes, {img.shape[1]}x{img.shape[0]} px")


if __name__ == "__main__":
    main()
