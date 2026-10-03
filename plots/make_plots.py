"""Plots from one run folder's JSONL files. One PNG per question.

    python plots/make_plots.py --run-dir metrics/rehearsal --label "rehearsal"

Reads <run-dir>/scrape_*.jsonl (control/scrape.py) and <run-dir>/traffic_*.jsonl
(traffic/run.py). Writes to --out-dir (default plots/):

- queue_depth.png             waiting and running requests over time, per worker
- kv_usage.png                KV cache usage over time, per worker
- prefix_hit_rate_by_mix.png  prefix-cache hits / queries inside each mix's time window
- ttft_by_mix.png             TTFT p50 and p95 per mix (laptop-measured)
- preemptions.png             cumulative preemptions over time, per worker

Each mix's time window (first request start to last request end) is shaded
on the time plots.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIX_ORDER = ["unique", "shared_chapter", "mixed"]


def load_jsonl(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))] if values else None


def mix_windows(traffic: list[dict]) -> dict[str, tuple[float, float]]:
    windows = {}
    for r in traffic:
        start, end = r["ts"], r["ts"] + (r["total_s"] or 0)
        lo, hi = windows.get(r["mix"], (start, end))
        windows[r["mix"]] = (min(lo, start), max(hi, end))
    return dict(sorted(windows.items(), key=lambda kv: kv[1][0]))


def title(text: str, label: str) -> str:
    return f"{text} ({label})" if label else text


def shade(ax, windows, t0):
    colors = ["#e8eef9", "#fbf1e1", "#e9f5ec", "#f6e8f3"]
    for i, (mix, (lo, hi)) in enumerate(windows.items()):
        ax.axvspan(lo - t0, hi - t0, color=colors[i % len(colors)], zorder=0)
        ax.text((lo + hi) / 2 - t0, 1.0, mix, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)


def time_series(scrape, field, ylabel, name, heading, label, windows, out_dir, scale=1.0, extra=None):
    ok = [r for r in scrape if r.get("ok") and r.get(field) is not None]
    if not ok:
        print(f"skip {name}: no {field} data")
        return
    t0 = min(r["ts"] for r in ok)
    fig, ax = plt.subplots(figsize=(10, 4))
    shade(ax, windows, t0)
    for worker in sorted({r["worker"] for r in ok}):
        rows = [r for r in ok if r["worker"] == worker]
        ax.plot([r["ts"] - t0 for r in rows], [r[field] * scale for r in rows], label=f"{worker} {field}")
        if extra:
            ax.plot([r["ts"] - t0 for r in rows], [(r.get(extra) or 0) * scale for r in rows],
                    linestyle="--", label=f"{worker} {extra}")
    ax.set_xlabel("seconds since first scrape")
    ax.set_ylabel(ylabel)
    ax.set_title(title(heading, label), pad=18)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / (name + '.png')}")


def counter_rows_around(rows, lo, hi):
    """Last row at or before lo, and first row at or after hi (else the nearest inside)."""
    before = [r for r in rows if r["ts"] <= lo]
    after = [r for r in rows if r["ts"] >= hi]
    inside = [r for r in rows if lo < r["ts"] < hi]
    first = before[-1] if before else (inside[0] if inside else None)
    last = after[0] if after else (inside[-1] if inside else None)
    return first, last


def prefix_hit_rate(scrape, windows, label, out_dir):
    names, rates = [], []
    for mix, (lo, hi) in windows.items():
        for worker in sorted({r["worker"] for r in scrape}):
            rows = [r for r in scrape if r["worker"] == worker and r.get("ok") and r.get("prefix_queries") is not None]
            first, last = counter_rows_around(rows, lo, hi)
            if first is None or last is None or first is last:
                continue
            dq = last["prefix_queries"] - first["prefix_queries"]
            dh = last["prefix_hits"] - first["prefix_hits"]
            if dq > 0:
                names.append(f"{mix}\n{worker}")
                rates.append(100 * dh / dq)
    if not rates:
        print("skip prefix_hit_rate_by_mix: no scrape rows inside any mix window")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, rates, color="#2457c5")
    ax.bar_label(bars, fmt="%.1f %%", fontsize=8)
    ax.set_ylabel("prefix-cache hit rate (% of prompt tokens)")
    ax.set_ylim(0, 100)
    ax.set_title(title("Prefix-cache hit rate per mix", label))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_hit_rate_by_mix.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / 'prefix_hit_rate_by_mix.png'}")


def ttft_by_mix(traffic, label, out_dir):
    mixes = [m for m in MIX_ORDER if any(r["mix"] == m for r in traffic)]
    mixes += sorted({r["mix"] for r in traffic} - set(mixes))
    p50s, p95s = [], []
    for mix in mixes:
        ttfts = [r["ttft_s"] * 1000 for r in traffic if r["mix"] == mix and r["status"] == 200 and r["ttft_s"]]
        p50s.append(percentile(ttfts, 50) or 0)
        p95s.append(percentile(ttfts, 95) or 0)
    if not mixes:
        print("skip ttft_by_mix: no traffic rows")
        return
    x = range(len(mixes))
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar([i - 0.2 for i in x], p50s, width=0.4, label="p50", color="#2457c5")
    b2 = ax.bar([i + 0.2 for i in x], p95s, width=0.4, label="p95", color="#9a5b00")
    ax.bar_label(b1, fmt="%.0f", fontsize=8)
    ax.bar_label(b2, fmt="%.0f", fontsize=8)
    ax.set_xticks(list(x), mixes)
    ax.set_ylabel("TTFT (ms, laptop-measured, includes network)")
    ax.set_title(title("TTFT p50 and p95 per mix", label))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ttft_by_mix.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / 'ttft_by_mix.png'}")


def est_vs_measured(gateway_rows, label, out_dir):
    rows = [r for r in gateway_rows if r.get("decision") == "ok" and r.get("est_ttft") is not None
            and r.get("ttft_measured") is not None]
    if not rows:
        print("skip est_vs_measured_ttft: no admitted gateway rows")
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {"WorkerA": "#2457c5", "WorkerB": "#9a5b00", "Overflow": "#2f7d4f"}
    for worker in sorted({r.get("worker") for r in rows}):
        pts = [r for r in rows if r.get("worker") == worker]
        ax.scatter([r["est_ttft"] for r in pts], [r["ttft_measured"] for r in pts], s=8, alpha=0.45,
                   color=colors.get(worker, "#666666"), label=f"{worker} ({len(pts)})")
    top = max(max(r["est_ttft"] for r in rows), max(r["ttft_measured"] for r in rows))
    ax.plot([0, top], [0, top], color="#888888", linestyle="--", linewidth=1, label="estimate = measured")
    ax.set_xlabel("est_ttft at the admit decision (s)")
    ax.set_ylabel("TTFT measured at the gateway, from arrival (s)")
    ax.set_title(title("Estimated vs measured TTFT", label))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "est_vs_measured_ttft.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / 'est_vs_measured_ttft.png'}")


def gateway_queue(scrape, windows, label, out_dir):
    rows = [r for r in scrape if r.get("worker") == "gateway" and r.get("ok") and r.get("gateway")]
    if not rows:
        print("skip gateway_queue_by_worker: no gateway scrape rows")
        return
    t0 = rows[0]["ts"]
    fig, ax = plt.subplots(figsize=(10, 4))
    shade(ax, windows, t0)
    for worker, color in (("WorkerA", "#2457c5"), ("WorkerB", "#9a5b00")):
        for series, style in (("gateway_queue_depth", "-"), ("gateway_in_flight", "--")):
            key = f'{series}{{worker="{worker}"}}'
            ax.plot([r["ts"] - t0 for r in rows], [r["gateway"].get(key, 0) for r in rows], style, color=color,
                    label=f"{worker} {'queue' if series.endswith('depth') else 'in flight'}")
    ax.set_xlabel("seconds since first scrape")
    ax.set_ylabel("requests")
    ax.set_title(title("Gateway queue depth and in flight, per worker", label), pad=18)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gateway_queue_by_worker.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / 'gateway_queue_by_worker.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "metrics" / "rehearsal")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--label", default="", help='added to every title, e.g. "rehearsal"')
    args = parser.parse_args()

    scrape = load_jsonl(sorted(args.run_dir.glob("scrape_*.jsonl")))
    traffic = load_jsonl(sorted(args.run_dir.glob("traffic_*.jsonl")))
    if not scrape and not traffic:
        raise SystemExit(f"no scrape_*.jsonl or traffic_*.jsonl in {args.run_dir}")
    print(f"{len(scrape)} scrape rows, {len(traffic)} traffic rows from {args.run_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = mix_windows(traffic)

    time_series(scrape, "waiting", "requests", "queue_depth", "Engine queue depth: waiting and running",
                args.label, windows, args.out_dir, extra="running")
    time_series(scrape, "kv_usage", "KV cache usage (%)", "kv_usage", "KV cache usage over time",
                args.label, windows, args.out_dir, scale=100)
    prefix_hit_rate(scrape, windows, args.label, args.out_dir)
    ttft_by_mix(traffic, args.label, args.out_dir)
    time_series(scrape, "preemptions", "preemptions (cumulative)", "preemptions", "Preemptions over time",
                args.label, windows, args.out_dir)
    gateway_rows = load_jsonl(sorted(args.run_dir.glob("gateway_requests.jsonl")))
    if gateway_rows:
        est_vs_measured(gateway_rows, args.label, args.out_dir)
        gateway_queue(scrape, windows, args.label, args.out_dir)


if __name__ == "__main__":
    main()
