"""Count client requests, 200s, 429s by reason, 503s, guard 400s and aborts per run, and write metrics/counts.md.

Run from anywhere:  .venv/bin/python metrics/counts.py

Source rules:
- Client side: every metrics/<run>/traffic_*.jsonl file of the run (status may be an int or a string;
  the 429 reason is read from the "error" JSON string).
- Guard 400 and aborted: gateway_requests.jsonl, where the run has one.
- run_l_guard and run_l_abort have no traffic file. Every column comes from their gateway rows.
  In run_l_abort an aborted row is counted as aborted only, not as 200.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ["baseline_a", "baseline_b", "run_c", "run_d", "run_e", "run_f", "run_f2", "run_g", "run_h1", "run_h2",
        "run_i", "run_j", "run_l_guard", "run_l_abort", "run_m", "run_m2"]
REASONS = ["tenant_tokens", "over_cap", "timeout_queue", "kv_free"]


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def reason_of(row):
    err = row.get("error")
    try:
        if isinstance(err, str):
            err = json.loads(err)
        return (err or {}).get("reason") or "unparsed"
    except (ValueError, AttributeError):
        return "unparsed"


def tally(status_reason_pairs):
    """Turn (status, reason) pairs into one counts dict."""
    c = Counter()
    for status, reason in status_reason_pairs:
        status = int(status) if status is not None else None
        c["requests"] += 1
        if status == 200:
            c["200"] += 1
        elif status == 429:
            c["429"] += 1
            c["429:" + (reason if reason in REASONS else "other")] += 1
        elif status == 503:
            c["503"] += 1
        elif status == 400:
            c["400"] += 1
        else:
            c["other"] += 1
    return c


def client_counts(run):
    files = sorted((HERE / run).glob("traffic_*.jsonl"))
    if not files:
        return None
    pairs = []
    for f in files:
        for r in read_jsonl(f):
            pairs.append((r.get("status"), reason_of(r) if str(r.get("status")) == "429" else None))
    return tally(pairs)


def gateway_counts(run):
    path = HERE / run / "gateway_requests.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    guard = sum(1 for r in rows if r.get("decision") == "guard")
    aborted = sum(1 for r in rows if r.get("decision") == "aborted")
    # For the gateway-only runs: an aborted row counts as aborted only, so it is left out of the 200 count.
    pairs = [(r.get("status"), r.get("reason") if r.get("status") == 429 else None)
             for r in rows if r.get("decision") != "aborted"]
    c = tally(pairs)
    c["requests"] += aborted
    return c, guard, aborted


def main():
    table, warnings = [], []
    for run in RUNS:
        client = client_counts(run)
        gw = gateway_counts(run)
        if client is None:                      # gateway-only run
            c, guard, aborted = gw
            c["400"] = guard                    # a guard row is the 400
            guard_n, aborted_n = guard, aborted
        else:
            c = client
            guard_n = gw[1] if gw else c["400"]  # baselines: no gateway, so the client-side 400 count
            aborted_n = gw[2] if gw else None
            if gw and gw[0]["requests"] != c["requests"]:
                warnings.append(f"{run}: gateway rows {gw[0]['requests']} but client rows {c['requests']}")
            if gw:
                for k in ("200", "429", "429:tenant_tokens", "429:over_cap", "429:timeout_queue", "429:kv_free"):
                    if gw[0][k] != c[k]:
                        warnings.append(f"{run}: {k} gateway {gw[0][k]} but client {c[k]}")
        table.append((run, c, guard_n, aborted_n))

    show_other_429 = any(c["429:other"] for _, c, _, _ in table)
    show_other = any(c["other"] for _, c, _, _ in table)
    header = ["run", "client requests", "200", "429 total"] + [f"429 {r}" for r in REASONS]
    if show_other_429:
        header.append("429 other reason")
    header += ["503", "400 (guard)", "aborted"]
    if show_other:
        header.append("other status")

    def n(v):
        return "–" if v is None else f"{v:,}"

    lines = ["# Request counts per run", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for run, c, guard_n, aborted_n in table:
        cells = [f"`{run}`", n(c["requests"]), n(c["200"]), n(c["429"])] + [n(c[f"429:{r}"]) for r in REASONS]
        if show_other_429:
            cells.append(n(c["429:other"]))
        cells += [n(c["503"]), n(guard_n), n(aborted_n)]
        if show_other:
            cells.append(n(c["other"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    total_503 = sum(c["503"] for _, c, _, _ in table)
    lines.append("503: no run had one." if total_503 == 0 else f"503: {total_503:,} in total across the runs above.")
    lines.append("Client counts come from the traffic files. Run L guard and run L abort have none, so their counts come from the gateway rows.")
    lines.append("In run L abort, 10 requests were aborted and 10 ended ok. The aborted ones are not counted under 200.")
    lines.append("The baselines have no gateway. Their aborted cell shows – and their 400 cell is the client-side count.")
    (HERE / "counts.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'counts.md'}")
    for w in warnings:
        print("WARNING:", w)
    if not warnings:
        print("cross-check ok: gateway rows match client rows for every run that has both")


if __name__ == "__main__":
    main()
