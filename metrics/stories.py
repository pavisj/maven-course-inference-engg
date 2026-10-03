"""Pick one real request per exit from the gateway logs and write metrics/stories.md.

Run from anywhere:  .venv/bin/python metrics/stories.py

Pick rule (deterministic): in the run's gateway_requests.jsonl, take the first matching row
whose arrival (ts) is at least 30 seconds after the ts of the run's first row.
If no matching row is that late, take the first matching row at all.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WARMUP_S = 30.0

# Every run folder with a gateway log, in run order. Used by the 503 search.
RUNS = ["run_c", "run_d", "run_e", "run_f", "run_f2", "run_g", "run_h1", "run_h2",
        "run_i", "run_j", "run_l_guard", "run_l_abort", "run_m", "run_m2"]
# Every run folder, including the baselines. Used by the traffic-side 503 check.
ALL_RUNS = ["baseline_a", "baseline_b"] + RUNS

PROSE = {"baseline_a": "baseline A", "baseline_b": "baseline B", "run_c": "run C", "run_d": "run D",
         "run_e": "run E", "run_f": "run F", "run_f2": "run F2", "run_g": "run G", "run_h1": "run H1",
         "run_h2": "run H2", "run_i": "run I", "run_j": "run J", "run_l_guard": "run L guard",
         "run_l_abort": "run L abort", "run_m": "run M", "run_m2": "run M2"}

KV_FREE_LINE = ("kv_free on the normal engine: never fired, needs ≈140 sequences, cap is 32. "
                "It fired only in run F2 (pressure2 engine: 0.4 GPU memory, 64 seats).")


def load(run):
    path = HERE / run / "gateway_requests.jsonl"
    rows = []
    with path.open() as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                rows.append((n, json.loads(line)))
    return rows


def pick(run, match):
    """Return (line_number, row, late_enough) for the first match, or None if nothing matches."""
    rows = load(run)
    t0 = rows[0][1]["ts"]
    hits = [(n, r) for n, r in rows if match(r)]
    if not hits:
        return None
    for n, r in hits:
        if r["ts"] - t0 >= WARMUP_S:
            return n, r, True
    n, r = hits[0]
    return n, r, False


# ---- the path text ---------------------------------------------------------------------------

def num(v):
    return "–" if v is None else str(v)


def place_text(r):
    worker = r.get("worker")
    if not worker:
        return None
    if worker == "Overflow":  # stream_out overwrites worker with Overflow; hopped still says where place put it
        if r.get("hopped"):
            return f"place (hopped from {r.get('preferred')})"
        return f"place {r.get('preferred')} (home)"
    if r.get("hopped"):
        return f"place {worker} (hopped from {r.get('preferred')})"
    return f"place {worker} (home)"


def admit_detail(r, show_queue_state):
    bits = []
    if r.get("est_ttft") is not None:
        bits.append(f"est_ttft {r['est_ttft']} s")
    if show_queue_state:
        bits.append(f"queue_depth {num(r.get('queue_depth'))}")
        bits.append(f"in_flight {num(r.get('in_flight'))}")
    return f" ({', '.join(bits)})" if bits else ""


def path_text(r, show_queue_state=False):
    decision, reason, stage = r.get("decision"), r.get("reason"), r.get("stage")
    if decision == "guard":
        return f"guard {r.get('status')} {reason} (stops at the guard)"
    parts = ["guard ok"]
    place = place_text(r)
    if place:
        parts.append(place)
    detail = admit_detail(r, show_queue_state)
    if decision == "ok":
        parts.append(f"admit ok{detail}")
        parts.append(f"queue wait {num(r.get('queue_wait_s'))} s")
        parts.append("engine ok")
    elif decision == "aborted":
        parts.append(f"admit ok{detail}")
        parts.append(f"queue wait {num(r.get('queue_wait_s'))} s")
        parts.append(f"engine aborted ({reason})")
    elif decision == "overflow":
        parts.append(f"admit {reason}{detail}")
        parts.append(f"Overflow {r.get('overflow_outcome')} ({r.get('via')})")
    elif decision == "shed":
        if stage == "queue":
            parts.append(f"admit ok{detail}")
            parts.append(f"queue {r.get('status')} {reason}")
        else:
            parts.append(f"admit {r.get('status')} {reason}{detail}")
        if r.get("overflow_outcome"):
            parts.append(f"Overflow {r['overflow_outcome']} → {r.get('status')}")
    else:
        parts.append(f"{stage} {decision} {reason}")
    return " → ".join(parts)


# ---- the 503 / 529 search ---------------------------------------------------------------------

def mentions_5xx(r):
    text = " ".join(str(r.get(k)) for k in ("outcome", "reason"))
    return "503" in text or "529" in text


def retried(r):
    return bool(r.get("retried_once")) or bool(r.get("retried_from")) or mentions_5xx(r)


def failure_overflow(r):
    out = str(r.get("outcome") or "")
    return (r.get("decision") == "leave" or out.startswith("leave_") or out == "overflow_unavailable"
            or r.get("reason") == "engine_503" or r.get("status") in (503, 529))


def traffic_5xx_count():
    """Count 503 and 529 client statuses across every traffic file. Used to confirm 'never happened'."""
    total = 0
    for run in ALL_RUNS:
        for f in (HERE / run).glob("traffic_*.jsonl"):
            with f.open() as fh:
                for line in fh:
                    if line.strip() and str(json.loads(line).get("status")) in ("503", "529"):
                        total += 1
    return total


def test_note():
    """Grep tests for the 503 -> other worker -> Overflow path and say what exists."""
    src = (ROOT / "tests" / "test_overflow.py").read_text()
    tree = ast.parse(src)
    with_503 = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                and "503" in ast.get_source_segment(src, n)]
    path_markers = ("retried_from", "open_upstream", "overflow_url")
    covers_path = any(m in p.read_text() for p in (ROOT / "tests").glob("test_*.py") for m in path_markers)
    if covers_path:
        return "Tests that mention 503 in tests/test_overflow.py: " + ", ".join(f"`{n}`" for n in with_503) + "."
    closest = "test_classify_stay_and_leave"
    if closest not in with_503:
        return ("No test in tests/test_overflow.py covers the full path 503 → other worker → Overflow. "
                "Tests that mention 503: " + ", ".join(f"`{n}`" for n in with_503) + ".")
    return (f"Closest test: `{closest}` in tests/test_overflow.py. It checks only that status 503 and 529 "
            f"classify as leave. No test covers the full path 503 → other worker → Overflow.")


# ---- the exits --------------------------------------------------------------------------------

EXITS = [
    (1, "guard reject", "run_l_guard", lambda r: r.get("decision") == "guard", False),
    (2, "429 tenant_tokens", "run_i",
     lambda r: r.get("decision") == "shed" and r.get("reason") == "tenant_tokens", False),
    (3, "429 over_cap (batch)", "run_i",
     lambda r: r.get("decision") == "shed" and r.get("reason") == "over_cap" and r.get("tenant") == "batch", False),
    (4, "over_cap → Overflow sent", "run_i",
     lambda r: r.get("decision") == "overflow" and r.get("overflow_outcome") == "sent", False),
    (5, "over_cap → Overflow limited, 429", "run_i",
     lambda r: (r.get("decision") == "shed" and r.get("reason") == "over_cap"
                and r.get("tenant") == "interactive" and r.get("overflow_outcome") == "limited"), False),
    (6, "429 timeout_queue", "run_m2",
     lambda r: (r.get("decision") == "shed" and r.get("reason") == "timeout_queue"
                and r.get("stage") == "admit" and r.get("tenant") == "interactive"), True),
    (7, "placed on WorkerA at home", "run_i",
     lambda r: r.get("decision") == "ok" and r.get("worker") == "WorkerA" and r.get("hopped") is False, False),
    (8, "placed on WorkerB, hopped", "run_i",
     lambda r: r.get("decision") == "ok" and r.get("worker") == "WorkerB" and r.get("hopped") is True, False),
    (9, "retried_once (503 → other worker)", None, retried, False),
    (10, "503 → failure Overflow", None, failure_overflow, False),
    (11, "aborted", "run_l_abort", lambda r: r.get("decision") == "aborted", False),
    (12, "429 kv_free", "run_f2",
     lambda r: r.get("decision") == "shed" and r.get("reason") == "kv_free", False),
]

HEADER = ["exit", "id", "tenant", "chapter", "prompt tokens", "decision at each stage",
          "est_ttft s", "queue_wait_s", "ttft_measured s", "total_s", "final code", "run"]


def story_row(exit_no, name, run, n, r, show_queue_state):
    return [f"{exit_no}. {name}", f"`{run}:{n}`", r.get("tenant", "–"), num(r.get("chapter")),
            num(r.get("prompt_tokens")), path_text(r, show_queue_state), num(r.get("est_ttft")),
            num(r.get("queue_wait_s")), num(r.get("ttft_measured")), num(r.get("total_s")),
            num(r.get("status")), PROSE[run]]


def search_all_runs(match):
    for run in RUNS:
        found = pick(run, match)
        if found:
            return run, found
    return None


def main():
    rows, fallbacks = [], []
    never_text = "never happened: no worker answered 503 or 529 in runs C to M2"
    for exit_no, name, run, match, show_q in EXITS:
        if run is None:
            hit = search_all_runs(match)
            if hit is None:
                extra = test_note()
                if traffic_5xx_count():
                    extra += " The client logs do show a 503 or 529: check by hand."
                rows.append([f"{exit_no}. {name}", "none", "–", "–", "–", f"{never_text}. {extra}",
                             "–", "–", "–", "–", "–", "runs C to M2"])
                continue
            run, found = hit
        else:
            found = pick(run, match)
            if found is None:
                rows.append([f"{exit_no}. {name}", "none", "–", "–", "–",
                             f"no matching row in {PROSE[run]}", "–", "–", "–", "–", "–", PROSE[run]])
                continue
        n, r, late = found
        if not late:
            fallbacks.append(f"`{run}:{n}` (exit {exit_no})")
        rows.append(story_row(exit_no, name, run, n, r, show_q))

    out = ["# One real request per exit", "",
           "Each story is the first matching gateway row that arrived at least 30 seconds after the first row of its run. "
           "If no match was that late, it is the first match at all.", "",
           "| " + " | ".join(HEADER) + " |", "|" + "|".join("---" for _ in HEADER) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out += ["", KV_FREE_LINE, ""]
    if fallbacks:
        out.append("No match arrived 30 seconds after the first row, so the first match was used: "
                   + ", ".join(fallbacks) + ".")
    out.append("The aborted row shows code 200 because the gateway records 200 once the stream starts. "
               "The client left after that.")
    out.append("The guard row has no prompt tokens because the gateway records them only after the guard passes.")
    (HERE / "stories.md").write_text("\n".join(out) + "\n")
    print(f"wrote {HERE / 'stories.md'}")


if __name__ == "__main__":
    main()
