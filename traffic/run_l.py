"""Run L: requests that must die at the guard, and clients that leave mid-stream.

    python -m traffic.run_l guard --out-dir metrics/run_l_guard
    python -m traffic.run_l abort --out-dir metrics/run_l_abort

Both parts need the gateway on 127.0.0.1:8765 (started with --out-dir = the same folder).

guard: 12 requests, 4 per reason, that the guard must reject with 400 before any engine call:
       prompt_too_long (about 9,000 tokens), unknown_model ("gpt-4o"), max_tokens_too_large (2,048 > 512).
       Proof of 0 engine calls: each worker's vllm:e2e_request_latency_seconds_count before and after.
abort: 20 interactive requests at once over the chapters, each asking for a long answer. The client
       closes 10 of them at 1.0 s. Scrape every 1 s for 30 s: vLLM running, waiting, KV usage, aborted
       requests, and the gateway's in-flight count per worker. Writes one plot: running and KV % over time.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from app.prompt import build_messages
from app.retrieve import retrieve
from control.scrape import parse
from traffic.mixes import unique

ROOT = Path(__file__).resolve().parent.parent
GATEWAY = "http://127.0.0.1:8765"
LONG = " Answer in detail, in about 400 words."
CLOSE_AT_S = 1.0
N_ABORT, N_ALL = 10, 20
SCRAPE_S, SCRAPE_EVERY_S, LEAD_S = 30.0, 1.0, 3.0


def workers() -> dict[str, str]:
    return json.loads((ROOT / "cluster" / "workers.json").read_text())


def tutor_request(question: str, max_tokens: int = 512, model: str = "tutor") -> tuple[dict, dict]:
    chunks = retrieve(question)
    chapter = chunks[0]["filename"].split("-", 1)[0] if chunks else ""
    payload = {"model": model, "messages": build_messages(question, chunks), "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    return payload, {"X-Tenant": "interactive", "X-Chapter": chapter}


async def engine_counts(client: httpx.AsyncClient) -> dict:
    out = {}
    for name, url in workers().items():
        totals, by_label = parse((await client.get(f"{url}/metrics", timeout=10)).text)
        out[name] = {"requests": totals.get("vllm:e2e_request_latency_seconds_count", 0.0),
                     "prompt_tokens": totals.get("vllm:prompt_tokens_total", 0.0),
                     "aborted": by_label["vllm:request_success_total"]["finished_reason"].get("abort", 0.0)}
    return out


async def gateway_counts(client: httpx.AsyncClient) -> dict:
    totals, by_label = parse((await client.get(f"{GATEWAY}/metrics", timeout=10)).text)
    return {"guard_rejects": dict(by_label["gateway_guard_rejects_total"]["reason"]),
            "aborts": dict(by_label["gateway_aborts_total"]["stage"]),
            "in_flight": dict(by_label["gateway_in_flight"]["worker"])}


async def guard_part(out: Path) -> dict:
    q = "What is the difference between TTFT and TPOT?"
    cases = []
    long_payload, h = tutor_request(q)
    long_payload["messages"][-1]["content"] += " filler" * 9000
    cases += [("prompt_too_long", long_payload, h)] * 4
    cases += [("unknown_model", tutor_request(q, model="gpt-4o")[0], h)] * 4
    cases += [("max_tokens_too_large", tutor_request(q, max_tokens=2048)[0], h)] * 4
    async with httpx.AsyncClient(timeout=60) as client:
        e0, g0 = await engine_counts(client), await gateway_counts(client)
        rows = []
        for expect, payload, headers in cases:
            t = time.perf_counter()
            r = await client.post(f"{GATEWAY}/v1/chat/completions", json=payload, headers=headers)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            rows.append({"expect": expect, "status": r.status_code, "reason": body.get("reason"),
                         "ms": round((time.perf_counter() - t) * 1000, 1)})
        await asyncio.sleep(1)
        e1, g1 = await engine_counts(client), await gateway_counts(client)
    (out / "guard_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "sent": len(rows), "rejected_400": sum(r["status"] == 400 for r in rows),
        "matched_expected_reason": sum(r["reason"] == r["expect"] for r in rows),
        "by_reason": {k: sum(r["reason"] == k for r in rows) for k in ("prompt_too_long", "unknown_model", "max_tokens_too_large")},
        "gateway_guard_rejects_delta": {k: g1["guard_rejects"].get(k, 0) - g0["guard_rejects"].get(k, 0) for k in g1["guard_rejects"]},
        "engine_requests_delta": {w: e1[w]["requests"] - e0[w]["requests"] for w in e1},
        "engine_prompt_tokens_delta": {w: e1[w]["prompt_tokens"] - e0[w]["prompt_tokens"] for w in e1},
        "guard_ms_p50": sorted(r["ms"] for r in rows)[len(rows) // 2],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def poll(client: httpx.AsyncClient, t0: float, rows: list, stop: asyncio.Event):
    ws = workers()
    while not stop.is_set():
        tick = time.perf_counter()
        async def one(name, url):
            try:
                totals, by_label = parse((await client.get(f"{url}/metrics", timeout=3)).text)
                return {"worker": name, "running": totals.get("vllm:num_requests_running"),
                        "waiting": totals.get("vllm:num_requests_waiting"),
                        "kv_pct": round(100 * totals.get("vllm:kv_cache_usage_perc", 0.0), 3),
                        "aborted_total": by_label["vllm:request_success_total"]["finished_reason"].get("abort", 0.0)}
            except httpx.HTTPError as e:
                return {"worker": name, "error": type(e).__name__}
        res = await asyncio.gather(*(one(n, u) for n, u in ws.items()), gateway_counts(client), return_exceptions=True)
        t = round(time.perf_counter() - t0, 2)
        gw = res[-1] if isinstance(res[-1], dict) else {}
        for r in res[:-1]:
            if isinstance(r, dict):
                rows.append({"t": t, **r, "gateway_in_flight": gw.get("in_flight", {}).get(r["worker"]),
                             "gateway_aborts": gw.get("aborts")})
        await asyncio.sleep(max(0.0, SCRAPE_EVERY_S - (time.perf_counter() - tick)))


async def one_stream(client, i, payload, headers, close: bool, t_launch: float) -> dict:
    """Stream one request. close=True: a timer closes it at CLOSE_AT_S after send, whether or not a token came."""
    rec = {"i": i, "closed_by_client": close, "chapter": headers["X-Chapter"], "ttft_s": None, "chunks": 0, "status": None}
    start = time.perf_counter()

    async def consume():
        async with client.stream("POST", f"{GATEWAY}/v1/chat/completions", json=payload, headers=headers) as r:
            rec["status"] = r.status_code
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    rec["chunks"] += 1
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = round(time.perf_counter() - start, 3)

    if close:
        try:
            await asyncio.wait_for(consume(), timeout=CLOSE_AT_S)   # cancel = leave the stream = the client is gone
        except asyncio.TimeoutError:
            rec["closed_at_s"] = round(time.perf_counter() - start, 3)
    else:
        await consume()
    rec["end_s"] = round(time.perf_counter() - t_launch, 3)
    return rec


async def abort_part(out: Path) -> dict:
    gen = unique(seed=11)
    questions = [next(gen)["question"] + LONG for _ in range(N_ALL)]
    reqs = [tutor_request(q) for q in questions]
    rows, stop = [], asyncio.Event()
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=64)) as client:
        e0 = await engine_counts(client)
        t0 = time.perf_counter()
        poller = asyncio.create_task(poll(client, t0, rows, stop))
        await asyncio.sleep(LEAD_S)                      # a few idle scrapes before the burst
        t_launch = time.perf_counter()
        launch_s = round(t_launch - t0, 2)
        recs = await asyncio.gather(*(one_stream(client, i, p, h, i % 2 == 0, t_launch)
                                      for i, (p, h) in enumerate(reqs)))
        await asyncio.sleep(max(0.0, SCRAPE_S - (time.perf_counter() - t0)))
        stop.set()
        await poller
        e1 = await engine_counts(client)
    (out / "abort_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "scrape_1s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "requests": len(recs), "closed_by_client": sum(r["closed_by_client"] for r in recs),
        "launch_s": launch_s, "close_s": round(launch_s + CLOSE_AT_S, 2),
        "engine_aborted_delta": {w: e1[w]["aborted"] - e0[w]["aborted"] for w in e1},
        "engine_requests_delta": {w: e1[w]["requests"] - e0[w]["requests"] for w in e1},
        "engine_prompt_tokens_delta": {w: e1[w]["prompt_tokens"] - e0[w]["prompt_tokens"] for w in e1},
        "closed_at_s_max": max(r.get("closed_at_s", 0) for r in recs),
        "kept_end_s_max": max(r["end_s"] for r in recs if not r["closed_by_client"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, summary, ROOT / "plots" / "run_l_abort" / "running_and_kv.png")
    return summary


def plot(rows: list, summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for w, color in (("WorkerA", "#2c5bcb"), ("WorkerB", "#8a5a00")):
        rs = [r for r in rows if r.get("worker") == w and r.get("running") is not None]
        a1.step([r["t"] for r in rs], [r["running"] for r in rs], where="post", color=color, label=f"{w} vLLM running")
        a1.plot([r["t"] for r in rs], [r["gateway_in_flight"] or 0 for r in rs], ":", color=color, label=f"{w} gateway in flight")
        a2.plot([r["t"] for r in rs], [r["kv_pct"] for r in rs], color=color, label=f"{w} KV %")
    for ax in (a1, a2):
        ax.axvline(summary["launch_s"], color="#586273", ls="--", lw=1)
        ax.axvline(summary["close_s"], color="#b0392a", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    a1.set_ylabel("requests")
    a2.set_ylabel("KV cache usage (%)")
    a2.set_xlabel("seconds (grey: 20 requests sent; red: client closes 10 at 1.0 s)")
    fig.suptitle("Run L: 20 requests, 10 closed by the client at 1.0 s")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("guard", "abort"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(guard_part(args.out_dir) if args.part == "guard" else abort_part(args.out_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
