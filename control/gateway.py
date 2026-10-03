"""The gateway: guard -> place -> admit -> queue -> forward -> overflow gate (Steps 4 to 8).

    python -m control.gateway
    python -m control.gateway --out-dir metrics/run_d --policy prefix_then_load

Binds 127.0.0.1:8765. Headers from the app: X-Tenant (interactive | batch), X-Chapter (placement key).
1. guard     control/guard.py   400 {"reason"} on a bad request
2. place     control/place.py   pick the worker; no warm worker -> a 503 that may leave the cluster
3. admit     control/admit.py   429 {"reason", "retry_after_seconds"} + Retry-After, checked on the chosen worker
4. queue     control/queue.py   at most --max-in-flight per worker; past the deadline -> 429 timeout_queue
5. forward   stream from the worker; a client that disconnects closes the upstream, so vLLM frees the KV
6. overflow  control/overflow.py  429/500/out-of-memory stay; 503/529 try the other worker once, then Overflow

Files in --out-dir: gateway_requests.jsonl (one row per request), hops.jsonl, warm_events.jsonl.
GET /metrics: Prometheus text (gateway_* series). GET /health: warm state per worker.
"""

import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from cluster.warm import probe_worker
import control.admit as admit
from control.admit import AdmitRequest, est_ttft, overflow_instead, should_shed
from control.guard import TENANTS, count_prompt_tokens, inspect
from control.hop import HopLog
from control.overflow import (OVERFLOW_REASON_HEADER, STAY_RETRY_S, VIA_HEADER, VIA_NONE, VIA_OVERFLOW,
                              OverflowLimiter, classify)
from control.place import DEFAULT_POLICY, LOAD_BOUND, POLICIES, PlaceRequest, Shed, pick, views_from_snapshot
from control.queue import DEFAULT_MAX_IN_FLIGHT, WorkerQueue
from control.snapshot import SnapshotSource, TenantWindow

ROOT = Path(__file__).resolve().parent.parent
HOST, PORT = "127.0.0.1", 8765
SHED_REASONS = ("tenant_tokens", "timeout_queue", "kv_free", "over_cap")
GUARD_REASONS = ("bad_body", "unknown_tenant", "prompt_too_long", "max_tokens_too_large")
WARM_ROUNDS = 3              # probe rounds before a worker counts as "not warm yet"
WARM_RETRY_S = 30.0          # wait before the next warm attempt after WARM_ROUNDS fail


def first_content(line: str) -> bool:
    """True if this SSE line carries a non-empty content delta."""
    if not line.startswith("data: ") or line == "data: [DONE]":
        return False
    try:
        chunk = json.loads(line[len("data: "):])
    except json.JSONDecodeError:
        return False
    return any(c.get("delta", {}).get("content") for c in chunk.get("choices", []))


def create_app(workers: dict[str, str], out_dir: Path, overflow_url: str | None = None,
               policy: str = DEFAULT_POLICY, max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
               load_bound: int = LOAD_BOUND) -> FastAPI:
    source = SnapshotSource(workers, slots=max_in_flight)
    queues = {w: WorkerQueue(w, max_in_flight) for w in workers}
    source.queues = queues
    hops = HopLog(out_dir / "hops.jsonl")
    limiter = OverflowLimiter()
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0),
                               limits=httpx.Limits(max_connections=512, max_keepalive_connections=256))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "gateway_requests.jsonl").open("a")
    warm_log = (out_dir / "warm_events.jsonl").open("a")
    c = {k: defaultdict(int) for k in ("sheds", "guard", "requests", "aborts", "overflow", "placed")}
    warming: dict[str, asyncio.Task] = {}
    next_warm_try: dict[str, float] = {}

    def record(**row) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()

    # ---- warm management -------------------------------------------------------------
    async def warm_worker(name: str) -> None:
        for round_no in range(1, WARM_ROUNDS + 1):
            try:
                res = await asyncio.to_thread(probe_worker, workers[name])
            except Exception as e:                       # worker went away mid-probe
                res = {"warm": False, "error": f"{type(e).__name__}: {e}"[:200]}
            warm_log.write(json.dumps({"ts": time.time(), "worker": name, "round": round_no, **res}) + "\n")
            warm_log.flush()
            if res.get("warm"):
                source.warm[name] = True
                source.restarted.discard(name)
                return
        next_warm_try[name] = time.time() + WARM_RETRY_S

    def on_refresh() -> None:
        now = time.time()
        for name, st in source.state.items():
            busy = name in warming and not warming[name].done()
            if st.ok and not source.warm.get(name) and not busy and now >= next_warm_try.get(name, 0):
                warming[name] = asyncio.get_running_loop().create_task(warm_worker(name))

    @asynccontextmanager
    async def lifespan(app):
        count_prompt_tokens([{"role": "user", "content": "warm up"}])  # load the tokenizer before traffic
        await asyncio.to_thread(source.refresh)
        on_refresh()                                                  # start warm checks right away
        task = asyncio.create_task(source.run(on_refresh))
        yield
        task.cancel()
        for t in warming.values():
            t.cancel()
        await client.aclose()
        hops.close()
        log.close()
        warm_log.close()

    app = FastAPI(lifespan=lifespan)

    # ---- forwarding helpers --------------------------------------------------------------
    async def open_upstream(url: str, payload: dict):
        """Returns ("stream", response) for a 200, or (status, body bytes, headers)."""
        try:
            up = await client.send(client.build_request("POST", url.rstrip("/") + "/v1/chat/completions",
                                                        json=payload), stream=True)
        except httpx.HTTPError as e:
            return 503, f"upstream unreachable: {type(e).__name__}".encode(), {}   # unreachable counts as unavailable
        if up.status_code == 200:
            return "stream", up, {}
        body = await up.aread()
        headers = dict(up.headers)
        await up.aclose()
        return up.status_code, body, headers

    def stream_out(up, row: dict, start: float, on_done, via: str | None = None,
                   extra_headers: dict | None = None) -> StreamingResponse:
        async def relay():
            ttft, outcome, out_tokens = None, "ok", None
            try:
                async for line in up.aiter_lines():
                    if ttft is None and first_content(line):
                        ttft = round(time.perf_counter() - start, 4)
                    if '"usage"' in line and line.startswith("data: "):
                        try:
                            usage = json.loads(line[6:]).get("usage") or {}
                            out_tokens = usage.get("completion_tokens", out_tokens)
                        except json.JSONDecodeError:
                            pass
                    yield line + "\n"
            except asyncio.CancelledError:
                outcome = "client_gone"            # closing the upstream below makes vLLM abort and free the KV
                c["aborts"]["in_flight"] += 1
                raise
            except httpx.HTTPError as e:
                outcome = f"upstream_error: {type(e).__name__}"
            finally:
                await up.aclose()
                on_done(outcome, out_tokens)
                record(**row, ttft_measured=ttft, status=200, outcome=outcome, via=via, output_tokens=out_tokens,
                       total_s=round(time.perf_counter() - start, 4))
        headers = {VIA_HEADER: via} if via else {}
        headers.update(extra_headers or {})
        return StreamingResponse(relay(), media_type="text/event-stream", headers=headers)

    async def leave(payload: dict, row: dict, start: float, status: int, body: bytes):
        """Failure overflow: the cluster answered 503/529 (or no worker is warm). Send it to Overflow, under the limiter."""
        reason = "no_warm_worker" if row.get("reason") == "no_warm_worker" else "engine_503"
        if not overflow_url:
            c["overflow"][(reason, "failed")] += 1
            record(**row, status=503, outcome="leave_no_target", via=VIA_NONE)
            return Response(body or b"no leave target", status_code=503, headers={VIA_HEADER: VIA_NONE})
        if not limiter.allow():
            c["overflow"][(reason, "limited")] += 1
            record(**row, status=status, outcome="leave_limited")
            return Response(body, status_code=status)
        result = await open_upstream(overflow_url, payload)
        if result[0] == "stream":
            c["overflow"][(reason, "sent")] += 1
            return stream_out(result[1], {**row, "worker": "Overflow"}, start, lambda outcome, out: None, via=VIA_OVERFLOW)
        c["overflow"][(reason, "failed")] += 1
        record(**row, worker="Overflow", status=result[0], outcome="overflow_unavailable", via=VIA_OVERFLOW)
        return Response(result[1], status_code=result[0], headers={VIA_HEADER: VIA_OVERFLOW})

    # ---- the request path ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0 = time.time()
        start = time.perf_counter()
        tenant = request.headers.get("X-Tenant")
        chapter = request.headers.get("X-Chapter")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        row = {"ts": t0, "tenant": tenant, "chapter": chapter, "policy": policy, "load_bound": load_bound}

        # 1. guard
        guard = inspect(payload, tenant)
        if not guard.ok:
            c["guard"][guard.reason] += 1
            c["requests"][(tenant if tenant in TENANTS else "unknown", "guard")] += 1
            record(**row, decision="guard", reason=guard.reason, stage="guard", status=guard.code)
            return JSONResponse({"reason": guard.reason}, status_code=guard.code)
        row.update(prompt_tokens=guard.prompt_tokens, max_tokens=guard.max_tokens)
        payload["max_tokens"] = guard.max_tokens
        payload["stream"] = True                       # the gateway always streams from the worker
        payload["stream_options"] = {"include_usage": True}   # the last chunk reports real output tokens (refund)

        # 2. place (no state changes until admit passes)
        snap = source.current(t0)
        placement = pick(PlaceRequest(tenant, chapter), views_from_snapshot(snap), policy=policy, load_bound=load_bound)
        if isinstance(placement, Shed):
            c["requests"][(tenant, "leave")] += 1
            return await leave(payload, {**row, "decision": "leave", "reason": placement.reason, "stage": "place"},
                               start, placement.code, placement.reason.encode())
        worker = placement.chosen
        est = round(est_ttft(snap, worker), 3)
        row.update(preferred=placement.preferred, worker=worker, hopped=placement.hopped, est_ttft=est)

        # 3. admit, on the chosen worker
        shed, code, reason, retry = should_shed(AdmitRequest(tenant, guard.prompt_tokens, guard.max_tokens, worker), snap)
        overflow_outcome = None
        if shed and overflow_url and overflow_instead(tenant, reason):
            # capacity overflow (Step 12): the gateway itself is over capacity -> Overflow, under the limiter
            if limiter.allow():
                ocharge = source.tenants.charge(tenant, guard.prompt_tokens + guard.max_tokens, t0)  # windows still count it
                result = await open_upstream(overflow_url, payload)
                orow = {**row, "decision": "overflow", "reason": reason, "stage": "admit", "worker": "Overflow",
                        "overflow_outcome": "sent"}
                if result[0] == "stream":
                    c["overflow"][(reason, "sent")] += 1
                    c["requests"][(tenant, "overflow")] += 1

                    def refund(outcome, out_tokens, ocharge=ocharge):
                        if out_tokens is not None:
                            TenantWindow.refund(ocharge, guard.max_tokens - out_tokens)

                    return stream_out(result[1], orow, start, refund, via=VIA_OVERFLOW,
                                      extra_headers={OVERFLOW_REASON_HEADER: reason})
                TenantWindow.refund(ocharge, guard.prompt_tokens + guard.max_tokens)
                c["overflow"][(reason, "failed")] += 1
                overflow_outcome = "failed"           # Overflow could not answer: fall back to the 429
            else:
                c["overflow"][(reason, "limited")] += 1
                overflow_outcome = "limited"          # over 10 per minute: the old 429
        if shed:
            c["sheds"][reason] += 1
            c["requests"][(tenant, "shed")] += 1
            record(**row, decision="shed", reason=reason, stage="admit", status=code, retry_after=retry,
                   overflow_outcome=overflow_outcome)
            return JSONResponse({"reason": reason, "retry_after_seconds": retry}, status_code=code,
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
        charge = source.tenants.charge(tenant, guard.prompt_tokens + guard.max_tokens, t0)
        source.record_prompt(guard.prompt_tokens, t0)
        c["placed"][worker] += 1
        if placement.hopped:
            hops.record(t0, placement.preferred, worker, guard.prompt_tokens, chapter)

        # 4. queue
        q = queues[worker]
        ticket = q.enqueue(tenant, t0)
        waited = await q.wait(ticket, request.is_disconnected)
        row["queue_wait_s"] = round(time.time() - t0, 4)
        if waited == "timeout":
            retry = round(min(10.0, max(1.0, est - (ticket.deadline - t0) + 1)) * random.uniform(0.7, 1.3), 2)
            c["sheds"]["timeout_queue"] += 1
            c["requests"][(tenant, "shed")] += 1
            record(**row, decision="shed", reason="timeout_queue", stage="queue", status=429, retry_after=retry)
            return JSONResponse({"reason": "timeout_queue", "retry_after_seconds": retry}, status_code=429,
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
        if waited == "gone":
            c["aborts"]["queued"] += 1
            record(**row, decision="abort", reason="client_gone", stage="queue", status=499)
            return Response(status_code=499)
        c["requests"][(tenant, "ok")] += 1
        row.update(decision="ok", reason="ok", stage="engine")

        # 5. forward, 6. overflow gate
        def done(w, wq):
            def _done(outcome, out_tokens):
                wq.release()
                if outcome == "ok":
                    source.record_completion(w, time.time())
                if out_tokens is not None:             # refund the answer budget it did not use
                    TenantWindow.refund(charge, guard.max_tokens - out_tokens)
            return _done

        result = await open_upstream(workers[worker], payload)
        if result[0] == "stream":
            return stream_out(result[1], row, start, done(worker, q))
        q.release()
        TenantWindow.refund(charge, guard.max_tokens)  # no answer was produced
        status, body, headers = result
        verdict = classify(status, body)
        if verdict == "stay":
            retry = headers.get("retry-after", STAY_RETRY_S)
            record(**row, status=status, outcome="stay")
            return Response(body, status_code=status, headers={"Retry-After": str(retry)})
        if verdict == "other":
            record(**row, status=status, outcome="upstream_status")
            return Response(body, status_code=status)
        # leave: try the other worker once, still inside the cluster
        others = [w for w in workers if w != worker and source.warm.get(w) and source.state.get(w) and source.state[w].ok]
        if others:
            other = others[0]
            oq = queues[other]
            oq.in_flight += 1                          # a direct retry: takes a seat without queueing
            hops.record(time.time(), worker, other, guard.prompt_tokens, chapter)
            result = await open_upstream(workers[other], payload)
            if result[0] == "stream":
                return stream_out(result[1], {**row, "worker": other, "retried_from": worker}, start, done(other, oq))
            oq.release()
            status, body = result[0], result[1]
        return await leave(payload, {**row, "decision": "leave", "stage": "overflow"}, start, status, body)

    # ---- observability ---------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics():
        now = time.time()
        snap = source.current(now)
        L = ["# TYPE gateway_sheds_total counter"]
        L += [f'gateway_sheds_total{{reason="{r}"}} {c["sheds"][r]}' for r in SHED_REASONS]
        L += ["# TYPE gateway_guard_rejects_total counter"]
        L += [f'gateway_guard_rejects_total{{reason="{r}"}} {c["guard"][r]}' for r in GUARD_REASONS]
        L += ["# TYPE gateway_requests_total counter"]
        L += [f'gateway_requests_total{{tenant="{t}",decision="{d}"}} {n}' for (t, d), n in sorted(c["requests"].items())]
        L += ["# TYPE gateway_aborts_total counter"]
        L += [f'gateway_aborts_total{{stage="{s}"}} {c["aborts"][s]}' for s in ("queued", "in_flight")]
        L += ["# TYPE gateway_hops_total counter"]
        L += [f'gateway_hops_total{{src="{s}",dst="{d}"}} {n}' for (s, d), n in sorted(hops.counts.items())]
        L += [f"gateway_hops_all_total {sum(hops.counts.values())}"]
        L += ["# TYPE gateway_overflow_total counter"]
        L += [f'gateway_overflow_total{{reason="{r}",outcome="{o}"}} {c["overflow"][(r, o)]}'
              for r in ("over_cap", "engine_503", "no_warm_worker") for o in ("sent", "limited", "failed")]
        L += ["# TYPE gateway_placed_total counter"]
        L += [f'gateway_placed_total{{worker="{w}"}} {c["placed"][w]}' for w in workers]
        for name, metric in (("gateway_queue_depth", snap.queue_depth), ("gateway_in_flight", snap.in_flight),
                             ("gateway_completion_rate", snap.completion_rate)):
            L += [f"# TYPE {name} gauge"] + [f'{name}{{worker="{w}"}} {metric.get(w, 0):.3f}' for w in workers]
        L += ["# TYPE gateway_worker_warm gauge"] + [f'gateway_worker_warm{{worker="{w}"}} {int(snap.warm.get(w, False))}' for w in workers]
        L += ["# TYPE gateway_est_ttft_seconds gauge"] + [f'gateway_est_ttft_seconds{{worker="{w}"}} {est_ttft(snap, w):.3f}' for w in workers]
        L += ["# TYPE gateway_tenant_tokens_60s gauge"]
        L += [f'gateway_tenant_tokens_60s{{tenant="{t}"}} {snap.tenant_tokens_60s.get(t, 0)}' for t in TENANTS]
        L += ["# TYPE gateway_avg_uncached_prompt_tokens gauge", f"gateway_avg_uncached_prompt_tokens {snap.avg_uncached_prompt_tokens:.1f}"]
        L += ["# TYPE gateway_snapshot_age_seconds gauge", f"gateway_snapshot_age_seconds {now - source.last_refresh:.2f}"]
        return PlainTextResponse("\n".join(L) + "\n")

    @app.get("/health")
    async def health():
        return {"ok": True, "policy": policy, "load_bound": load_bound, "warm": dict(source.warm),
                "snapshot_age_s": round(time.time() - source.last_refresh, 2)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--workers", type=Path, default=ROOT / "cluster" / "workers.json", help="JSON {name: url}")
    parser.add_argument("--overflow", type=Path, default=ROOT / "cluster" / "overflow.json", help="JSON {Overflow: url}")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--policy", choices=POLICIES, default=os.environ.get("PLACE_POLICY", DEFAULT_POLICY),
                        help="placement policy; env PLACE_POLICY")
    parser.add_argument("--load-bound", type=int, default=int(os.environ.get("LOAD_BOUND", LOAD_BOUND)),
                        help="prefix_then_load: hop when the home has this many in flight; env LOAD_BOUND")
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT, help="seats per worker (the engine's --max-num-seqs)")
    parser.add_argument("--tenant-window-scale", type=float, default=1.0,
                        help="multiply both tenant windows (run F: a large value to isolate kv_free; run J: 2)")
    parser.add_argument("--over-cap-ttft", type=float, default=admit.OVER_CAP_TTFT_S,
                        help="interactive over_cap trigger in seconds (run I: 2; run J: 4)")
    parser.add_argument("--base-ttft", type=float, default=admit.BASE_TTFT_S,
                        help="prefill-backlog floor in the estimate, seconds (run I: 0.4; run J: 0 = removed)")
    args = parser.parse_args()
    settings = admit.apply_settings(args.over_cap_ttft, args.tenant_window_scale, args.base_ttft)
    print(f"admit settings: {settings}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "admit_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    overflow = json.loads(args.overflow.read_text()).get("Overflow") if args.overflow.exists() else None
    app = create_app(json.loads(args.workers.read_text()), args.out_dir, overflow, args.policy, args.max_in_flight,
                     args.load_bound)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
