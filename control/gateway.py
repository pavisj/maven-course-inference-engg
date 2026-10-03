"""The thinnest gateway: guard -> admit -> forward to worker A. No placement and no queue (Step 4).

    python -m control.gateway
    python -m control.gateway --log metrics/run_c/gateway_requests.jsonl

Binds 127.0.0.1:8765. The tenant comes from the X-Tenant header.
- POST /v1/chat/completions  guard fail -> 400 {"reason"}; shed -> 429 {"reason", "retry_after_seconds"}
                             plus a Retry-After header; admitted -> streamed from worker A.
- GET  /metrics              Prometheus text (gateway_* series)
- GET  /health               200 with the snapshot age

Every request appends one row to the --log JSONL file: ts, tenant, decision (ok|guard|shed),
reason, prompt_tokens, est_ttft, ttft_measured (gateway receive -> first content token from
the worker), status.
"""

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from control.admit import TARGET, AdmitRequest, est_ttft, should_shed
from control.guard import TENANTS, count_prompt_tokens, inspect
from control.snapshot import SnapshotSource

ROOT = Path(__file__).resolve().parent.parent
HOST, PORT = "127.0.0.1", 8765
TENANT_HEADER = "X-Tenant"
SHED_REASONS = ("tenant_tokens", "timeout_queue", "kv_free", "over_cap")
GUARD_REASONS = ("bad_body", "unknown_tenant", "prompt_too_long", "max_tokens_too_large")


def first_content(line: str) -> bool:
    """True if this SSE line carries a non-empty content delta."""
    if not line.startswith("data: ") or line == "data: [DONE]":
        return False
    try:
        chunk = json.loads(line[len("data: "):])
    except json.JSONDecodeError:
        return False
    return any(c.get("delta", {}).get("content") for c in chunk.get("choices", []))


def create_app(workers: dict[str, str], log_path: Path) -> FastAPI:
    source = SnapshotSource(workers)
    target_url = workers[TARGET].rstrip("/") + "/v1/chat/completions"
    sheds = defaultdict(int)
    guard_rejects = defaultdict(int)
    requests_total = defaultdict(int)
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0),
                               limits=httpx.Limits(max_connections=256, max_keepalive_connections=128))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")

    def record(**row) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()

    @asynccontextmanager
    async def lifespan(app):
        count_prompt_tokens([{"role": "user", "content": "warm up"}])  # load the tokenizer before traffic
        await asyncio.to_thread(source.refresh)                       # first engine snapshot before traffic
        task = asyncio.create_task(source.run())
        yield
        task.cancel()
        await client.aclose()
        log.close()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0 = time.time()
        start = time.perf_counter()
        tenant = request.headers.get(TENANT_HEADER)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        base = {"ts": t0, "tenant": tenant}

        # 1. guard: the first no
        guard = inspect(payload, tenant)
        if not guard.ok:
            label = tenant if tenant in TENANTS else "unknown"
            guard_rejects[guard.reason] += 1
            requests_total[(label, "guard")] += 1
            record(**base, decision="guard", reason=guard.reason, prompt_tokens=guard.prompt_tokens or None,
                   max_tokens=guard.max_tokens or None, est_ttft=None, ttft_measured=None, status=guard.code)
            return JSONResponse({"reason": guard.reason}, status_code=guard.code)

        # 2. admit: no await between the decision and the charge, so the decision is atomic
        snap = source.current(t0)
        est = round(est_ttft(snap, TARGET), 3)
        shed, code, reason, retry = should_shed(AdmitRequest(tenant, guard.prompt_tokens, guard.max_tokens), snap)
        if shed:
            sheds[reason] += 1
            requests_total[(tenant, "shed")] += 1
            record(**base, decision="shed", reason=reason, prompt_tokens=guard.prompt_tokens,
                   max_tokens=guard.max_tokens, est_ttft=est, ttft_measured=None, status=code, retry_after=retry)
            return JSONResponse({"reason": reason, "retry_after_seconds": retry}, status_code=code,
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
        source.tenants.charge(tenant, guard.prompt_tokens + guard.max_tokens, t0)
        source.record_prompt(guard.prompt_tokens, t0)
        requests_total[(tenant, "ok")] += 1
        payload["max_tokens"] = guard.max_tokens
        row = {**base, "decision": "ok", "reason": "ok", "prompt_tokens": guard.prompt_tokens,
               "max_tokens": guard.max_tokens, "est_ttft": est}

        # 3. forward to worker A
        source.in_flight += 1
        if not payload.get("stream"):
            try:
                r = await client.post(target_url, json=payload)
            except httpx.HTTPError as e:
                source.in_flight -= 1
                record(**row, ttft_measured=None, status=502, outcome=f"upstream_error: {type(e).__name__}")
                return JSONResponse({"reason": "upstream_error"}, status_code=502)
            source.in_flight -= 1
            record(**row, ttft_measured=round(time.perf_counter() - start, 4), status=r.status_code, outcome="ok")
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

        try:
            upstream = await client.send(client.build_request("POST", target_url, json=payload), stream=True)
        except httpx.HTTPError as e:
            source.in_flight -= 1
            record(**row, ttft_measured=None, status=502, outcome=f"upstream_error: {type(e).__name__}")
            return JSONResponse({"reason": "upstream_error"}, status_code=502)
        if upstream.status_code != 200:
            body = await upstream.aread()
            await upstream.aclose()
            source.in_flight -= 1
            record(**row, ttft_measured=None, status=upstream.status_code, outcome="upstream_status")
            return Response(body, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

        async def relay():
            ttft = None
            outcome = "ok"
            try:
                async for line in upstream.aiter_lines():
                    if ttft is None and first_content(line):
                        ttft = round(time.perf_counter() - start, 4)
                    yield line + "\n"
            except asyncio.CancelledError:
                outcome = "client_gone"       # closing upstream below makes vLLM abort the request and free its KV
                raise
            except httpx.HTTPError as e:
                outcome = f"upstream_error: {type(e).__name__}"
            finally:
                await upstream.aclose()
                source.in_flight -= 1
                record(**row, ttft_measured=ttft, status=200, outcome=outcome,
                       total_s=round(time.perf_counter() - start, 4))

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.get("/metrics")
    async def metrics():
        now = time.time()
        snap = source.current(now)
        lines = ["# TYPE gateway_sheds_total counter"]
        lines += [f'gateway_sheds_total{{reason="{r}"}} {sheds[r]}' for r in SHED_REASONS]
        lines += ["# TYPE gateway_guard_rejects_total counter"]
        lines += [f'gateway_guard_rejects_total{{reason="{r}"}} {guard_rejects[r]}' for r in GUARD_REASONS]
        lines += ["# TYPE gateway_requests_total counter"]
        lines += [f'gateway_requests_total{{tenant="{t}",decision="{d}"}} {n}' for (t, d), n in sorted(requests_total.items())]
        lines += ["# TYPE gateway_queue_depth gauge", f"gateway_queue_depth {snap.queue_depth}"]
        lines += ["# TYPE gateway_in_flight gauge", f"gateway_in_flight {source.in_flight}"]
        lines += ["# TYPE gateway_tenant_tokens_60s gauge"]
        lines += [f'gateway_tenant_tokens_60s{{tenant="{t}"}} {snap.tenant_tokens_60s.get(t, 0)}' for t in TENANTS]
        lines += ["# TYPE gateway_est_ttft_seconds gauge", f"gateway_est_ttft_seconds {est_ttft(snap, TARGET):.3f}"]
        lines += ["# TYPE gateway_avg_uncached_prompt_tokens gauge",
                  f"gateway_avg_uncached_prompt_tokens {snap.avg_uncached_prompt_tokens:.1f}"]
        lines += ["# TYPE gateway_snapshot_age_seconds gauge", f"gateway_snapshot_age_seconds {now - source.last_refresh:.2f}"]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/health")
    async def health():
        return {"ok": True, "target": TARGET, "snapshot_age_s": round(time.time() - source.last_refresh, 2)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--workers", type=Path, default=ROOT / "cluster" / "workers.json", help="JSON {name: url}")
    parser.add_argument("--log", type=Path, default=ROOT / "metrics" / "gateway_requests.jsonl")
    args = parser.parse_args()
    app = create_app(json.loads(args.workers.read_text()), args.log)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
