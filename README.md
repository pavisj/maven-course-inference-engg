# Tutor: an inference cluster with its own control plane

Maven Inference Engineering final project. A **tutor app** answers questions about the course's revision notes (RAG, Track A). It runs on **two vLLM workers on Modal** (Qwen2.5-3B-Instruct on one A10 each). A **gateway** on the laptop decides what enters, where it goes and what leaves: guard, admit, place, queue, hop, warm and overflow.

The full design, the numbers behind every constant, and the proof runs are in **[DESIGN.md](DESIGN.md)**. The analysis notebook is **[notebook/analysis.ipynb](notebook/analysis.ipynb)**.

## Architecture

```mermaid
flowchart LR
  app["Tutor app<br/>app/tutor.py<br/>BM25 top 5 + prompt<br/>X-Tenant, X-Chapter"] --> gw
  subgraph gw["Gateway · control/gateway.py · 127.0.0.1:8765"]
    direction LR
    guard["Guard<br/>400"] --> place["Place<br/>chapter home, hop"] --> admit["Admit<br/>429 or overflow"] --> queue["Queue<br/>32 seats per worker"] --> fwd["Forward<br/>stream"] --> gate["Overflow gate<br/>stay or leave"]
  end
  fwd --> A["WorkerA · A10 · vLLM<br/>home: 00, 03a, 03b, 05"]
  fwd --> B["WorkerB · A10 · vLLM<br/>home: 06, 06b, 07, 09, 10"]
  A <-. "hop = recompute" .-> B
  admit -. "capacity overflow (interactive over_cap)" .-> O["Overflow · A10<br/>Qwen2.5-1.5B, from zero<br/>10 per minute"]
  gate -. "failure overflow (503/529)" .-> O
  snap["Snapshot every 2 s<br/>vLLM /metrics + gateway counters"] -.-> admit
```

| Box | Code | What it decides |
|---|---|---|
| App | `app/` | retrieve 5 chunks (BM25), build the prompt, send `X-Tenant` and `X-Chapter` |
| Guard | `control/guard.py` | 400 for a bad body, an unknown tenant, a prompt that is too long, or too many answer tokens |
| Place | `control/place.py` | the chapter's home worker, unless it is busy or not warm (then a hop) |
| Admit | `control/admit.py` | 429 for `tenant_tokens`, `timeout_queue`, `kv_free`, `over_cap`; interactive `over_cap` may overflow |
| Queue | `control/queue.py` | 32 seats per worker; request 33 waits here, interactive first |
| Hop | `control/hop.py` | a record of every request placed away from home (the prefix is recomputed) |
| Warm | `cluster/warm.py` | a worker takes traffic only after 5 probes show it is warm |
| Overflow | `control/overflow.py` | 429/500/out-of-memory stay; 503/529 and capacity overflow leave to the 1.5B model |
| Engine | `cluster/modal_vllm.py` | vLLM 0.21.0: prefix caching, chunked prefill, 32 decode slots, 8,192 tokens per sequence |

## How to run it

From the repo root, Python 3.10:

```bash
# 0. Laptop tools (pinned)
uv venv --python 3.10 && source .venv/bin/activate && uv pip install -r requirements.txt
modal setup                                   # once per laptop

# 1. Corpus and chunks (the PDFs are not in git: put them in corpus/, see corpus/manifest.json)
python -m app.ingest

# 2. Start the cluster: WorkerA, WorkerB (always on) and Overflow (from zero)
modal deploy cluster/modal_vllm.py            # about 2 to 3 minutes until healthy
python cluster/write_workers.py               # writes cluster/workers.json and cluster/overflow.json

# 3. One question through the gateway
python -m control.gateway &                   # 127.0.0.1:8765; waits until both workers are warm
python -m app.tutor "Why does prefix caching save prefill time?" --base-url http://127.0.0.1:8765

# 4. A load run: gateway + scrape + traffic, all logs in metrics/<run>/
RUN=run_x MIXES=mixed USERS=96 DURATION=240 bash traffic/gateway_run.sh
python metrics/summarise_run.py metrics/run_x
python plots/make_plots.py --run-dir metrics/run_x --out-dir plots/run_x --label "run X"

# 5. Cross-run analysis
python metrics/goodput.py                     # metrics/goodput.json, plots/goodput_vs_offered.png
python metrics/whatif.py                      # metrics/whatif.json (first-order replay)
pytest tests/                                 # guard, admit, place, queue, overflow
jupyter lab notebook/analysis.ipynb

# 6. Stop billing (both A10s bill while deployed)
modal app stop tutor-cluster
```

Settings for `traffic/gateway_run.sh`: `RUN`, `MIXES` (`unique`, `shared_chapter`, `mixed`), `CHAPTER`, `USERS`, `DURATION`, `PLACE_POLICY` (`prefix_then_load`, `least_loaded`, `p2c`), `LOAD_BOUND`, `MAX_IN_FLIGHT`, `TENANT_SCALE`. Engine profiles for KV-pressure tests: `ENGINE_PROFILE=pressure` or `pressure2` before `modal deploy`.

## Where each run's data is

Each folder in `metrics/` holds the traffic log (`traffic_*.jsonl`, one row per client request), the scrape (`scrape_*.jsonl`, workers and gateway every 5 s), and for gateway runs the gateway log (`gateway_requests.jsonl`), hops (`hops.jsonl`), warm checks (`warm_events.jsonl`) and `summary.json`. Plots are in `plots/<run>/`.

| Run | Folder | What it tested |
|---|---|---|
| A | `metrics/baseline_a/` | 8 users, 3 mixes, straight to WorkerA: no queue |
| B | `metrics/baseline_b/` | 48 users, 3 mixes, straight to WorkerA: over the 32 slots |
| C | `metrics/run_c/` | Step 4 gateway: guard and admit, one worker |
| D | `metrics/run_d/` | Steps 5 to 8: placement and hops, 3 mixes, 48 users |
| E | `metrics/run_e/` | WorkerB cold start: warm proof and hops |
| F | `metrics/run_f/` | pressure engine (0.5 memory, 64 seats), 64 users |
| G | `metrics/run_g/` | 96 users after the fixes: the gateway queue |
| F2 | `metrics/run_f2/` | pressure2 engine (0.4 memory), 128 users: `kv_free` and preemptions |
| H1, H2 | `metrics/run_h1/`, `metrics/run_h2/` | `least_loaded` vs `prefix_then_load`: does prefix routing help? |
| I | `metrics/run_i/` | 96 users with capacity overflow to the 1.5B model |

Other files in `metrics/`: `gpu_facts.json` and `worker_startup_log.txt` (Part 1), `warm_proof.json`, `overflow_check.json`, `network_offset.json`, `goodput.json`, `whatif.json`, `metric_names.txt`.

## Layout

```
app/        the tutor: ingest, BM25 retrieve, prompt, CLI
control/    the gateway: guard, admit, place, queue, hop, overflow, snapshot, scrape
cluster/    the engine on Modal, warm checks, cold start, probes
traffic/    load generator (3 mixes) and run scripts
tests/      pytest for guard, admit, place, queue, overflow
metrics/    run data, summaries, goodput, what-if replay
plots/      plots per run
notebook/   analysis.ipynb, in the dashboard order
DESIGN.md   the design, Parts 1 to 8
```

## Cost

WorkerA and WorkerB keep one A10 each while deployed: about $2.20 per hour for the GPUs, $2.35 to $2.85 with CPU and memory. Overflow costs nothing until a request reaches it, and stops 5 minutes after the last one. Stop everything with `modal app stop tutor-cluster`.
