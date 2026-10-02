# Maven Inference Engineering: final project

Step 1 is the cluster: two vLLM workers on Modal (`WorkerA`, `WorkerB`), each with its own stable URL.
Each worker serves `Qwen/Qwen2.5-3B-Instruct` as the model `tutor` on one A10 GPU and exposes `/health`, `/v1/chat/completions` and `/metrics`.
There is no gateway or app yet. See `DESIGN.md`.

## Cost

Both workers run with `min_containers=1`, so **both GPUs bill for as long as the app is deployed, even with zero traffic.**

- **GPU:** an A10 costs $0.000306 per second on Modal, which is $1.10 per hour. Two workers cost $2.20 per hour.
- **CPU and memory:** billed on top, by use. Expect roughly $0.15 to $0.65 per hour for both workers together.
- **Total:** about $2.35 to $2.85 per hour, or about $53 to $68 per day if left running.

Run step 7 when you finish.

## Commands, in order

Run every command from the repo root.

```bash
# 0. Laptop tools (pinned): a virtual environment with modal and httpx
uv venv --python 3.10 && source .venv/bin/activate && uv pip install -r requirements.txt

# 1. Log in to Modal (once per laptop)
modal setup

# 2. Deploy both workers. The first deploy builds the image and downloads the model.
modal deploy cluster/modal_vllm.py

# 3. Write the two stable worker URLs to cluster/workers.json
python cluster/write_workers.py

# 4. Smoke each worker, then save a /metrics scrape and the real metric names to metrics/
python cluster/smoke.py

# 5. Warm-up probe: the same request 5 times per worker, saved to metrics/warmup_<worker>.json
python cluster/warm_probe.py

# 6. GPU memory and model config for Part 1, saved to metrics/gpu_facts.json
modal run cluster/gpu_facts.py

# 7. Stop billing
modal app stop tutor-cluster
```

## FAST_BOOT

`FAST_BOOT = True` at the top of `cluster/modal_vllm.py` adds `--enforce-eager`.
That skips torch.compile and CUDA graph capture, so the first deploy boots fast.
After the first deploy works, set it to `False` and deploy again for real decode speed.
Take the Part 6 warm-up numbers with `FAST_BOOT = False`, because CUDA graphs are part of what "warm" means.

## Layout

```
app/        the application (empty in step 1)
control/    guard / admit / place / queue / hop (empty in step 1)
cluster/    how the engine comes up, plus laptop-side probes
plots/      plots from the cluster (empty in step 1)
metrics/    scrapes, warm-up results, GPU facts
notebook/   the Part 5 notebook (empty in step 1)
DESIGN.md   answers to Parts 1 to 8
```
