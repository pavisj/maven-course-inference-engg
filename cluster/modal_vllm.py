"""Step 1: the cluster. Two vLLM workers on Modal, one stable URL each.

Each worker serves Qwen/Qwen2.5-3B-Instruct as the model name "tutor" and
exposes /health, /v1/chat/completions and /metrics. vLLM's scheduler is left
alone; we only set engine arguments.

Deploy:  modal deploy cluster/modal_vllm.py
Stop:    modal app stop tutor-cluster

Follows Modal's `@app.server` vLLM example:
https://modal.com/docs/examples/vllm_inference
"""

import modal

# True: --enforce-eager. No torch.compile, no CUDA graphs, so the boot is fast.
# Use it for the first deploy. Set it to False after that for real decode speed.
FAST_BOOT = False

APP_NAME = "tutor-cluster"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # HF main on 2026-10-03
SERVED_MODEL_NAME = "tutor"

VLLM_VERSION = "0.21.0"  # the version Modal's example is tested with
MINUTES = 60  # seconds
VLLM_PORT = 8000

# Modal's docs list "A10" and no longer list "A10G".
# CUDA reports the device as "NVIDIA A10", 22.06 GiB total (metrics/gpu_facts.json).
GPU = "A10"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Engine arguments: the only knobs we turn on vLLM. One comment per flag.
ENGINE_ARGS = [
    # clients ask for "tutor", not the HF repo id; the gateway can swap models later
    "--served-model-name", SERVED_MODEL_NAME,
    # longest prompt + answer one request may hold; caps KV per sequence
    "--max-model-len", "8192",
    # decode slots: at most 32 sequences in the running batch at once
    "--max-num-seqs", "32",
    # token budget per engine step; long prefills get chunked to fit it
    "--max-num-batched-tokens", "4096",
    # share of the 24 GB vLLM may claim for weights + activations + KV blocks
    "--gpu-memory-utilization", "0.90",
    # reuse KV blocks for a shared prefix such as the system prompt
    "--enable-prefix-caching",
    # take dtype from the model config (bfloat16 for Qwen2.5)
    "--dtype", "auto",
]
# eager mode skips compile and CUDA graph capture: faster boot, slower decode
ENGINE_ARGS += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]


def start_vllm():
    """Start `vllm serve` in a subprocess. Modal routes traffic once port 8000 listens."""
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        *ENGINE_ARGS,
    ]
    print(*cmd)
    return subprocess.Popen(cmd)


app = modal.App(APP_NAME)

# Both workers share these settings, so they stay identical except for their names.
# target_concurrency and max_concurrency are left unset on purpose:
# with one container per worker there is nothing to autoscale,
# and admission belongs to our gateway, not to Modal's proxy.
WORKER_SETTINGS = dict(
    image=vllm_image,
    gpu=GPU,
    min_containers=1,  # keep one container warm at all times (billed while deployed)
    max_containers=1,  # never add a second one: one worker = one GPU = one KV store
    scaledown_window=15 * MINUTES,  # how long an idle container stays up above min_containers
    startup_timeout=10 * MINUTES,  # how long Modal waits for the container to start
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    unauthenticated=True,  # public for now; auth is a follow-up (see DESIGN.md Part 2)
)


@app.server(**WORKER_SETTINGS)
class WorkerA:
    @modal.enter()
    def start(self):
        self.process = start_vllm()

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.server(**WORKER_SETTINGS)
class WorkerB:
    @modal.enter()
    def start(self):
        self.process = start_vllm()

    @modal.exit()
    def stop(self):
        self.process.terminate()
