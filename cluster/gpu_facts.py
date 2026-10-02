"""GPU and model facts for DESIGN.md Part 1 (capacity on paper).

Runs one short function on the same image and GPU as the workers:
    modal run cluster/gpu_facts.py

Prints and saves metrics/gpu_facts.json:
- the GPU name and torch.cuda.mem_get_info() (free and total bytes)
- the model's HF config fields that set KV bytes per token
"""

import json
from pathlib import Path

import modal

from modal_vllm import GPU, MODEL_NAME, MODEL_REVISION, VLLM_VERSION, hf_cache_vol, vllm_image

OUT = Path(__file__).resolve().parent.parent / "metrics" / "gpu_facts.json"
CONFIG_FIELDS = ["num_hidden_layers", "num_key_value_heads", "hidden_size", "num_attention_heads"]

app = modal.App("tutor-gpu-facts")


@app.function(
    image=vllm_image.add_local_python_source("modal_vllm"),
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=10 * 60,
)
def gpu_facts() -> dict:
    import torch
    from transformers import AutoConfig

    free_b, total_b = torch.cuda.mem_get_info()
    config = AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    return {
        "gpu_requested": GPU,
        "gpu_name": torch.cuda.get_device_name(0),
        "mem_get_info": {
            "free_bytes": free_b,
            "total_bytes": total_b,
            "free_gib": round(free_b / 2**30, 2),
            "total_gib": round(total_b / 2**30, 2),
        },
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "config": {field: getattr(config, field) for field in CONFIG_FIELDS},
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        # str(): torch.__version__ is a torch type the laptop cannot unpickle without torch
        "versions": {"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "vllm": VLLM_VERSION},
    }


@app.local_entrypoint()
def main():
    facts = gpu_facts.remote()
    print(json.dumps(facts, indent=2))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=2) + "\n")
    print(f"wrote {OUT}")
