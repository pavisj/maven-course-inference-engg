# DESIGN

## Part 1. Capacity on paper

## Part 2. Design the cluster

### 1. GPU

- **What we run:** one NVIDIA A10 per worker on Modal, two workers in total (`WorkerA`, `WorkerB` in `cluster/modal_vllm.py`).
- **Name:** Modal's GPU string is `"A10"`. Modal's docs no longer list `"A10G"`. This is the 24 GB Ampere card that AWS sells as the A10G.
- **Memory:** 24 GB GDDR6 per card, about 600 GB/s bandwidth (NVIDIA datasheet, not measured). This is GDDR6, not HBM. The brief's "HBM" in Part 1 means this 24 GB here.
- **Measured:** fill after `modal run cluster/gpu_facts.py`, from `metrics/gpu_facts.json`:
  - GPU name reported by CUDA: _fill after_
  - `torch.cuda.mem_get_info()` total: _fill after_ GiB
  - `torch.cuda.mem_get_info()` free before vLLM: _fill after_ GiB
- **vLLM's share:** `--gpu-memory-utilization 0.90`, so vLLM may claim 90 % of total memory for weights, activations and KV blocks.
- **Why this and not a smaller or cheaper one:** _to decide_. Points to weigh:
  - The weights take 6.17 GB, so a 16 GB card leaves under 10 GB for KV and activations.
  - Decode is limited by memory bandwidth, so the bandwidth of a cheaper 24 GB card matters as much as its size.
  - bf16 needs Ampere or newer. A T4 (Turing) would force fp16.

### 2. Model

- **Model:** `Qwen/Qwen2.5-3B-Instruct`, pinned to revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, served as `tutor`.
- **Parameters:** 3,085,938,688 (Hugging Face safetensors index), in bf16.
- **Weights:** 3,085,938,688 parameters × 2 bytes = 6.17 GB (5.75 GiB).
- **Config fields** (HF `config.json`, to be confirmed by `metrics/gpu_facts.json`):
  - `num_hidden_layers` = 36
  - `num_attention_heads` = 16 query heads
  - `num_key_value_heads` = 2 KV heads (grouped-query attention: 8 query heads share one KV head)
  - `hidden_size` = 2048, so `head_dim` = 2048 / 16 = 128
- **KV bytes per token:** 2 (K and V) × 36 layers × 2 KV heads × 128 head_dim × 2 bytes (bf16) = 36,864 bytes = 36 KiB per token.
- **KV per sequence at `--max-model-len 8192`:** 36 KiB × 8192 tokens = 288 MiB.
- **KV for all 32 decode slots at 8192 tokens:** 288 MiB × 32 = 9 GiB.
- **KV blocks vLLM actually allocated:** _fill after deploy_, from the worker log line "GPU KV cache size: N tokens".
- **App length:** _fill after the app exists_. Part 1 compares capacity at 8192 tokens with capacity at the length the app sends.

### 3. Topology

### 4. Concurrency and max length

### 5. Hop backend

### 6. Overflow

### Gateway and engine

### Security

- **Auth is a follow-up.** Both workers run with `unauthenticated=True` for now, so anyone with a worker URL can call it.
- **The gateway binds to localhost.** The app and the control plane listen only on localhost; only the engine URLs are on Modal.
- **Follow-up:** set `unauthenticated=False` and send a Modal Proxy Token from the gateway (`Authorization: Bearer wk-<id>.ws-<secret>`).

## Part 3. Guardrails, admit, stay vs leave

## Part 4. Place

## Part 5. Queue

## Part 6. Hop and warm-up

## Part 7. Wire the app to the cluster

## Part 8. Proof

## Submit
