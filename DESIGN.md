# DESIGN

## Part 1. Capacity on paper

_Draft. Inputs come from `metrics/gpu_facts.json` unless a line says otherwise. Every assumption is marked **verify**._

### Inputs

- `num_hidden_layers` = 36, `num_key_value_heads` = 2, `hidden_size` = 2048, `num_attention_heads` = 16 (`metrics/gpu_facts.json`)
- `mem_get_info` free = 23,437,180,928 bytes, total = 23,684,841,472 bytes (`metrics/gpu_facts.json`)
- Parameter count = 3,085,938,688 (Hugging Face safetensors index; not in `gpu_facts.json`, **verify**)
- `--gpu-memory-utilization` = 0.90, `--max-model-len` = 8192, `--max-num-seqs` = 32 (`cluster/modal_vllm.py`)

### KV bytes per token

- head_dim = hidden_size / num_attention_heads = 2048 / 16 = 128
- KV bytes per token = 2 (K and V) × num_hidden_layers × num_key_value_heads × head_dim × 2 bytes = 2 × 36 × 2 × 128 × 2 = 36,864 bytes = 36 KiB
- Model switch: none yet. Same model since step 1, so bytes per token have not changed.

### Weights

- Weights at FP16 = parameter count × 2 bytes = 3,085,938,688 × 2 = 6,171,877,376 bytes = 6.17 GB (5.75 GiB)
- Measured: vLLM logged 5.79 GiB for weights (`metrics/worker_startup_log.txt`).

### Free memory for KV

- Activations = 1 GB = 1,000,000,000 bytes (**verify**: a round assumption, not measured)
- Free for KV = (mem_get_info free − weights − activations) × 0.90 = (23,437,180,928 − 6,171,877,376 − 1,000,000,000) × 0.90 = 16,265,303,552 × 0.90 = 14,638,773,197 bytes = 14.64 GB (13.63 GiB)
- KV capacity = free for KV / KV bytes per token = 14,638,773,197 / 36,864 = 397,102 tokens
- Measured: vLLM allocated 13.67 GiB = 398,016 tokens (`metrics/worker_startup_log.txt`). vLLM applies 0.90 to total memory, then subtracts weights and its measured activation peak. This formula applies 0.90 after subtracting. The two land within 0.3 % (**verify** the cause if the gap grows).

### max_concurrent_seqs

`max_concurrent_seqs ≈ (HBM − weights − activations) / (kv_bytes_per_token × max_len)`

- At max_len 8192: 14,638,773,197 / (36,864 × 8192) = 14,638,773,197 / 301,989,888 = 48.47 sequences
  - KV per sequence = 36,864 × 8192 = 301,989,888 bytes = 288 MiB
  - Measured: vLLM logged 48.59 sequences at 8,192 tokens.
- At the app's length, 2,500 tokens (**verify**: assumed until `app/` exists; recomputed below in "Real prompt length"): 14,638,773,197 / (36,864 × 2,500) = 14,638,773,197 / 92,160,000 = 158.84 sequences
  - KV per sequence = 36,864 × 2,500 = 92,160,000 bytes = 87.9 MiB
- The engine cap is `--max-num-seqs` = 32 at both lengths. 32 sequences need 9.0 GiB of KV at 8192 tokens and 2.75 GiB at 2,500 tokens. Both fit in 13.63 GiB.

### Expected first limiter

- At 8192 tokens: prefill compute, because KV holds 48 full-length sequences while the engine runs at most 32, and each 8192-token prompt is a long prefill (**verify** under load).
- At 2,500 tokens: prefill compute, because KV holds 158 sequences, five times the 32-slot cap, so KV stays far from full while every request brings about 2,000 uncached prompt tokens to prefill (**verify** under load).
  - Measured hint: a warm cache miss of 495 prompt tokens took 53 to 84 ms of engine TTFT (`metrics/warmup_WorkerA_warm.json`, `metrics/warmup_WorkerB_warm.json`). That is roughly 6,000 to 9,000 prompt tokens per second per worker (**verify**: one request at a time, not under load).

## Part 2. Design the cluster

### 1. GPU

- **What we run:** one NVIDIA A10 per worker on Modal, two workers in total (`WorkerA`, `WorkerB` in `cluster/modal_vllm.py`).
- **Name:** Modal's GPU string is `"A10"`. Modal's docs no longer list `"A10G"`, so the prompt's `"A10G"` was not used.
- **Memory:** 24 GB GDDR6 per card, about 600 GB/s bandwidth (NVIDIA A10 datasheet, not measured). This is GDDR6, not HBM. The brief's "HBM" in Part 1 means this memory here.
- **Measured** (`modal run cluster/gpu_facts.py`, saved to `metrics/gpu_facts.json`):
  - GPU name reported by CUDA: `NVIDIA A10`
  - `torch.cuda.mem_get_info()` total: 22.06 GiB (23,684,841,472 bytes)
  - `torch.cuda.mem_get_info()` free before vLLM: 21.83 GiB (23,437,180,928 bytes)
  - Software: torch 2.11.0 built for CUDA 13.0, vLLM 0.21.0
- **vLLM's share:** `--gpu-memory-utilization 0.90`, so vLLM may claim 90 % of 22.06 GiB, which is 19.85 GiB.
  - Weights take 5.79 GiB and KV takes 13.67 GiB, which is 19.46 GiB together.
  - The other 0.39 GiB covers activations and the rest of vLLM's working memory.
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
- **Measured on the first deploy** (`metrics/worker_startup_log.txt`, identical on both workers):
  - Weights on the GPU: 5.79 GiB ("Model loading took 5.79 GiB"). The paper number is 5.75 GiB.
  - Memory left for KV: 13.67 GiB ("Available KV cache memory").
  - KV capacity: 398,016 tokens ("GPU KV cache size"). 13.67 GiB ÷ 36 KiB per token = about 398,000 tokens, so the 36 KiB per token figure holds.
  - Full-length sequences that fit: 48.59 at 8192 tokens each ("Maximum concurrency for 8,192 tokens per request").
- **First limiter at full length:** the decode slots, not KV. `--max-num-seqs 32` caps the batch at 32 sequences, but KV holds 48 full-length sequences.
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
