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

Main line: activations measured on the A10 we ran. Conservative case: the 1 GB assumption, which leaves less memory for KV.

- **Activations, measured: 0.39 GiB** = 418,759,311 bytes. vLLM's budget is 0.90 × 22.06 GiB = 19.85 GiB. It logged 5.79 GiB of weights and 13.67 GiB of KV. 19.85 − 5.79 − 13.67 = 0.39 GiB is left for activations and working memory (`metrics/gpu_facts.json`, `metrics/worker_startup_log.txt`).
- **Main: free for KV = 0.90 × mem_get_info total − weights − activations** = 0.90 × 23,684,841,472 − 6,171,877,376 − 418,759,311 = 21,316,357,325 − 6,171,877,376 − 418,759,311 = 14,725,720,637 bytes = 14.73 GB (13.71 GiB)
  - KV capacity = 14,725,720,637 / 36,864 = **399,461 tokens**
  - Check: vLLM allocated 398,016 tokens. The main line is 1,445 tokens (0.36 %) above it. About 1,170 tokens of that gap are the 0.04 GiB between paper weights (5.75 GiB) and logged weights (5.79 GiB); rounding in the logged GiB values covers the rest.
  - Why this order: vLLM applies 0.90 to total memory first, then subtracts weights and activations. The Task 1 order, (free − weights − activations) × 0.90, gives 411,293 tokens with 0.39 GiB: 3.3 % above what vLLM allocated.
- **Conservative case, 1 GB assumed:** free for KV = (mem_get_info free − weights − 1 GB) × 0.90 = (23,437,180,928 − 6,171,877,376 − 1,000,000,000) × 0.90 = 14,638,773,197 bytes = 14.64 GB (13.63 GiB); KV capacity = 14,638,773,197 / 36,864 = 397,102 tokens

### max_concurrent_seqs

`max_concurrent_seqs ≈ (HBM − weights − activations) / (kv_bytes_per_token × max_len)` = KV capacity / max_len

| max_len | Why this length | Main (399,461 tokens) | Conservative (397,102 tokens) | KV per sequence | KV for 32 sequences |
|---|---|---|---|---|---|
| 8,192 tokens | `--max-model-len` | 48.76 | 48.47 | 288.0 MiB | 9.00 GiB |
| 2,500 tokens | first assumption, before `app/` | 159.78 | 158.84 | 87.9 MiB | 2.75 GiB |
| 2,660 tokens | interactive median: 2,148 prompt + 512 answer | 150.17 | 149.29 | 93.5 MiB | 2.92 GiB |
| 2,832 tokens | interactive maximum: 2,320 prompt + 512 answer | 141.05 | 140.22 | 99.6 MiB | 3.11 GiB |
| 2,272 tokens | batch median: 2,016 prompt + 256 answer | 175.82 | 174.78 | 79.9 MiB | 2.50 GiB |
| 2,607 tokens | batch maximum: 2,351 prompt + 256 answer | 153.23 | 152.32 | 91.7 MiB | 2.86 GiB |

- Measured: vLLM logged 48.59 sequences at 8,192 tokens.
- The engine cap is `--max-num-seqs` = 32 at every length. 32 sequences need at most 9.00 GiB of KV, out of 13.71 GiB.

### Real prompt length

Measured with the Qwen tokenizer (`python -m app.prompt "<question>"`). Tokenizer only, no engine run (**verify** against the engine's `prompt_tokens` on the next deploy). Chunks under 50 tokens are dropped at ingest (624 chunks remain).

- System prompt: 431 tokens, the same for every request: the shared prefix.
- **Interactive** (12 questions, listed below; retrieval over the whole corpus):
  - Retrieved chunks, top 5 by BM25: 1,470 to 1,861 tokens. Question: 10 to 17 tokens. Template: 14 tokens.
  - **Total prompt: median 2,148 tokens, minimum 1,931, maximum 2,320.** Answer budget: 512 tokens.
- **Batch** (one "Write 3 practice questions" request per chapter label, 7 labels; retrieval inside the chapter):
  - Retrieved chunks: 1,131 to 1,864 tokens. Question: 22 to 42 tokens. Template: 13 to 14 tokens.
  - **Total prompt: median 2,016 tokens, minimum 1,604, maximum 2,351.** Answer budget: 256 tokens.
- Prompt share of the sequence: 81 % to 82 % interactive, 89 % to 90 % batch. The workload is mostly prefill, as the brief says for Track A.

Interactive questions measured: Little's Law and admission control; why prefix caching saves prefill; how PagedAttention stores KV in blocks; TTFT versus TPOT; when a gateway sheds instead of queueing; continuous versus static batching; chunked prefill and its flag; why decode is memory-bandwidth bound; KV after preemption; routing a shared-prefix request; cold starts and warming; KV bytes per token.

### Expected first limiter

- At 8192 tokens: prefill compute, because KV holds 48 full-length sequences while the engine runs at most 32, and each 8192-token prompt is a long prefill (**verify** under load).
- At 2,500 tokens: prefill compute, because KV holds 159 sequences, five times the 32-slot cap, so KV stays far from full while every request brings about 2,000 uncached prompt tokens to prefill (**verify** under load).
- At the real length, 2,272 to 2,832 tokens: prefill compute, for the same reason: KV holds 141 to 176 sequences against 32 slots, and only the 431-token system prompt is shared across all requests (**verify** under load, since the prefix cache reuses whole 16-token blocks only).
  - Uncached prompt tokens per request (prompt − 431): interactive 1,500 to 1,889, median 1,717; batch 1,173 to 1,920, median 1,585.
  - Measured hint: a warm cache miss of 495 prompt tokens took 53 to 84 ms of engine TTFT (`metrics/warmup_WorkerA_warm.json`, `metrics/warmup_WorkerB_warm.json`). That is roughly 6,000 to 9,000 prompt tokens per second per worker (**verify**: one request at a time, not under load).
- At 48 users (the rehearsal load), up to 48 requests are in flight against 32 slots, so about 16 should wait in the engine queue at peak (**verify**: `vllm:num_requests_waiting` in the rehearsal scrape).

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
