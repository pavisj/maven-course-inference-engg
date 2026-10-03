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

_Step 4. Code: `control/guard.py`, `control/admit.py`, `control/snapshot.py`, `control/gateway.py`. Tests: `tests/`. Run: `metrics/run_c/`._

### Where it runs

`control/gateway.py` listens on 127.0.0.1:8765 only (see Part 2, Security). Every request goes guard → admit → worker A. Step 4 has no placement and no gateway queue, so `queue_depth` is 0 and every admitted request goes to worker A. The tenant comes from the `X-Tenant` header.

### Guard: the first no (`inspect(payload, tenant)`)

| Order | Reason | Code | Rule | Constant and source |
|---|---|---|---|---|
| 1 | `bad_body` | 400 | `messages` is a non-empty list of `{role, content}`; `max_tokens`, if present, is a positive integer | none |
| 2 | `unknown_tenant` | 400 | `X-Tenant` is `interactive` or `batch` | tenants: decision 6 |
| 3 | `prompt_too_long` | 400 | prompt_tokens + max_tokens ≤ 8,192 | 8,192 = `--max-model-len` (Part 1) |
| 4 | `max_tokens_too_large` | 400 | max_tokens ≤ 512 interactive, ≤ 256 batch | 512 and 256: decision 6 |

- `prompt_tokens` is counted with the Qwen chat template, as the engine counts it. Check: the guard counted 1,946 tokens, and vLLM reported 1,946 prompt tokens for the same request.
- A request without `max_tokens` gets its tenant's budget. Otherwise vLLM would allow the rest of the 8,192-token context.

### Admit: should it enter the engine now? (`should_shed(req, snap)`)

The tests run in this order; the first match wins. Every shed is a 429 with a `Retry-After` header.

| Order | Reason | Shed when | Constants and source | retry_after |
|---|---|---|---|---|
| 1 | `tenant_tokens` | tenant tokens in the last 60 s + (prompt_tokens + max_tokens) > window | interactive 400,000, batch 150,000. Baseline B `mixed` charged 337,306 interactive and 290,105 batch tokens per minute. 400,000 gives interactive about 19 % headroom; 150,000 holds batch to about half its demand. | seconds until enough old charges leave the 60 s window |
| 2 | `timeout_queue` | est_ttft > deadline | deadline: interactive 6.0 s (above baseline B's worst laptop TTFT p95, 5,391 ms), batch 60.0 s | min(10, est_ttft − deadline + 1) |
| 3 | `kv_free` | kv_usage_pct + ceil((prompt_tokens + max_tokens) / 16) / 24,876 × 100 > 85 | 16 = vLLM `block_size`; 24,876 = vLLM `num_gpu_blocks` = 398,016 KV tokens / 16 (Part 1); 85 % keeps 15 % free for decode growth (baseline B peaked at 14.5 %) | 5 s |
| 4 | `over_cap` | running + queue_depth ≥ 32 and est_ttft > 2.0 s | 32 = `--max-num-seqs` (Part 1); 2.0 s: in baseline B `mixed`, est_ttft passes 2 s at about 13 waiting requests | min(10, est_ttft − deadline + 1), floored at 1 s |

**est_ttft** = (waiting + queue_depth) × avg_uncached_prompt_tokens / 4,096 × 0.85 + 0.4 seconds.

- 4,096 = `--max-num-batched-tokens`: prefill tokens per engine step (Part 1).
- 0.85 s per full step: assumed. Baseline B implies 1.04 s (`unique`), 1.50 s (`mixed`) and 2.67 s (`shared_chapter`) per step. See "Gaps" below.
- 0.4 s: TTFT with nothing waiting. Baseline A engine TTFT mean was 120 to 216 ms, plus margin.
- avg_uncached_prompt_tokens = mean prompt tokens of admitted requests (last 60 s) × (1 − prefix hit rate over the last 30 s of scrapes). Default 1,200 (Part 1: 1,173 to 1,920 uncached tokens) until both exist.
- waiting comes from worker A's `/metrics`, refreshed every 2 s, so it can be up to 2 s old. Tenant tokens and queue depth are exact at the moment of the decision.

**Change to the spec:** for `over_cap`, est_ttft is usually below the deadline, so min(10, est_ttft − deadline + 1) is negative. Retry-After must be at least 0, so retry_after is clamped to 1 to 10 s.

### Stay vs leave

Every shed in Step 4 is a 429: the request stays and may try here again after `retry_after_seconds`. Nothing overflows to another model yet. The guard's 400s are final. The traffic runner honours Retry-After: after a 429, a user waits `retry_after_seconds` before its next request.

### Tests (`pytest tests/`, 15 pass)

- Guard: one test per reason (`bad_body`, `unknown_tenant`, `prompt_too_long`, `max_tokens_too_large`), a pass with a correct token count, and the default `max_tokens` per tenant.
- Admit, with a fake snapshot: one test per reason (`tenant_tokens` with a 10 s retry, `timeout_queue` at est_ttft 7.87 s, `kv_free` at 84.5 %, `over_cap` at 32 running and 13 waiting).
- A 2,100-token interactive request passes on an idle snapshot.
- `kv_free` does not fire at 14.5 % KV usage, the baseline B peak.
- Order: a request over its tenant window and past its deadline is shed as `tenant_tokens`. Batch is not shed by the interactive deadline. `over_cap` needs both of its conditions.

### Run C: 48 users, `mixed`, 4 minutes, through the gateway (`metrics/run_c/`)

| | Baseline B `mixed` (no gateway) | Run C (gateway) |
|---|---|---|
| Requests sent | 1,071 | 1,286 |
| Admitted and answered | 1,071 | 758 (interactive 494, batch 264) |
| Shed: `tenant_tokens` | n/a | 248 (all batch) |
| Shed: `over_cap` | n/a | 280 (interactive 149, batch 131) |
| Shed: `timeout_queue`, `kv_free` | n/a | 0, 0 |
| Guard rejects | n/a | 0 |
| Laptop TTFT p50 / p95 | 3,787 ms / 4,629 ms | 2,394 ms / 5,820 ms |
| Gateway TTFT p50 / p95 | n/a | 2,352 ms / 5,613 ms (interactive 1,990 / 4,396; batch 3,089 / 6,554) |
| Engine TTFT mean | 3,002 ms | 2,028 ms |
| Max waiting in vLLM | 16 | 16 |
| Mean waiting during the run | 12.2 | 4.8 |
| Max KV usage | 10.8 % | 12.5 % |
| Preemptions | 0 | 0 |
| Prefix hit rate | 69.7 % | 67.0 % |

### Engine-side vs laptop-side TTFT

| Run | Laptop p50 / p95 | Gateway p50 / p95 | Engine mean | Engine p50 / p95 (histogram) |
|---|---|---|---|---|
| Baseline A, 8 users, direct | 906 ms / 1,063 ms | n/a | 153 ms | no buckets saved |
| Run C, 48 users, gateway | 2,394 ms / 5,820 ms | 2,352 ms / 5,613 ms | 2,028 ms | about 1,886 ms / 4,981 ms |

- Run C's engine percentiles are interpolated inside vLLM's histogram buckets, which are 1.0, 2.5, 5.0 and 7.5 s wide in this range: treat them as rough.
- The laptop adds about 40 ms to the gateway's number (localhost). The gateway adds about 470 ms to the engine's p50: the round trip from India to Modal us-east.
- Baseline A's 750 ms gap between laptop p50 and engine mean is the same network, with nothing waiting in the engine.

### Gaps run C shows (**verify** each in Step 5)

1. **est_ttft underestimates by about 3×.** At decision time its median was 0.65 s; the engine's real TTFT mean was 2.03 s. At 48 users a request waits for a decode slot to free, not for prefill. With a 67 % prefix hit rate the prefill part is small, but the slot wait is not. So `timeout_queue` never fired.
2. **`over_cap` does not separate tenants.** It shed 149 interactive requests, 23 % of all interactive traffic.
3. **p95 got worse while p50 got better.** Laptop p50 fell from 3,787 to 2,394 ms, but p95 rose from 4,629 to 5,820 ms. One possible cause: users shed by `over_cap` all wait the 1 s floor and come back together.
4. **The engine view is up to 2 s old,** and `queue_depth` is 0 until Step 5 adds a queue.

## Part 4. Place

## Part 5. Queue

## Part 6. Hop and warm-up

## Part 7. Wire the app to the cluster

## Part 8. Proof

## Submit
