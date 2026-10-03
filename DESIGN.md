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

- **Two colocated replicas, not a prefill/decode split.** WorkerA and WorkerB each run the whole model (prefill and decode) on one A10. Part 1 shows the first limit is prefill compute and decode slots, not KV memory, so both phases fit on one GPU. A split would add a KV transfer on every request for no memory gain.
- **No slicing.** One worker = one Modal container = one GPU = one KV cache (`min_containers=1`, `max_containers=1`). HAMi-style GPU slicing does not exist on Modal.
- **Affinity by chapter.** The gateway sends each chapter to a home worker (Part 4), so each worker's prefix cache holds half the corpus.
- **If we scale,** add decode slots, not replicas of the same size: baseline B ran 32 sequences with KV at 14.5 %. Run F tested a bigger `--max-num-seqs` with less memory (Part 5).

### 4. Concurrency and max length

- `--max-num-seqs 32` decode slots and `--max-model-len 8192` per worker (Part 1). The gateway holds request 33 in its own queue (Part 5), so vLLM's waiting queue stays near 0.

### 5. Hop backend

- **Backend: `recompute`.** Modal containers do not share memory or a network fabric for KV, so there is no Mooncake or NIXL here. A hop moves the request, not its KV: the destination prefills the prompt again. `control/hop.py` records every hop (Part 6).

### 6. Overflow

- **Who receives a 503 or 529:** first the other worker, once (still inside the cluster). Then the `Overflow` server: `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa79`) on its own A10, served as `tutor`, 0 to 1 containers, gone 5 minutes after its last request (`cluster/modal_vllm.py`).
- **Why that model:** the same family and tokenizer as the 3B, so the same prompt works unchanged. It needs 28 KiB of KV per token (28 layers × 2 KV heads × 128 × 2 × 2 bytes) against 36 KiB for the 3B, and about 3.1 GB of weights. It answers worse, but it answers.
- **Limiter:** at most 10 overflow requests per minute (`control/overflow.py`). Above that the client gets the 503. Every overflow answer carries `X-Via: overflow:qwen2.5-1.5b`.
- **Measured:** from zero, Overflow returned 503 for 137.7 s, then answered (`metrics/overflow_check.json`: 1,946 prompt tokens, 1.8 s laptop TTFT). So a cold Overflow cannot catch a sudden 503 burst; the gateway passes that 503 on with `X-Via` and counts `overflow_unavailable`.
- **What stays:** 429, 500 and out-of-memory errors never leave. They are our own decisions or our own faults.
- **Two kinds of overflow, one limiter** (Step 12):

| | Failure overflow (Step 8) | Capacity overflow (Step 12) |
|---|---|---|
| Who decides | an engine: it answers 503 or 529 (or no worker is warm) | the gateway: admit says it is over capacity |
| Trigger | 503, 529, or `no_warm_worker` | `over_cap` for an **interactive** request |
| Path | the other worker once, then Overflow | straight to Overflow |
| Over the limiter (10 per minute) | the 503 | the old 429 `over_cap` with retry_after |
| Headers | `X-Via: overflow:qwen2.5-1.5b` | `X-Via: overflow:qwen2.5-1.5b`, `X-Overflow-Reason: over_cap` |
| Metric | `gateway_overflow_total{reason="engine_503"\|"no_warm_worker", outcome}` | `gateway_overflow_total{reason="over_cap", outcome}` |

  - Source for capacity overflow: the instructor's AMA whiteboard (~01:18): "any time there is a request overflow on the gateway itself, this goes directly to the Overflow."
  - Batch `over_cap` stays a 429: batch can wait 60 s, so it never leaves.
  - Overflowed requests still count against the tenant's token window, with the same refund of unused answer budget.
  - If Overflow answers with an error, the gateway refunds the tokens and returns the 429 `over_cap` (outcome `failed`).

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

### Step 5 changes to admit (`control/admit.py`)

- **est_ttft = max(Little's law, prefill backlog).** Little's law: (vLLM waiting + gateway queue depth) / completion rate + 0.15 s. The completion rate counts the gateway's completions per worker over the last 30 s, floor 0.5/s. 0.15 s is baseline A's engine TTFT mean. The Step 4 formula is now only a lower bound.
- **Jitter:** every retry_after is multiplied by a random factor in [0.7, 1.3].
- **over_cap by tenant:** batch is shed when vLLM waiting + gateway queue for its worker reaches 8. Interactive is shed only when all seats are taken (gateway in-flight = 32) and est_ttft > 2.0 s.
- **Admit runs after place,** on the chosen worker, because its queue, KV and waiting count are what matter. Place changes no state until admit passes.
- **Every request logs est_ttft and the measured TTFT** (from arrival, so it includes the queue wait): `gateway_requests.jsonl`, plotted as `est_vs_measured_ttft.png`.

### Fixes after runs D and E (Pavithra approved, 2026-10-03)

| Problem seen | Fix | Constant and source |
|---|---|---|
| Run D: one-worker tenant windows on two workers shed 6,018 of 8,027 requests (`tenant_tokens`), with about 8 in flight per worker | The window is per warm worker: × the number of warm workers | 400,000 and 150,000 per worker: two warm workers give 800,000 and 300,000 |
| Each request was charged its full answer budget (512 or 256); the median answer is about 170 tokens | Unused budget is refunded when the stream ends (from vLLM's `usage`); a request with no answer gets the whole budget back | none |
| Run E: with no completions yet, the 0.5/s floor made 6 waiting requests look like 12 s; 14 false `timeout_queue` sheds in the first 7 s | The completion rate starts at a seed until 10 completions exist | 4.0 per second per worker: baseline B `mixed` completed 1,071 requests in 4.16 min = 4.3/s |
| Run D: sub-second `tenant_tokens` retries made 8,027 requests for 2,005 admitted | 2 s floor on `tenant_tokens` retry_after, after the jitter | 2.0 s |
| Warm line 0.18 s sat 3 % above warm workers' probes | Warm line 0.211 s (Part 6) | 0.176 s warm-probe reference × 1.2 |

Run G (after the fixes): the estimate's median was within 5 % of the measured TTFT (0.82 s against 0.98 s).

## Part 4. Place

`control/place.py`: `pick(req, workers, *, policy) -> Placement | Shed`. Default policy `prefix_then_load`; also `least_loaded` and `p2c`.

| Prefix key (file prefix of the top-ranked chunk) | Home worker |
|---|---|
| 00, 03a, 03b, 05 | WorkerA |
| 06, 06b, 07, 09, 10 | WorkerB |

- **Prefix key:** the app sends `X-Chapter` = the file prefix of its top-ranked chunk (`app/tutor.py`). The manifest's chapter labels are digits only, so the file prefix is what tells 03a from 03b and 06 from 06b.
- **Load bound:** if the home worker has 22 or more requests in flight (70 % of 32 seats), the request goes to the usable worker with the lowest score. If the other worker is busier, it stays home.
- **Change to the spec:** the bound uses the gateway's in-flight count only. The spec said running + in-flight, but vLLM's running requests are a subset of the gateway's in-flight requests, so the sum counts each request about twice. It would hop at about 11 real requests instead of 22.
- **Score** = gateway in-flight + gateway queue depth + vLLM waiting. Queue depth is a scorer, not only an admit input.
- **Usable** = reachable and warm (Part 6). No usable worker → `Shed(503, "no_warm_worker")`, which goes through the overflow gate.
- **No bounce:** a request is placed once. The one exception is a 503 or 529 from the chosen worker: the overflow gate tries the other worker once (Part 2, item 6).
- **Every admitted request logs** `preferred`, `worker` (chosen) and `hopped`.

Measured (`metrics/run_d/summary.json`, `metrics/run_e/summary.json`):

| Run | Admitted | Preferred A / B | Placed A / B | Hops |
|---|---|---|---|---|
| D: 3 mixes, 48 users | 2,005 | 711 / 1,294 | 946 / 1,059 | 561 (28 %): B→A 398, A→B 163 |
| E: chapter 07 only, 48 users | 305 | 0 / 305 | 147 / 158 | 147 (48 %): all B→A |

### Run H: does prefix routing help?

**Answer: prefix routing raised the prefix hit rate by 9.9 points (75.0 % → 84.9 %) and cut gateway TTFT p50 by 40 ms (530 → 490 ms) and p95 by 811 ms (1,887 → 1,076 ms).**

Both runs: `shared_chapter` on chapter 07 (home: WorkerB), 48 users, 4 min, normal engine, both workers warm, tenant windows per warm worker. The policy and the bound are set at gateway start (`PLACE_POLICY`, `LOAD_BOUND`). `preferred` is always the chapter home, so "away from home" means the same thing in both runs.

| | H1: `least_loaded` | H2: `prefix_then_load`, LOAD_BOUND 28 |
|---|---|---|
| Prefix hit rate, both workers | 75.0 % | 84.9 % |
| Prefix hit rate A / B | 75.7 % / 74.1 % | 84.4 % / 85.2 % |
| Hops (share of admitted) | 652 (53 %) | 579 (42 %) |
| Prompt tokens recomputed away from home | 1,382,112 | 1,234,077 |
| Gateway TTFT p50 / p95 | 530 / 1,887 ms | 490 / 1,076 ms |
| Engine TTFT mean A / B | 195 / 210 ms | 136 / 136 ms |
| Max in flight A / B | 32 / 24 | 20 / 28 |
| Admitted / shed (`tenant_tokens`) | 1,232 / 809 (+33 `over_cap`) | 1,367 / 520 |

- Why the gain is only 10 points: chapter 07 has 33 chunks, so even `least_loaded` soon has the whole chapter cached on both workers (75 % hit rate). Prefix routing keeps more of each request's 5-chunk set on one worker.
- H2 still hopped 42 % of requests: 48 users on one chapter is more than the home's 28-seat bound, so the excess goes to WorkerA.

## Part 5. Queue

`control/queue.py`: one queue per worker, in the gateway.

- **Seats:** at most 32 requests in flight per worker (`--max-in-flight`, the engine's `--max-num-seqs`). Request 33 waits in the gateway, not in vLLM's waiting queue.
- **Order:** interactive before batch, then arrival time.
- **Deadline in queue,** from arrival: interactive 6 s, batch 60 s. Past it, the request leaves the queue with 429 `timeout_queue`.
- **Client gone while queued:** dropped, `gateway_aborts_total{stage="queued"}`. **Client gone while in flight:** the gateway closes the upstream stream, so vLLM aborts the request and frees its KV blocks; `gateway_aborts_total{stage="in_flight"}`.
- **Who sits where:** your queue = the gateway queue (`gateway_queue_depth{worker}`). The engine's waiting queue = `vllm:num_requests_waiting`. With the 32-seat cap, vLLM waiting stayed at 0 to 2 in runs D and E.
- **Who goes first:** the gateway decides the order between tenants. Inside the engine, vLLM's scheduler decides prefill chunks and decode steps; we do not change it.

Measured: in runs D, E and F, the gateway queue never held a request (max depth 0). **Run G proves the queue.**

- **Run G** (normal engine, 96 users, `mixed`, 4 min, after the Part 3 fixes): 2,747 requests, 1,317 admitted.
  - The gateway queue held up to 8 requests per worker (mean 1.7 to 1.8); queue wait p95 2.19 s. vLLM's waiting queue stayed at 0: the waiting moved into the gateway, as designed.
  - Interactive goes first: gateway TTFT interactive p50 716 ms, p95 2,172 ms; batch p50 1,369 ms, p95 4,850 ms.
  - 3 requests left the queue at their deadline (429 `timeout_queue`, stage queue).
  - The queue stops at 8 because `over_cap` sheds before it grows: batch at 8 waiting, and interactive when the seats are full and est_ttft > 2 s (568 interactive sheds).
  - KV peaked at 12.9 %; no preemptions.

- Runs D and E: the tenant windows held the load below 26 in flight per worker (Part 3), so there was nothing to queue.
- **Run F** (pressure engine: `--gpu-memory-utilization 0.5`, `--max-num-seqs 64`; gateway 64 seats per worker; tenant windows ×10 so `tenant_tokens` cannot hide `kv_free`; 64 users, `mixed`, 3 min): 1,528 requests, all admitted, 0 sheds, 0 preemptions.
  - vLLM's KV shrank to 7,906 blocks = 126,496 tokens per worker (normal engine: 398,016).
  - KV usage peaked at 36.1 % (WorkerA) and 31.7 % (WorkerB); in flight averaged 30 per worker.
  - **The expected `kv_free` sheds did not happen.** 64 users split over two workers is about 32 per worker. `kv_free` fires at 85 % = about 107,500 tokens, about 44 sequences of 2,400 tokens on one worker, so about 90 users. The run proved the gateway admits a 64-seat engine without false sheds; it did not prove `kv_free`. A follow-up needs about 96 users (or one worker), and a second pressure deploy.
- **Run F2** (second pressure deploy, approved: `--gpu-memory-utilization 0.4`, `--max-num-seqs 64`; 64 seats per worker; tenant windows ×10; 128 users, `mixed`, 3 min): 2,599 requests, 1,818 admitted, 781 shed by **`kv_free`** (389 batch, 392 interactive), 0 failures.
  - vLLM's KV: 4,443 blocks = 71,088 tokens per worker.
  - KV still peaked at 99.8 % (WorkerA) and 99.4 % (WorkerB); vLLM preempted 15 requests (7 and 8), and its waiting queue reached 22.
  - **KV full after admit: the door sheds, and the engine preempts the rest.** `kv_free` reads KV usage up to 2 s old and counts only the new request's blocks, not the decode growth of requests already running. So it stopped 781 requests at the door, and 15 still had to be preempted inside the engine.
  - **Client gone:** no client left mid-stream in these runs (aborts 0). The path is tested: closing the upstream stream makes vLLM abort the request and free its blocks (Step 4 stub test, `client_gone`).
- **Which saved memory on the shared-prefix mix?** PagedAttention packs every sequence into 16-token blocks, so KV grows with real length, not with `--max-model-len`. The prefix cache reuses blocks: 56 % to 66 % of prompt tokens were cache hits per worker in runs D to F, so those tokens took no new blocks. On this workload the prefix cache saved more: at 30 sequences, KV stayed at 31 % to 36 % of a halved cache.

### Part 5 questions: Pavi's answers (Step 10b)

> **10b · Q5.1 · Pavi to answer.** Who sits in **your** queue vs vLLM's **waiting** queue?
> Evidence to paste: `gateway_queue_depth{worker}` and `vllm:num_requests_waiting` from `metrics/run_g/scrape_run_g.jsonl`; plot `plots/run_g/gateway_queue_by_worker.png`.

> **10b · Q5.2 · Pavi to answer.** Waiting / running / swapped (or preempted)?
> Evidence to paste: `metrics/run_f2/summary.json` (workers: max_running, max_waiting, preemptions); plot `plots/run_f2/preemptions.png`.

> **10b · Q5.3 · Pavi to answer.** `orch_replica_queue_depth` (or your equivalent): which pod, under which mix?
> Evidence to paste: `gateway_queue_depth{worker}` per run; `metrics/run_*/summary.json` (gateway_queue_max per worker); notebook section "Workers".

> **10b · Q5.4 · Pavi to answer.** If a 32k RAG retrieve and a short agent decode are both ready, who goes first: you or the engine?
> Evidence to paste: `control/queue.py` (priority order); run G gateway TTFT by tenant (`metrics/run_g/summary.json`).

> **10b · Q5.5 · Pavi to answer.** PagedAttention packs KV; prefix / radix cache reuses KV. Which one saved memory on your shared-prefix mix?
> Evidence to paste: prefix hit rate per worker (`metrics/run_h1/summary.json`, `metrics/run_h2/summary.json`); KV usage in runs F and F2.

> **10b · Q5.6 · Pavi to answer.** Chunked prefill and continuous batching: which engine flags (`max-num-batched-tokens`, `max-num-seqs`) did you use, and why?
> Evidence to paste: `ENGINE_ARGS` and `PROFILES` in `cluster/modal_vllm.py`; Part 1 of this file.

> **10b · Q5.7 · Pavi to answer.** KV full after admit: do you shed at the door, or does the engine preempt?
> Evidence to paste: run F2: 781 `kv_free` sheds and 15 preemptions (`metrics/run_f2/summary.json`); plot `plots/run_f2/kv_usage.png`.

> **10b · Q5.8 · Pavi to answer.** Client gone (`aborted`): who frees the KV? And how?
> Evidence to paste: `gateway_aborts_total{stage}`; `relay()` in `control/gateway.py`; `vllm:request_success_total{finished_reason="abort"}` in a scrape.

> **10b · Q5.9 · Pavi to answer.** After a worker returns: slam it at 100 %, or ramp while p99 holds?
> Evidence to paste: `metrics/warm_proof.json`; `metrics/run_e/warm_events.jsonl`; the warm gate in Part 6.

## Part 6. Hop and warm-up

**Hop.** On Modal, containers do not share KV memory. A hop means the destination recomputes the prefix. Nothing is copied. The record proves where the recompute happened: `{ts, src: preferred, dst: chosen, tokens: prompt_tokens, prefix: chapter, backend: "recompute"}`, appended to `hops.jsonl` and counted in `gateway_hops_total{src,dst}`.

- Run D: 561 hops, 1,179,270 prompt tokens recomputed away from their home worker.
- Run E (all traffic on chapter 07, home WorkerB): 147 of 305 requests hopped to WorkerA; 310,030 tokens recomputed. WorkerB's prefix hit rate was 59.3 %, WorkerA's 61.2 %.

**Warm-up.** `cluster/warm.py` sends 5 probes to a worker: chapters 00, 03, 05, 07 and 10, each with the real system prompt and a real chunk set. A "Probe <run id>." line keeps the chunks uncached on each run. Engine TTFT per probe comes from the change in `vllm:time_to_first_token_seconds_sum`.

- **Warm** = the last probe's engine TTFT is within 20 % of the warm-probe reference: 0.176 s × 1.2 = 0.211 s.
  - The first line was 20 % of baseline A's engine mean, 0.15 s × 1.2 = 0.18 s. It was too tight: baseline A's mean includes prefix-cache hits, but each probe prefills about 1,600 uncached tokens. Warm workers measured 175 to 193 ms, so a warm worker could fail. The reference is now the same probes on warm workers (run E, last probe: 175 and 176 ms). Cold first probes (1,007 and 1,026 ms) still fail by far. Changed on 2026-10-03 at Pavithra's request.
  - Runs D and E used the 0.18 s line; both workers still passed it on the first round.
- The gateway runs the probes when it starts and whenever a worker comes back from being unreachable or restarted. It tries up to 3 rounds, then again every 30 s. `gateway_worker_warm{worker}` is 0 or 1, and place never uses a worker that is not warm.

**Proof** (`metrics/warm_proof.json`): WorkerB only was cold-started (its container stopped; Modal started a fresh one, healthy after 133.7 s), then probed at once.

| Worker | Probe 1 | Probe 2 | Probe 3 | Probe 4 | Probe 5 | Warm |
|---|---|---|---|---|---|---|
| WorkerA (warm, untouched) | 365 ms | 184 ms | 185 ms | 176 ms | 176 ms | yes |
| WorkerB (fresh container) | 1,007 ms | 187 ms | 187 ms | 176 ms | 175 ms | yes |

- A healthy replica is not a warm one: WorkerB answered `/health` with 200, and its first request still took 1,007 ms in the engine, 5.7 times its warm value.
- The same pattern showed after the Step 8 deploy restarted both workers: first probes 1,014 and 1,026 ms, then 174 to 186 ms.
- **Change to the spec:** "a fresh deploy of WorkerB only" is not possible with `modal deploy`, which restarts every server in the app. `cluster/cold_start.py` stops WorkerB's container instead, which restarts WorkerB only.

## Part 7. Wire the app to the cluster

```
app turn: python -m app.tutor "question" --base-url http://127.0.0.1:8765
  retrieve (BM25 top 5)  -> prompt (system + sorted chunks + question)
  headers: X-Tenant, X-Chapter (file prefix of the top chunk)
        ↓
control/gateway.py  127.0.0.1:8765
  guard.inspect       400 bad_body | unknown_tenant | prompt_too_long | max_tokens_too_large
  place.pick          preferred = chapter home; hop if home ≥ 22 in flight or not warm; 503 if no warm worker
  admit.should_shed   429 tenant_tokens | timeout_queue | kv_free | over_cap (+ jittered Retry-After)
  queue.wait          32 seats per worker; 429 timeout_queue past the deadline; drop if the client is gone
  forward             stream from WorkerA or WorkerB
  overflow gate       429/500/OOM stay · 503/529: other worker once, then Overflow (10 per minute, X-Via)
        ↓
vLLM on Modal (WorkerA, WorkerB): waiting / running / preempt, prefix cache, chunked prefill
```

- Traffic runs through the same path: `traffic/gateway_run.sh` (runs C to F) or `python -m traffic.run --base-url http://127.0.0.1:8765`.
- The baselines (A, B) skip the gateway on purpose: `--base-url` defaults to WorkerA.

### Two kinds of overflow on this path

- **Capacity overflow** happens at admit, before any engine sees the request: interactive `over_cap` → Overflow (Step 12). The gateway decides.
- **Failure overflow** happens after forward: a worker answered 503 or 529 → the other worker once → Overflow (Step 8). An engine decides.
- Both share one limiter (10 per minute) and the same `X-Via` header; Part 2, item 6 has the full comparison.

### Run I: what happened when the gateway overflowed?

**Answer: 40 interactive requests went to the 1.5B Overflow (the limiter's 10 per minute for 4 minutes); 496 more hit the limiter and still got the 429 `over_cap`. In run G, all 568 got a 429.**

Run I: run G's load (96 users, `mixed`, 4 min, normal engine), with capacity overflow on. Overflow pre-warmed first (`metrics/run_i/overflow_prewarm.json`: up after 132.8 s).

| | Run G (no capacity overflow) | Run I (capacity overflow) |
|---|---|---|
| Interactive `over_cap` | 568, all 429 | 536: 40 sent to Overflow, 496 limited (429), 0 failed |
| Overflow TTFT p50 / p95 | none | 1,007 / 2,143 ms (prompt median 2,118 tokens) |
| Cluster interactive TTFT p50 / p95 | 716 / 2,172 ms | 757 / 2,187 ms |
| Goodput (answered within deadline per minute) | 321.3 | 322.6 (Overflow 9.7 of it) |

- Overflow answered as fast as the cluster: its p95 was 2,143 ms against the cluster's 2,187 ms. The 1.5B model has less to compute per token.
- The limiter, not Overflow, is the bottleneck: 93 % of interactive `over_cap` requests still got the 429.
- **Went wrong:** 20 interactive `timeout_queue` sheds in the first 7.6 to 10.6 s. After the 10th completion, the rate divided by the time since the gateway started, which included the 12 s warm-up wait. Fixed after run I (rate span from the first completion, with a test), not re-measured on GPU.

### Run J: the three open decisions, tested together

**Answer: goodput rose from 322.6 to 432.0 per minute, but only for interactive. Batch goodput fell from 130.1 to 7.4 per minute. Interactive TTFT p50 rose from 757 ms to 3,506 ms. The defaults stay run I's values.**

Run J: run I's load (96 users, `mixed`, 4 min, normal engine, capacity overflow on). Overflow was pre-warmed first: up after 129.8 s (`metrics/run_j/overflow_prewarm.json`). Each setting sits behind a gateway flag. Run I's value stays the default.

| Setting | Run I (default) | Run J | Gateway flag / `gateway_run.sh` env |
|---|---|---|---|
| Interactive `over_cap` trigger | 2 s | 4 s | `--over-cap-ttft` / `OVER_CAP_TTFT` |
| Tenant windows per warm worker | 400,000 interactive, 150,000 batch | 800,000 interactive, 300,000 batch | `--tenant-window-scale` / `TENANT_SCALE` |
| Prefill-backlog floor in est_ttft | 0.4 s | 0 (removed) | `--base-ttft` / `BASE_TTFT` |

```bash
RUN=run_j MIXES=mixed USERS=96 DURATION=240 TENANT_SCALE=2 OVER_CAP_TTFT=4 BASE_TTFT=0 WARM_TTFT=0.28 bash traffic/gateway_run.sh
```

The gateway writes the active values to `metrics/run_j/admit_settings.json`. Tests: `tests/test_admit.py`, one per setting and one for the defaults (47 pass).

**Two differences from run I come from the deploy, not from the settings:**

- **The fresh workers were slower.** Warm probes measured 229 to 244 ms on the last probe (run I: 168 to 192 ms), across 6 rounds. There was no cold first probe (about 200 ms, not 1,000 ms). The 0.211 s warm line would never pass. **Decision:** run J used a warm line of 0.28 s, the 233 ms steady probe × 1.2 (`--warm-ttft`, env `WARM_TTFT`). The default stays 0.211 s. Both workers passed on the first round, after 12 s, as in run I.
- **The network trip was shorter:** 276 ms, against 348 ms in run I (`metrics/run_j/network_offset.json`). **Decision:** run J's "trip removed" numbers use its own 276 ms. `summarise_run.py` and `make_plots.py` now read `metrics/<run>/network_offset.json` first.

| | Run I | Run J |
|---|---|---|
| Requests sent / admitted | 2,637 / 1,302 | 3,626 / 1,803 |
| Goodput per minute, total | 322.6 | 432.0 |
| Goodput per minute, interactive / batch | 192.5 / 130.1 | 424.6 / 7.4 |
| Goodput per minute, WorkerA / WorkerB / Overflow | 160.7 / 152.7 / 9.7 | 214.7 / 214.9 / 2.4 |
| Shed `tenant_tokens`, interactive / batch | 0 / 731 | 0 / 0 |
| Shed `over_cap`, interactive / batch | 496 / 48 | 19 / 1,767 |
| Shed `timeout_queue`, interactive / batch | 20 / 0, all at admit | 12 / 15, all in the queue |
| Gateway TTFT p50 / p95, all | 1,080 / 3,205 ms | 3,518 / 4,576 ms |
| Gateway TTFT p50 / p95, interactive | 757 / 2,187 ms | 3,506 / 4,557 ms |
| Gateway TTFT p50 / p95, batch | 1,634 / 5,444 ms | 4,236 / 6,624 ms |
| Overflow sent / limited / failed | 40 / 496 / 0 | 10 / 19 / 0 |
| Overflow TTFT p50 / p95 | 1,007 / 2,143 ms | 992 / 1,026 ms |
| est_ttft p50 / measured p50 | 0.817 / 1.080 s | 3.118 / 3.518 s |
| Est within 50 %, raw | 60.4 % | 93.2 % |
| Est within 50 %, trip removed | 47.7 % (348 ms trip) | 89.5 % (276 ms trip); 88.2 % with run I's 348 ms |
| In flight per worker, mean | 17.2 / 16.8 | 30.3 / 30.4 |
| Gateway queue per worker, max / mean | 8 / 1.6 | 13 / 11.4 and 11.0 |
| Hops (share of admitted) | 436 (33.5 %) | 899 (49.9 %) |
| Prefix hit rate, WorkerA / WorkerB | 69.5 % / 68.5 % | 53.3 % / 54.0 % |
| Engine TTFT mean, WorkerA / WorkerB | 250 / 278 ms | 395 / 413 ms |
| KV max, preemptions | 11.6 %, 0 | 14.3 %, 0 |

Plots: `plots/run_j/`. Full comparison: `metrics/run_j/compare_run_i.json` (`python metrics/compare_runs.py --base run_i --new run_j`).

**What happened:**

1. **The seats filled.** In run I, the 2 s trigger shed interactive requests while the seats were only half used: 17 of 32 in flight on average. With 4 s, admit lets the gateway queue grow until est_ttft reaches 4 s. At about 3.7 completions per second per worker, that is about 13 waiting. In flight stayed at 32 on both workers for the whole run (`plots/run_j/gateway_queue_by_worker.png`). **The trigger works as a queue-length cap: trigger × completion rate.**
2. **Batch starved.** Batch `over_cap` sheds batch when vLLM waiting + gateway queue ≥ 8. The queue held 11 to 13 for the whole run. So 0 batch requests were admitted in minutes 1 to 3. The tenant windows ×2 removed every `tenant_tokens` shed (731 → 0), but batch then failed the next test, `over_cap`.
3. **Interactive paid in TTFT.** After the first 30 s, an interactive TTFT was about 3.0 s of gateway queue wait plus 0.29 s in the engine plus the trip. In run I the queue wait was 36 ms. Goodput counts every answer within 6 s, so goodput does not show the 4.6× slower p50.
4. **The load generator amplified it.** In `mixed`, each user alternates one interactive and one batch request. A shed batch request costs the user about 1 s (the 1 s retry floor × jitter). Then the user sends its next interactive request. So batch sheds became extra interactive load: interactive offered 432 per minute, against 318 in run I.
5. **More hops.** With both workers full, place hopped half of all requests (49.9 %, run I 33.5 %). The prefix hit rate fell about 15 points. Engine TTFT rose from about 260 ms to about 400 ms. Part of that rise is the slower deploy.

**Which setting did what:** one run tests all three together, so the split rests on the mechanism, not on a separate run.

- The 4 s trigger filled the seats and the queue. That gave the interactive gain and the batch starvation.
- Tenant windows ×2 removed `tenant_tokens`, but gave batch nothing while batch `over_cap` binds.
- The floor removal changed only 67 decisions (see below).

#### Did the run I estimator fix work?

**Yes, at admit: run J had 0 `timeout_queue` sheds at admit in its first 15 s. Run I had 20, at 8.0 to 11.0 s after its first request, with estimates of 7.2 to 16.2 s.**

- Counting by arrival time in the first 15 s, run J had 26 `timeout_queue` sheds, all at the queue stage:
  - 12 interactive requests arrived at 0.28 to 0.35 s. Each waited the full 6 s deadline in the gateway queue.
  - 14 batch requests arrived in the first 15 s. Each waited the full 60 s batch deadline behind interactive.
- These are real waits, not false estimates. But the 12 interactive sheds show a second, smaller miss at start. Admit estimated 0.15 to 3.4 s for them. The completion-rate seed (4.0 per second per worker) assumes a steady state. At start, the first 32 answers on each worker finish at about the same time, so the first seats free later than the seed says.
- Run I had no queue-stage sheds because its 2 s trigger sent these requests to Overflow or a 429 at admit.

#### Estimate vs measured: what the floor removal showed

- Within 50 %: 93.2 % raw and 89.5 % with the trip removed, against 60.4 % and 47.7 % in run I.
- **This does not prove the floor removal.** Run J ran full for all 4 minutes: 1,730 of 1,803 admitted requests had est_ttft of 2 s or more. In that range the queue term dominates, and the estimate tracked in every loaded run (G, F2, I).
- Only 67 decisions had est_ttft below 0.5 s, all in the first 0.3 s. With the floor removed they estimated 0.15 s. They measured about 3.4 s with the trip removed, because 64 prompts of about 2,100 tokens started prefill at once. The 0.4 s floor would have missed them too. Little's law counts waiting and queued requests, not prefills already in flight.
- The light-load case, where the floor made the estimate too high (see "Network offset"), did not happen in run J. It needs a light-load run to test.

#### What-if predictions vs run J

| Knob | First-order prediction from run I | Run J measured | Held? |
|---|---|---|---|
| Interactive `over_cap` trigger 2 s → 4 s | 520 of 536 interactive `over_cap` admitted; +121.0 interactive goodput per minute | interactive `over_cap` attempts 536 → 29; interactive admitted +1,009; interactive goodput +232.1 per minute | **Sign and shed count held; the size was 2× too low** |
| Tenant windows ×1 → ×2 | 509 of 731 batch `tenant_tokens` admitted; +127.5 batch goodput per minute | `tenant_tokens` 731 → 0; batch admitted −508; batch goodput −122.7 per minute | **No: wrong sign** |
| Both together (sum) | +248.6 per minute (about 571 per minute) | +109.4 per minute (432.0) | **No** |
| Overflow limiter 10 → 30 per minute | +20.0 per minute (80 more sent) | not changed in run J; only 29 attempts, 19 limited, so 30 per minute would add at most 19 sends (about 4.5 per minute) | **Premise gone** |
| LOAD_BOUND 22 → 28 | 0 per minute | not changed in run J; hop share rose to 49.9 % with full workers | **Not tested** |

Why the predictions held or failed:

- **The replay keeps the logged queue state.** In run I the gateway queue stayed at 8 or less, so batch `over_cap` fired only 48 times. The trigger change raised the queue to 13, so batch's next test fired 1,767 times. The two knobs interact, and a one-knob replay cannot see that.
- **The load is closed-loop.** Admitted users come back sooner, and shed batch users send their interactive request sooner. That extra interactive load met the seats that batch left free. So the trigger's gain was twice the first-order number.
- **The shed count held because it does not depend on later state:** 536 − 520 = 16 interactive `over_cap` left in the prediction; 29 measured.

#### Decisions taken in this session (Pavithra to confirm)

1. Run J used a warm line of 0.28 s (above). The default stays 0.211 s.
2. Run J's "trip removed" numbers use run J's own 276 ms trip.
3. "First 15 s" counts requests by arrival time from the first gateway request. Admit-stage and queue-stage sheds are counted apart.
4. The defaults stay run I's values. Run J's settings buy interactive goodput with batch goodput and with interactive TTFT, so I did not make them the default.

**Next step:** give batch a protected share before the 4 s trigger is tried again, for example seats that only batch may use. Run J shows that, at full seats, interactive-first order plus batch `over_cap` at 8 leaves batch nothing.

### Goodput

goodput = requests answered within deadline per minute: status 200 and the **first token** within the deadline from arrival (interactive 6 s, batch 60 s; the same deadlines admit and the queue use). Shed, failed or late = 0. Arrival is when the client sent the request, so the client-side TTFT counts. `metrics/goodput.json`, `plots/goodput_vs_offered.png`.

| Run | Users | Offered / min | Goodput / min | Share | Interactive / batch goodput | Per worker (A / B) | Late |
|---|---|---|---|---|---|---|---|
| A | 8 | 96.9 | 96.9 | 100 % | 81.2 / 15.6 | 96.9 / direct | 0 |
| B | 48 | 256.9 | 253.8 | 99 % | 211.3 / 42.5 | 253.8 / direct | 39 |
| C | 48 | 320.4 | 186.6 | 58 % | 120.8 / 65.8 | 186.6 / none | 9 |
| D | 48 | 660.9 | 165.1 | 25 % | 143.3 / 21.7 | 77.9 / 87.2 | 0 |
| E | 48 | 477.5 | 152.7 | 32 % | 152.7 / none | 73.6 / 79.1 | 0 |
| F | 64 | 490.6 | 490.6 | 100 % | 245.3 / 245.3 | 242.8 / 247.9 | 0 |
| G | 96 | 671.7 | 321.3 | 48 % | 190.5 / 130.8 | 165.3 / 156.2 | 3 |
| F2 | 128 | 836.6 | 577.8 | 69 % | 284.9 / 292.9 | 293.2 / 287.8 | 23 |
| H1 | 48 | 500.3 | 294.3 | 59 % | 294.3 / none | 155.6 / 138.7 | 12 |
| H2 | 48 | 471.8 | 341.8 | 72 % | 341.8 / none | 144.8 / 197.0 | 0 |
| I | 96 | 636.3 | 322.6 | 51 % | 192.5 / 130.1 | 160.7 / 152.7 (+ Overflow 9.7) | 5 |
| J | 96 | 864.0 | 432.0 | 50 % | 424.6 / 7.4 | 214.7 / 214.9 (+ Overflow 2.4) | 0 |

- Offered load includes retries after a 429, so a run that sheds more offers more.
- Runs F and F2 used tenant windows ×10 (KV tests): their goodput shows what the two workers can deliver without tenant limits, about 490 to 580 per minute.
- Late answers stay small (0 to 23 per run): the gateway sheds work it cannot finish in time instead of answering late.
- **Run J has the highest goodput at 96 users, but 98 % of it is interactive.** Goodput per minute does not show tenant fairness or the TTFT inside the deadline. Read it with goodput by tenant and TTFT p50.

### Policy ledger

| Stage | Condition | Action | Constant | Source | Run that tested it | Goodput effect | Alert |
|---|---|---|---|---|---|---|---|
| guard | body malformed | 400 `bad_body` | none | Step 4 | unit tests, stub test | none: never a GPU second | `bad_body` > 1 % of requests (client bug) |
| guard | `X-Tenant` not interactive or batch | 400 `unknown_tenant` | 2 tenants | decision 6 | unit tests | none | any `unknown_tenant` (misconfigured client) |
| guard | prompt + max_tokens > 8,192 | 400 `prompt_too_long` | 8,192 | `--max-model-len` | unit tests | protects KV per sequence | none |
| guard | max_tokens > 512 / 256 | 400 `max_tokens_too_large` | 512 / 256 | decision 6 | unit tests | bounds decode time | none |
| place | home has < LOAD_BOUND in flight and is warm | go home | LOAD_BOUND 22 (28 in H2) | 70 % of 32 slots | D to I; H1 vs H2 | H2 vs H1: +9.9 hit-rate points, +47.5 goodput / min | hop share > 50 % for 10 min |
| place | home busy or not warm | hop to the lowest-score worker | score = in flight + queue + waiting | Step 5b | D, E, G, I | E: 310,030 tokens recomputed | tokens recomputed / min rising |
| place | no warm worker | 503 → failure overflow | none | Step 7 | none (never happened) | none measured | any `no_warm_worker` |
| admit | tenant tokens + request > window × warm workers | 429 `tenant_tokens`, retry ≥ 2 s | 400,000 / 150,000 per worker | baseline B | D (6,018), G (770), I (731), J at ×2 (0) | D capped at 165 / min; replay ×2: +109 to +129 / min; J at ×2: batch −122.7 / min (batch moved to `over_cap`) | `tenant_tokens` > 20 % of a tenant's requests for 5 min |
| admit | est_ttft > deadline | 429 `timeout_queue` | 6 s / 60 s; Little's law | baseline B p95 | E (14), G (24), I (20), J (0) | false sheds at start: fixed twice; J: 0 at admit in the first 15 s | `timeout_queue` > 5 % of requests |
| admit | KV % + request blocks > 85 % | 429 `kv_free` | 85 %; num_gpu_blocks | Part 1 | F2 (781) | F2: 578 / min at KV 99.8 % | KV usage > 90 % or any preemption |
| admit | batch: waiting + gateway queue ≥ 8 | 429 `over_cap` | 8 | Step 5a | G (68), I (48), J (1,767) | protects interactive TTFT; J: starved batch when the queue held 13 | batch admitted = 0 for 1 min |
| admit | interactive: seats full and est_ttft > 2 s | Overflow under the limiter, else 429 `over_cap` | 2 s (J: 4 s); 32 seats | baseline B | G (568 × 429), I (40 sent, 496 × 429), J at 4 s (10 sent, 19 × 429) | replay 2 → 4 s: +121 to +131 / min; J: interactive +232.1 / min, TTFT p50 757 → 3,506 ms | interactive `over_cap` > 10 % |
| queue | 32 in flight on the worker | wait in the gateway; interactive first | 32 seats | `--max-num-seqs` | G, I (max 8 waiting), J (max 13) | interactive p50 716 ms vs batch 1,369 ms (G); J: 3 s queue wait | gateway queue wait p95 > 3 s |
| queue | deadline passes while queued | 429 `timeout_queue` (stage queue) | 6 s / 60 s | Step 6 | G (3), J (27) | J: 12 interactive at start; 15 batch waited 60 s behind interactive | none |
| queue | client gone (queued / in flight) | drop / close upstream so vLLM aborts | none | Step 6 | stub test | frees KV for others | in-flight aborts spike |
| hop | chosen ≠ chapter home | record `recompute` | backend recompute | Part 6 | D to I | the price of a hop: lost prefix hits | none |
| warm | last probe engine TTFT ≤ 0.211 s | mark warm; place may use it | 0.211 s (J: 0.28 s) | warm probes (run E) | E: 1,007 → 175 ms; J deploy steady at 229 to 244 ms | avoids a 1 s first request | worker not warm > 5 min |
| overflow | 429, 500 or out of memory from a worker | stay: return with Retry-After | 2 s | Step 8 | unit tests | none measured | 500 rate > 0 |
| overflow | 503 or 529 from a worker | the other worker once, then Overflow | 10 per minute | Step 8 | unit tests; Overflow check | none measured (no 503 in runs) | any engine 503 |
| overflow | interactive `over_cap` (capacity) | Overflow under the limiter | 10 per minute | AMA whiteboard | I (40 sent), J (10 sent) | +9.7 goodput / min; replay 10 → 30: +20 / min; J: 29 attempts, so the limiter barely matters at 4 s | overflow limited > 0 for 5 min |

### What-if replay (first-order)

**First-order only: a changed decision changes the later queue state, and the replay keeps the logged state, so each number is a bound, not a measurement.** `metrics/whatif.py`, `metrics/whatif.json`.

| Knob | Run | Logged decisions | Change bucket | First-order goodput change |
|---|---|---|---|---|
| LOAD_BOUND 22 → 28 | D / E / G / F2 / I | 561 / 147 / 478 / 894 / 436 hops | 561 / 147 / 83 / 0 / 82 stay home | 0 / min in every run: hopped requests were not late |
| Interactive `over_cap` trigger 2 s → 4 s | G / I | 568 / 536 | 524 / 520 admitted | +131.1 / +121.0 per min |
| Overflow limiter 10 → 30 per minute | I | 536 attempts | 80 more sent | +20.0 per min (all of run I's overflow answers were on time) |
| Tenant windows ×1 → ×2 | D / E / G / I | 6,018 / 635 / 770 / 731 `tenant_tokens` | 1,441 / 218 / 516 / 509 admitted | +120.1 / +109.3 / +129.1 / +127.5 per min |

- LOAD_BOUND: the replay reads each worker's in-flight count from the 5 s gateway scrape, not the exact value at decision time, so its "stay home" counts are rough.
- **Run J checked two of these knobs together** (see "Run J" above). The `over_cap` trigger held in sign and was 2× too low. Tenant windows ×2 had the wrong sign: batch moved from `tenant_tokens` to `over_cap`.

### Network offset

The estimate predicts engine TTFT; the gateway measures TTFT from arrival, which includes the laptop → Modal us-east trip.

- Trip: median of 20 `GET /health` round trips from the laptop to each worker: **347.8 ms** (WorkerA 348.6 ms, WorkerB 347.0 ms; range 344 to 357 ms). `metrics/network_offset.json`.
- Plots: `plots/<run>/est_vs_measured_ttft.png` (as measured) and `est_vs_measured_ttft_minus_trip.png` (trip removed).

| Run | est p50 | measured p50 | measured − trip p50 | within 50 %, raw | within 50 %, trip removed |
|---|---|---|---|---|---|
| D | 0.400 s | 0.574 s | 0.226 s | 67.7 % | 46.1 % |
| E | 0.400 s | 0.530 s | 0.182 s | 60.3 % | 20.0 % |
| F | 0.400 s | 0.501 s | 0.154 s | 93.6 % | 25.8 % |
| G | 0.824 s | 0.977 s | 0.629 s | 56.5 % | 35.8 % |
| F2 | 0.400 s | 0.863 s | 0.515 s | 42.8 % | 63.7 % |
| H1 | 0.400 s | 0.530 s | 0.182 s | 78.8 % | 36.3 % |
| H2 | 0.400 s | 0.490 s | 0.142 s | 84.3 % | 26.9 % |
| I | 0.817 s | 1.080 s | 0.732 s | 60.4 % | 47.7 % |
| J (floor removed; own trip 276 ms) | 3.118 s | 3.518 s | 3.241 s | 93.2 % | 89.5 % |

- **The estimate is too high at light load.** With the trip removed, an idle engine answers in 142 to 226 ms, but est_ttft never goes below 0.4 s: that is the Step 4 prefill-backlog floor (`BASE_TTFT_S = 0.4`), which `max(Little, backlog)` keeps. The raw "within 50 %" looked good only because the 0.35 s trip filled that gap.
- Under load (F2, G, I) the queue term dominates and the estimate tracks better.
- Run J removed the floor, but ran full for the whole run. Its 89.5 % comes from the loaded regime, not from the floor removal. The light-load case is still untested.

## Part 8. Proof

_Run data for every answer: `metrics/run_*/summary.json`, `metrics/goodput.json`, the notebook. The 13 answers below are Pavi's (Step 10b)._

> **10b · Q8.1 · Pavi to answer.** What is the app, and which tokens are shared vs unique?
> Evidence to paste: Part 1 "Real prompt length"; `app/prompt.py`.

> **10b · Q8.2 · Pavi to answer.** What dies at guardrails vs admit vs place vs queue?
> Evidence to paste: sheds by stage and reason in `metrics/run_*/summary.json`; the policy ledger in Part 7.

> **10b · Q8.3 · Pavi to answer.** Where do I prevent work that will time out? (`timeout_queue`)
> Evidence to paste: `should_shed` test 2 in `control/admit.py`; `QUEUE_DEADLINE_S` in `control/queue.py`; run G `timeout_queue` counts.

> **10b · Q8.4 · Pavi to answer.** Where do I protect KV?
> Evidence to paste: `kv_free` in `control/admit.py`; run F2.

> **10b · Q8.5 · Pavi to answer.** Where do I prioritize interactive traffic? (`p99_spread`)
> Evidence to paste: `PRIORITY` in `control/queue.py`; run G TTFT by tenant; capacity overflow in run I.

> **10b · Q8.6 · Pavi to answer.** Where do I stop one tenant from owning the GPU?
> Evidence to paste: `tenant_tokens` in `control/admit.py`; runs D and G.

> **10b · Q8.7 · Pavi to answer.** Where do I hop, and what is not copied?
> Evidence to paste: `control/hop.py`; `metrics/run_*/hops.jsonl`; Part 6.

> **10b · Q8.8 · Pavi to answer.** Where do I evict, and what becomes a ghost if I skip it?
> Evidence to paste: `cancel()` and `release()` in `control/queue.py`; `relay()` in `control/gateway.py`.

> **10b · Q8.9 · Pavi to answer.** Where does the engine scheduler sit vs my admit / place / queue?
> Evidence to paste: Part 7 diagram; `vllm:num_requests_waiting` vs `gateway_queue_depth` in run G.

> **10b · Q8.10 · Pavi to answer.** What limited concurrency on this GPU for this app?
> Evidence to paste: Part 1 "Expected first limiter"; runs B, G and F2.

> **10b · Q8.11 · Pavi to answer.** Four production alerts you would set.
> Evidence to paste: the "alert" column of the policy ledger in Part 7.

> **10b · Q8.12 · Pavi to answer.** If I scale, which pool: prefill tokens or decode slots?
> Evidence to paste: runs B, F and F2; Part 2 item 3.

> **10b · Q8.13 · Pavi to answer.** What would I change at 10× traffic, and which three knobs are the wrong next move?
> Evidence to paste: the what-if replay in Part 7; `metrics/goodput.json`.

## Submit
