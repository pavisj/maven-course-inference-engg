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

Hypothesis: KV first. Result: max_num_seqs=32 bound first, then compute (TTFT), KV peaked at 14.5 %; KV would bind near 140 sequences.


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
| 1b | `unknown_model` | 400 | `model`, if present, is `tutor` (added for run L: `gpt-4o` used to reach vLLM) | `--served-model-name` |
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
  - **Finding (run L, on GPU):** the gateway does cancel the upstream vLLM request when the client closes. 10 of 20 streams closed at 1.0 s: vLLM running fell from 19 to 10 and KV from 7.7 % to 4.0 % within about 1.5 s, and only the 10 kept requests completed. Two changes before the run: the relay's cleanup is shielded, so a cancelled stream still closes the upstream and releases its seat; and the row says `decision=aborted`. vLLM 0.21 does not count these aborts in `request_success_total{finished_reason="abort"}`. Part 7, "Runs L and M".
- **Which saved memory on the shared-prefix mix?** PagedAttention packs every sequence into 16-token blocks, so KV grows with real length, not with `--max-model-len`. The prefix cache reuses blocks: 56 % to 66 % of prompt tokens were cache hits per worker in runs D to F, so those tokens took no new blocks. On this workload the prefix cache saved more: at 30 sequences, KV stayed at 31 % to 36 % of a halved cache.

### Part 5 questions: Pavi's answers (Step 10b)

**Q5.1 · Who sits in your queue vs vLLM's waiting queue?**

Two queues, two owners, two meanings.

**My queue** , one per worker in the gateway, holds requests that passed guard, admit and place and have no seat yet: the worker already has 32 in flight. It is ordered interactive first, and it can still say no: `timeout_queue` sheds a request whose estimated wait would miss its deadline, before it costs any GPU time.

**vLLM's waiting** , inside the worker, holds requests I have released that the engine has not started: a seat is free, but this step's 4,096-token budget is spent on other prefills, or a KV block is not free yet. It is first come, first served, and it cannot shed. Once a request is there, it is committed.

So the split is: no seat → mine; seat, no prefill turn yet → vLLM's. Measured: Baseline B, 48 users straight at the engine, had 16 in vLLM's waiting. With the gateway in front, vLLM's waiting stayed at 0 in Runs G, H and I. The line moved into my queue: max 8 and mean 1.6 per worker in Run I, max 13 and mean 11 in Run J. Same users, same GPU; the line now sits where it can be reordered and shed.

**Point at:** `control/queue.py`. The gateway queue-depth gauge beside `vllm:num_requests_waiting` in `metrics/run_b/` and `metrics/run_d/`. Notebook, cell 14.

Evidence: `gateway_queue_depth{worker}` and `vllm:num_requests_waiting` from `metrics/run_g/scrape_run_g.jsonl`; plot `plots/run_g/gateway_queue_by_worker.png`.

**Q5.2 · Waiting / running / swapped (or preempted)?**

Four engine states, and three of them tell the whole story of this cluster.

  * **Running:** the sequences in the current batch. Capped at 32, and pinned at 32 from 48 users upward.
  * **Waiting:** released to the engine, not started. Peaked at 16 in Baseline B, straight at the engine. With the gateway in front it stayed at 0 in Runs G, H and I: my queue releases only when a seat is free.
  * **Swapped:** 0\. This vLLM build preempts by recompute, and no CPU swap space is configured, so nothing is ever swapped out.
  * **Preempted:** 0 in every normal run. Preemption happens when KV blocks run out in the middle of decode, and KV never passed 14.5 % on the normal engine. Run F2 forced it: 40 % of GPU memory, 64 seats, KV at 99.8 %, 15 preemptions behind 781 `kv_free` sheds.



Running full, waiting non-zero, preempted zero: that pattern says seat-bound and compute-bound, not memory-bound. Had preempted been above zero, the right move would have been fewer seats or an earlier `kv_free` gate, never another replica.

**Point at:** `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total` in `metrics/run_b/` and `metrics/run_g/`. Notebook, cell 15.

Evidence: `metrics/run_f2/summary.json` (workers: max_running, max_waiting, preemptions); plot `plots/run_f2/preemptions.png`.

**Q5.3 · `orch_replica_queue_depth` (or your equivalent): which pod, under which mix?**

My equivalent of `orch_replica_queue_depth` is the gateway queue depth per worker, reported as `gateway_queue_max` and mean in every run summary, next to `gateway_in_flight_mean`. Depth is what waits for a seat; in flight is what holds one.

**Which pod.** Under the shared-chapter mix with `prefix_then_load`, depth follows the chapter map. Chapters 00, 03a, 03b and 05 are at home on WorkerA; 06, 06b, 07, 09 and 10 on WorkerB. B has five chapters to A's four, so with an even spread of questions B's queue grows first. The load bound of 22 of 32 is what stops it: when the home worker is above the bound, place sends the request to the other worker and records a hop. That is why hops were 28 to 49 % of admitted. Under `least_loaded` (Run H1, chapter 07 only) placement split 652 to A and 580 to B with max in flight 32 and 24; under `prefix_then_load` with bound 28 (H2) it skewed to home: 579 to A and 788 to B, in flight 20 and 28, and hops fell from 53 % to 42 %.

**Under which mix.** Interactive-only traffic queues shallow and short, because every request has the same deadline. The mixed tenant runs queue deeper for `batch`: a batch request sees the whole interactive line ahead of it, so its wait is the interactive line's wait plus its own. Peaks: Run I max 8, mean 1.6 per worker with about 17 seats used; Run J max 13, mean 11.4 on A and 11.0 on B with 30 of 32 seats used all run.

Depth is also a scorer, not only an admit input: place reads it through the load bound, and admit reads it through the Little's-law wait estimate that drives `timeout_queue`.

**Point at:** `control/queue.py` for the gauge. `metrics/run_h1/` and `metrics/run_h2/` for depth per worker by policy. `plots/05_queue_depth_by_pod.png`. Notebook, cell 16.

Evidence: `gateway_queue_depth{worker}` per run; `metrics/run_*/summary.json` (gateway_queue_max per worker); notebook section "Workers".

**Q5.4 · If a 32k RAG retrieve and a short agent decode are both ready, who goes first: you or the engine?**

Both, and in a fixed order: I decide who enters, the engine decides how tokens interleave.

First, the brief's 32k retrieve never reaches the queue. `--max-model-len` is 8,192 and the guard refuses any request whose prompt tokens plus `max_tokens` exceed 8,192, so it dies at the door as `prompt_too_long` with no GPU time spent; Run L shows four of them, 3.6 ms each. The real contest is an 8k retrieve against a short decode request.

**Me.** My queue releases by tenant first, `interactive` ahead of `batch`, then by arrival. It does not look at size. So if the retrieve is interactive and the short request is batch, the retrieve goes first. If both are interactive, the earlier one goes first. What my queue does know about size is the wait: a big prefill ahead inflates the Little's-law estimate for everything behind it, and a request that would miss its deadline is shed as `timeout_queue` rather than released.

**The engine.** Once both are released, vLLM's scheduler owns the step. With chunked prefill, running sequences get their decode token first, then the remaining token budget of 4,096 is filled with prefill chunks. The 8k prefill is cut into about two steps, and the short request's tokens keep flowing between those chunks. So the long retrieve does not stall the short decode; it pays with its own TTFT spread over two steps. I never override that order, and I should not: it is the engine protecting inter-token latency for everyone already running.

**Point at:** `control/guard.py` for the token limit. `control/queue.py` for the tenant order. `--max-num-batched-tokens 4096` in `cluster/modal_vllm.py`. Runs G to J: vLLM waiting stayed at 0 while 30 of 32 seats ran, so released work started at once; the token interleaving inside a step is the engine's chunked-prefill default, a design claim here, not a measured one.

Evidence: `control/queue.py` (priority order); run G gateway TTFT by tenant (`metrics/run_g/summary.json`).

**Q5.5 · PagedAttention packs KV; prefix / radix cache reuses KV. Which one saved memory on your shared-prefix mix?**

Packing saved the memory. Reuse saved the time, and a little memory.

**Packing, the block table.** Without PagedAttention each of the 32 seats would reserve its maximum: 8,192 tokens × 36 KB = 295 MB, so 32 seats = 9.7 GB reserved before any request arrives. With 16-token blocks allocated as the sequence grows, 32 sequences at the app's length need about 32 × 2,400 tokens × 36 KB ≈ 2.8 GB. The measured peak was 14.5 % of the 14.7 GB pool, about 2.1 GB, lower still because shared blocks count once and many answers end early. Packing is the difference between 9.7 GB and 2.1 GB on this GPU.

**Reuse, the prefix cache.** A hit means the engine does not recompute the matching blocks, and resident sequences share them. The 431-token system prompt is 27 blocks, about 15.5 MB, held once per worker instead of once per seat: at 32 seats that is about 0.5 GB not duplicated. The 70 % hit rate at 96 users is mostly a compute saving: about 1,460 of 2,094 prompt tokens skipped prefill on each hit, which is what kept TTFT near the baseline at 8 users and bounded it under load.

So on this shared-prefix mix: packing is why 32 seats fit in 2 GB; reuse is why a 2,094-token prompt costs the GPU like a 630-token one.

**Point at:** `vllm:gpu_cache_usage_perc` at 14.5 % in `metrics/run_b/`; prefix hit rate 70.8 % and 67.2 % in `metrics/run_g/`; the block and byte arithmetic in `DESIGN.md` Part 1. Notebook, cell 18.

Evidence: prefix hit rate per worker (`metrics/run_h1/summary.json`, `metrics/run_h2/summary.json`); KV usage in runs F and F2.

**Q5.6 · Chunked prefill and continuous batching: which engine flags (`max-num-batched-tokens`, `max-num-seqs`) did you use, and why?**

Five flags, one reason each. Continuous batching and chunked prefill are on by default in this vLLM; the flags set their limits.

Flag| Value| Why this value  
---|---|---  
`--max-num-seqs`| 32| The seat cap. Set at the compute edge for 2k-token prompts, where TTFT starts to climb. KV would allow about 140.  
`--max-num-batched-tokens`| 4,096| The step budget. About two prefills of this app's size per step, with room left for every running sequence's decode token. Higher gives more prefill throughput but longer steps and worse inter-token latency; lower spreads one prefill over more steps and raises its TTFT.  
`--max-model-len`| 8,192| 4× the app's prompt: headroom for the batch sweep without shrinking seats. And a guarantee: 32 seats × 8,192 tokens × 36 KB = 9.7 GB, less than the 14.7 GB pool, so the KV cache cannot run out at these flags even if every seat holds a maximum-length sequence.  
`--gpu-memory-utilization`| 0.90| 21.6 GB for weights plus KV; 2.4 GB left for activations and CUDA graphs. 0.95 gained seats I did not need and risked an OOM at startup.  
`--enable-prefix-caching`| on| The reuse in question 18. The system prompt and repeated chapter chunks skip prefill.  
  
That third row is why preemptions were 0 in every normal run: not luck, arithmetic. The flags make the engine's last resort unreachable at 32 seats, which leaves my admit and queue as the only places that say no.

**Point at:** `cluster/modal_vllm.py`, the vLLM command line. `DESIGN.md` Part 2, item 4. The startup log line with the KV token count, pasted in `DESIGN.md` Part 1.

Evidence: `ENGINE_ARGS` and `PROFILES` in `cluster/modal_vllm.py`; Part 1 of this file.

**Q5.7 · KV full after admit: do you shed at the door, or does the engine preempt?**

The door, by design. And at these flags the engine cannot preempt at all.

**The door.** Admit reads the worker's `gpu_cache_usage_perc` from the last scrape and sheds with `kv_free` above 85 %, with a `retry_after`. The scrape is 5 s old at worst, so this gate catches a trend, not a race. It never fired on the normal engine: KV peaked at 14.5 %. It did fire in Run F2, on a pressure profile with 40 % of GPU memory and 64 seats, which is the proof the branch is wired.

**The engine.** vLLM preempts a running sequence when blocks run out between my scrapes, and recomputes it later. That costs the preempted request its decode progress and hurts the p99 of work already running. It fired 0 times, and it could not have: 32 seats × 8,192 tokens × 36 KB is 9.7 GB, and the pool is 14.7 GB. With these flags the cache has room for every seat at maximum length.

**Why the door should fire first when it can.** A shed at the door costs 0 GPU seconds and gives the client a time to come back. A preemption throws away work the GPU already did. So the rule is: set the door's threshold so the engine's preemption stays a backstop, not a policy. If I raise seats toward 96 at 10× traffic, 96 × 8,192 × 36 KB is 29 GB, more than the pool, and preemption becomes possible. That is exactly when the 85 % gate starts doing real work, and when I would lower it.

**Point at:** `control/admit.py`, the `kv_free` branch of `should_shed`. `vllm:num_preemptions_total` at 0 in every normal `metrics/run_*/`, and at 15 in `metrics/run_f2/summary.json` next to 781 `kv_free` sheds. The 9.7 GB vs 14.7 GB line in `DESIGN.md` Part 1. Notebook, cell 20.

Evidence: run F2: 781 `kv_free` sheds and 15 preemptions (`metrics/run_f2/summary.json`); plot `plots/run_f2/kv_usage.png`.

**Q5.8 · Client gone (`aborted`): who frees the KV? And how?**

The engine frees it. The gateway has to tell it.

**How.** When the client closes the stream, the gateway cancels the upstream vLLM request: it closes the HTTP stream to the worker and cancels the task that was reading it. vLLM aborts that sequence, returns its KV blocks to the pool, and the seat opens for the next request. On the gateway side, `relay()` in `control/gateway.py` closes the upstream, `gateway_aborts_total{stage}` counts it, the in-flight count drops, the request row says `decision=aborted` (code 499 since Step 18; Run L's rows still show 200 because the stream had started), and the tenant window keeps the prompt tokens, because the GPU did spend them on the prefill. The engine's own count is `vllm:request_success_total{finished_reason="abort"}`.

**Measured, Run L.** 20 interactive requests at once; the client closed 10 of them at 1.0 s. vLLM running fell from 19 to 10 and KV from 7.7 % to 4.0 % within about 1.5 s. The 10 closed requests had been prefilled, 42,321 prompt tokens between them, and only the 10 kept requests completed. After 11.6 s running and KV were back at 0.

**What went wrong first.** The first abort script only closed when a token arrived, so busy streams closed late. The fix is to close on the client's disconnect, not on the next token. The first attempt is kept in `run_l_abort/attempt1`.

If the gateway did not forward the cancel, the engine would decode the full answer for nobody, hold the seat and the blocks until it finished, and the dashboard would show 32 running with fewer clients than that. That is the leak the brief is asking about.

**Point at:** The cancel path in `control/` (the stream handler). `metrics/run_l_abort/`: the per-second scrape of running and KV, the plot below, and the `decision=aborted` rows.

Evidence: `gateway_aborts_total{stage}`; `relay()` in `control/gateway.py`; `vllm:request_success_total{finished_reason="abort"}` in a scrape.

**Q5.9 · After a worker returns: slam it at 100 %, or ramp while p99 holds?**

Neither a slam nor a timed ramp. A probe decides when the worker exists; the load bound decides how fast it fills.

**First, it is not back when the weights load.** Modal answers `/health` with 503 while the container cold-starts (129.8 s for Overflow before Run J). Then the server answers, and the first requests are still slow while the prefix cache is empty. So the gateway sends real prompts per chapter group and declares the worker warm only when the probe TTFT is at most 1.2 × the steady probe. Until then place does not choose it, so a cold replica never sets the SLO. The probe also seeds the system prompt into the cache.

**Then, traffic arrives by the bound, not by a switch.** A warm worker joins the chapter map and gets its home chapters as they come. Hops from the other worker land on it only when that worker is above 22 of 32. That is a natural ramp: the new worker fills at the rate the old one overflows.

**Where I did slam, and what it cost.** Every run starts with all users at once. In Run J both workers went to 30 of 32 in flight within seconds, and 12 interactive requests that arrived in the first 0.35 s waited the full 6 s in the queue, because the estimator's steady-state seed does not know that the first 32 answers finish together. That start burst is the slam. The next fix is an admission ramp for the first 15 s after a worker is declared warm: release at most the completion rate per second, so p99 holds while the cache fills.

**And re-measure the line each time.** The Run J deploy probed at 229 to 244 ms against 168 to 192 ms before. A worker that "returns" on a different GPU host is a different worker; the warm line has to come from its own steady probe.

**Point at:** `cluster/warm.py`. The warm table in section 6. The Run J report: first-15-seconds analysis and the "Decisions taken" note on the warm line.

Evidence: `metrics/warm_proof.json`; `metrics/run_e/warm_events.jsonl`; the warm gate in Part 6.

## Part 6. Hop and warm-up

**Hop.** On Modal, containers do not share KV memory. A hop means the destination recomputes the prefix. Nothing is copied. The record proves where the recompute happened: `{ts, src: preferred, dst: chosen, tokens: prompt_tokens, prefix: chapter, backend: "recompute"}`, appended to `hops.jsonl` and counted in `gateway_hops_total{src,dst}`.

- Run D: 561 hops, 1,179,270 prompt tokens recomputed away from their home worker.
- Run E (all traffic on chapter 07, home WorkerB): 147 of 305 requests hopped to WorkerA; 310,030 tokens recomputed. WorkerB's prefix hit rate was 59.3 %, WorkerA's 61.2 %.

**Hop cost (Step 17, `metrics/hop_cost.py`, `metrics/hops.json`, `plots/07_hops.png`).** Runs D to L, gateway TTFT from arrival, answered requests only. Runs D to G have no `policy` field; the gateway then always used `prefix_then_load`.

| Policy | Hopped: n, TTFT p50 / p95 | Not hopped: n, TTFT p50 / p95 |
|---|---|---|
| `prefix_then_load` (D, E, F, F2, G, H2, I, J, L) | 4,717; 830 / 4,166 ms | 6,738; 745 / 4,025 ms |
| `least_loaded` (H1) | 652; 528 / 1,974 ms | 580; 531 / 1,512 ms |

- **Tokens recomputed per hop = prompt tokens − 431** (the system prompt, which every worker already caches). 5,382 hops recomputed 8,905,885 tokens: median 1,668 per hop. `metrics/hops.json` holds one record per hop: `run, src, dst, tokens, prefix, backend=recompute`.
- **Under load a hop costs time:** run G hopped p50 1,168 ms against 905 ms at home; run I 1,268 against 1,031 ms. The hop recomputes about 1,700 tokens, and it happens when the home worker is already busy.
- **At light load a hop costs nothing you can see:** in runs D, F, F2 and H2 hopped and home p50 are within 17 ms, and hopped p95 is lower. The hop goes to the less busy worker, which pays back the recompute.
- The comparison is not a controlled test: `prefix_then_load` hops only when the home worker has 22 or more in flight, so hopped requests arrive at busier moments.

**Evict (Step 17, `control/evict.py`, `gateway_affinity_evict_total`).** Nothing is copied, so KV has nothing to evict; the ghost is a stale affinity or warm flag after a Modal container restart.

- A restarted container has an empty prefix cache and cold kernels (first probe about 1,000 ms). If the gateway still treats it as the warm home of its chapters, the first requests for those chapters pay the cold start.
- **The evict:** when a worker fails its warm probe (all 3 rounds), cannot be read, or restarts (its counters go back), the gateway drops that worker's chapter affinity and its warm flag, once per failure, and counts `gateway_affinity_evict_total`. Place reads the warm flag, so the chapters hop to the other worker. When the worker passes its warm probe again, its chapters come back.
- **Count:** the unit test with a fake worker (`tests/test_evict.py`, no GPU) counts 3. The live runs count 0: no worker failed all 3 warm rounds or restarted while a gateway ran (`metrics/evict_count.txt`).

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
- **Per prompt token (E1), Overflow was slower, not equal.** Both prompt medians were about 2,120 tokens. With the trip removed, Overflow took 313 ms per 1,000 prompt tokens (p50), the cluster 195 ms. The cluster reuses its prefix cache (69 % hits); Overflow starts each prompt cold. Every Overflow TTFT with its prompt tokens: `metrics/overflow_per_token.json` (run I: 40 requests, run J: 10).
- **Retries (E1):** the Step 8 retry ("503 → the other worker once → Overflow") now sets `retried_once=true` on the row and counts `gateway_retry_total{src,dst}`. Count in runs C to M2: **0**. No worker answered 503 or 529.
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
- The light-load case, where the floor made the estimate too high (see "Network offset"), did not happen in run J. **Run M tested it:** 62.4 % within 50 % at light load without the floor (see "Runs L and M").

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

### Runs L and M: guard deaths, client abort, and `timeout_queue` that must fire

**Answer: the guard stopped 12 of 12 bad requests with 0 engine calls. A client that leaves frees its engine slot within about 1.5 s. At a 3 s interactive deadline, admit shed 252 interactive requests after warm-up, at 0 GPU seconds each. Run G's 3 late interactive answers cost 15.8 engine-slot seconds for answers past the deadline.**

A second deploy (18:25 to 18:40 IST, approved by Pavithra) ran L, M and M2 with run J's settings. Warm probes: WorkerA 228 ms, WorkerB 179 ms; network trip 346 ms (`metrics/run_m/`). Engine speed depends on which GPU host Modal gives each container.

#### Run L, part 1: guard deaths (`metrics/run_l_guard/`)

| Reason | Sent | Rejected 400 | `gateway_guard_rejects_total` change |
|---|---|---|---|
| `prompt_too_long` (about 9,000 prompt tokens + 512 > 8,192) | 4 | 4 | +4 |
| `unknown_model` (`"model": "gpt-4o"`) | 4 | 4 | +4 |
| `max_tokens_too_large` (2,048 > 512) | 4 | 4 | +4 |

- **0 engine calls:** vLLM's request count and prompt-token count did not change on either worker.
- Each rejection took 3.6 ms (median) in the gateway.
- **Change to the guard:** before run L the guard had no model check, so `gpt-4o` reached vLLM. The new reason `unknown_model` rejects any `model` other than `tutor`. A request without `model` still passes.

#### Run L, part 2: client abort (`metrics/run_l_abort/`, `plots/run_l_abort/running_and_kv.png`)

20 interactive requests at once over the chapters, each asking for about 400 words. The client closed 10 of them at 1.0 s. Scrape every 1 s for 30 s.

| Time after the burst | vLLM running, WorkerA + WorkerB | KV %, WorkerA + WorkerB | Gateway in flight |
|---|---|---|---|
| +1.4 s (0.4 s after the close) | 5 + 14 = 19 | 2.3 + 5.4 = 7.7 | 4 + 15 |
| +2.5 s | 0 + 10 = 10 | 0 + 4.0 = 4.0 | 0 + 10 |
| +11.6 s | 0 | 0 | 0 |

- **Running dropped by 9 and KV % fell by half within about 1.5 s of the close.** The other 10 ran to the end (last one at 10.8 s).
- **Proof that vLLM aborted:** the 10 closed requests reached the engine (42,321 prompt tokens prefilled), but only the 10 kept requests completed. vLLM 0.21 does not count these in `vllm:request_success_total{finished_reason="abort"}`, so the proof is the completion count and the running and KV drop.
- Request rows: 10 × `decision=aborted`, `reason=client_gone`, and 10 × `ok`.
- **Attempt 1 (kept in `metrics/run_l_abort/attempt1/`):** my script checked the close time only when a token arrived. On the busy worker the first token came after 2 to 3 s, so those streams closed late. The abort still worked. The fix: a timer closes the stream at 1.0 s.

#### Run M: `timeout_queue` at a 3 s interactive deadline (`metrics/run_m/`, `metrics/run_m2/`)

Run J had 0 admit-stage `timeout_queue` sheds after its first 15 s. Its only later `timeout_queue` shed was a batch request that waited 60 s in the queue. **Decision:** run M ran, because the admit-stage shed it tests never fired in run J.

- **Setting:** `--interactive-deadline` (env `INTERACTIVE_DEADLINE`) sets the interactive deadline in admit and in the queue together. Default 6 s. One test.
- **Run M (64 users, as in the brief): 0 sheds.** 64 users on 64 seats (32 per worker) never queue, so the estimate stayed near 0.15 s. 1,537 requests, all answered.
- **Run M2 (96 users, run J's load; decision: added after M showed no sheds):** 380 `timeout_queue`.

| `timeout_queue` after the first 15 s, run M2 | Count | Estimate at the shed | Queue on that worker at the shed |
|---|---|---|---|
| Interactive, at admit | 252 (WorkerA 196, WorkerB 56) | 3.001 to 3.481 s (median 3.111 s) | 8 to 13 (median 11); 32 of 32 seats in use; vLLM waiting ≤ 2 |
| Interactive and batch, in the queue | 62 (interactive 53, batch 9) | median 2.62 s at admit | waited the full deadline (3 s or 60 s) |

- With a 3 s deadline and a 4 s `over_cap` trigger, admit test 2 (`timeout_queue`) fires before test 4 (`over_cap`). So no interactive request reached `over_cap` or Overflow in run M2.
- **WorkerA took 196 of the 252.** It was on the slower GPU host, so its completion rate was lower and its estimate higher.

**What a shed saves** (`metrics/run_m2/late_cost.json`; engine-slot seconds = gateway total time − queue wait − network trip):

| | Requests | Engine-slot seconds |
|---|---|---|
| Run M2 interactive `timeout_queue` sheds at admit | 252 | 0 |
| Run G interactive answers with first token after 6 s (client side) | 3 | 15.8 (3.5, 3.7, 8.6 s each) |
| Run M2 interactive answers with first token after 3 s | 99 | 809.9 (median 6.95 s each) |

**A `timeout_queue` shed costs 0 GPU seconds; run G's 3 late interactive answers cost 15.8 engine-slot seconds for answers the user no longer waited for.**

- **Finding: the queue deadline is measured to the seat, not to the first token.** Run G's 3 late answers were estimated at 0.4 to 1.1 s, then waited 5.3 to 6.0 s in the gateway queue. They got a seat just before the 6 s queue deadline, so their first token came after 6 s. Run M2's 99 late answers show the same at 3 s. A fix to consider: the queue releases a request only if deadline − wait is more than the engine's expected TTFT.

**Run M answers the floor question from run J.** Run M ran at light load with the floor removed: est p50 0.15 s against a measured p50 of 0.237 s with the trip removed. 62.4 % of its estimates were within 50 %. The light-load runs with the floor scored 20.0 % to 46.1 % (D, E, F, H1, H2). Removing the floor helps at light load.

| | Run I | Run J | Run M | Run M2 |
|---|---|---|---|---|
| Users; interactive deadline | 96; 6 s | 96; 6 s | 64; 3 s | 96; 3 s |
| Goodput per minute, total (6 s deadline) | 322.6 | 432.0 | 487.0 | 424.6 |
| Goodput per minute, interactive / batch | 192.5 / 130.1 | 424.6 / 7.4 | 243.7 / 243.4 | 404.5 / 20.1 |
| Shed `tenant_tokens`, interactive / batch | 0 / 731 | 0 / 0 | 0 / 0 | 0 / 0 |
| Shed `over_cap`, interactive / batch | 496 / 48 | 19 / 1,767 | 0 / 0 | 0 / 1,561 |
| Shed `timeout_queue`, interactive / batch | 20 / 0 | 12 / 15 | 0 / 0 | 357 / 23 |
| Gateway TTFT p50 / p95, interactive | 757 / 2,187 ms | 3,506 / 4,557 ms | 607 / 1,534 ms | 1,625 / 3,201 ms |
| Overflow sent / limited / failed | 40 / 496 / 0 | 10 / 19 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Est within 50 %, raw / trip removed | 60.4 % / 47.7 % | 93.2 % / 89.5 % | 0.2 % / 62.4 % | 62.5 % / 43.6 % |
| Network trip | 348 ms | 276 ms | 346 ms | 346 ms |

- Run M2's goodput at its own 3 s deadline: 416.1 per minute.
- **Run M has the best goodput of the four, with 0 sheds and both tenants served.** At 64 users the two workers are full but not over. At 96 users the extra 32 users add sheds and retries, not answers.

### Policy ledger

| Stage | Condition | Action | Constant | Source | Run that tested it | Goodput effect | Alert |
|---|---|---|---|---|---|---|---|
| guard | body malformed | 400 `bad_body` | none | Step 4 | unit tests, stub test | none: never a GPU second | `bad_body` > 1 % of requests (client bug) |
| guard | `X-Tenant` not interactive or batch | 400 `unknown_tenant` | 2 tenants | decision 6 | unit tests | none | any `unknown_tenant` (misconfigured client) |
| guard | prompt + max_tokens > 8,192 | 400 `prompt_too_long` | 8,192 | `--max-model-len` | unit tests | protects KV per sequence | none |
| guard | max_tokens > 512 / 256 | 400 `max_tokens_too_large` | 512 / 256 | decision 6 | unit tests; L (4) | bounds decode time | none |
| guard | model is not `tutor` | 400 `unknown_model` | `tutor` | `--served-model-name` | unit test; L (4) | none: never a GPU second | any `unknown_model` (client points at the wrong API) |
| place | home has < LOAD_BOUND in flight and is warm | go home | LOAD_BOUND 22 (28 in H2) | 70 % of 32 slots | D to I; H1 vs H2 | H2 vs H1: +9.9 hit-rate points, +47.5 goodput / min | hop share > 50 % for 10 min |
| place | home busy or not warm | hop to the lowest-score worker | score = in flight + queue + waiting | Step 5b | D, E, G, I | E: 310,030 tokens recomputed | tokens recomputed / min rising |
| place | no warm worker | 503 → failure overflow | none | Step 7 | none (never happened) | none measured | any `no_warm_worker` |
| admit | tenant tokens + request > window × warm workers | 429 `tenant_tokens`, retry ≥ 2 s | 400,000 / 150,000 per worker | baseline B | D (6,018), G (770), I (731), J at ×2 (0) | D capped at 165 / min; replay ×2: +109 to +129 / min; J at ×2: batch −122.7 / min (batch moved to `over_cap`) | `tenant_tokens` > 20 % of a tenant's requests for 5 min |
| admit | est_ttft > deadline | 429 `timeout_queue` | 6 s / 60 s; Little's law | baseline B p95 | E (14), G (24), I (20), J (0), M2 at 3 s (252 after 15 s) | false sheds at start: fixed twice; J: 0 at admit in the first 15 s; M2: each shed 0 GPU s vs 15.8 engine-slot s for G's 3 late answers | `timeout_queue` > 5 % of requests |
| admit | KV % + request blocks > 85 % | 429 `kv_free` | 85 %; num_gpu_blocks | Part 1 | F2 (781) | F2: 578 / min at KV 99.8 % | KV usage > 90 % or any preemption |
| admit | batch: waiting + gateway queue ≥ 8 | 429 `over_cap` | 8 | Step 5a | G (68), I (48), J (1,767) | protects interactive TTFT; J: starved batch when the queue held 13 | batch admitted = 0 for 1 min |
| admit | interactive: seats full and est_ttft > 2 s | Overflow under the limiter, else 429 `over_cap` | 2 s (J: 4 s); 32 seats | baseline B | G (568 × 429), I (40 sent, 496 × 429), J at 4 s (10 sent, 19 × 429) | replay 2 → 4 s: +121 to +131 / min; J: interactive +232.1 / min, TTFT p50 757 → 3,506 ms | interactive `over_cap` > 10 % |
| queue | 32 in flight on the worker | wait in the gateway; interactive first | 32 seats | `--max-num-seqs` | G, I (max 8 waiting), J (max 13) | interactive p50 716 ms vs batch 1,369 ms (G); J: 3 s queue wait | gateway queue wait p95 > 3 s |
| queue | deadline passes while queued | 429 `timeout_queue` (stage queue) | 6 s / 60 s | Step 6 | G (3), J (27), M2 (62 after 15 s) | J: 12 interactive at start; 15 batch waited 60 s behind interactive. Released just before the deadline = late answer (G: 3; M2: 99) | late answers > 1 % |
| queue | client gone (queued / in flight) | drop / close upstream so vLLM aborts | none | Step 6 | stub test; L (10 of 20 closed) | L: running 19 → 10, KV 7.7 % → 4.0 % within 1.5 s | in-flight aborts spike |
| hop | chosen ≠ chapter home | record `recompute` | backend recompute | Part 6 | D to I | the price of a hop: lost prefix hits | none |
| warm | last probe engine TTFT ≤ 0.211 s | mark warm; place may use it | 0.211 s (J: 0.28 s) | warm probes (run E) | E: 1,007 → 175 ms; J deploy steady at 229 to 244 ms | avoids a 1 s first request | worker not warm > 5 min |
| overflow | 429, 500 or out of memory from a worker | stay: return with Retry-After | 2 s | Step 8 | unit tests | none measured | 500 rate > 0 |
| overflow | 503 or 529 from a worker | the other worker once, then Overflow | 10 per minute | Step 8 | unit tests; Overflow check | none measured (no 503 in runs) | any engine 503 |
| overflow | interactive `over_cap` (capacity) | Overflow under the limiter | 10 per minute | AMA whiteboard | I (40 sent), J (10 sent) | +9.7 goodput / min; replay 10 → 30: +20 / min; J: 29 attempts, so the limiter barely matters at 4 s | overflow limited > 0 for 5 min |

### Goodput

goodput = requests answered within deadline per minute: status 200 and the **first token** within the deadline from arrival (interactive 6 s, batch 60 s; the same deadlines admit and the queue use). Shed, failed or late = 0. Arrival is when the client sent the request, so the client-side TTFT counts. `metrics/goodput.json`, `plots/goodput_vs_offered.png`.

| Run | Users | Offered / min | Goodput / min | Share | Interactive / batch goodput | Per worker (A / B) | Late | Late (gateway, run's deadline) / wasted engine s |
|---|---|---|---|---|---|---|---|---|
| A | 8 | 96.9 | 96.9 | 100 % | 81.2 / 15.6 | 96.9 / direct | 0 | n/a (no gateway) |
| B | 48 | 256.9 | 253.8 | 99 % | 211.3 / 42.5 | 253.8 / direct | 39 | n/a (no gateway) |
| C | 48 | 320.4 | 186.6 | 58 % | 120.8 / 65.8 | 186.6 / none | 9 | 9 / 131.4 |
| D | 48 | 660.9 | 165.1 | 25 % | 143.3 / 21.7 | 77.9 / 87.2 | 0 | 0 / 0 |
| E | 48 | 477.5 | 152.7 | 32 % | 152.7 / none | 73.6 / 79.1 | 0 | 0 / 0 |
| F | 64 | 490.6 | 490.6 | 100 % | 245.3 / 245.3 | 242.8 / 247.9 | 0 | 0 / 0 |
| G | 96 | 671.7 | 321.3 | 48 % | 190.5 / 130.8 | 165.3 / 156.2 | 3 | 2 / 12.1 |
| F2 | 128 | 836.6 | 577.8 | 69 % | 284.9 / 292.9 | 293.2 / 287.8 | 23 | 13 / 186.9 |
| H1 | 48 | 500.3 | 294.3 | 59 % | 294.3 / none | 155.6 / 138.7 | 12 | 12 / 325.9 |
| H2 | 48 | 471.8 | 341.8 | 72 % | 341.8 / none | 144.8 / 197.0 | 0 | 0 / 0 |
| I | 96 | 636.3 | 322.6 | 51 % | 192.5 / 130.1 | 160.7 / 152.7 (+ Overflow 9.7) | 5 | 3 / 30.0 |
| J | 96 | 864.0 | 432.0 | 50 % | 424.6 / 7.4 | 214.7 / 214.9 (+ Overflow 2.4) | 0 | 0 / 0 |
| M | 64 | 487.0 | 487.0 | 100 % | 243.7 / 243.4 | 241.4 / 245.6 | 0 | 21 / 177.0 |
| M2 | 96 | 1,034.9 | 424.6 | 41 % | 404.5 / 20.1 | 213.6 / 211.1 | 5 | 101 / 818.8 |

- Offered load includes retries after a 429, so a run that sheds more offers more.
- Runs F and F2 used tenant windows ×10 (KV tests): their goodput shows what the two workers can deliver without tenant limits, about 490 to 580 per minute.
- Late answers stay small (0 to 23 per run): the gateway sheds work it cannot finish in time instead of answering late.
- **Wasted engine seconds (E2)** = engine time on answers whose first token came after the deadline: gateway total time − queue wait − network trip, summed (`wasted_engine_seconds` in `metrics/goodput.json`). It uses the gateway log and each run's own deadline (M and M2: 3 s), so its late count can differ from the client-side "Late" column. Run G: 2 late at the gateway, 12.1 s; the client saw 3, 15.8 s (Runs L and M).
- **Run J has the highest goodput at 96 users, but 98 % of it is interactive.** Goodput per minute does not show tenant fairness or the TTFT inside the deadline. Read it with goodput by tenant and TTFT p50.

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
| M (floor removed; 64 users; trip 346 ms) | 0.150 s | 0.583 s | 0.237 s | 0.2 % | 62.4 % |
| M2 (floor removed; 96 users, 3 s deadline) | 2.654 s | 1.650 s | 1.304 s | 62.5 % | 43.6 % |

- **The estimate is too high at light load.** With the trip removed, an idle engine answers in 142 to 226 ms, but est_ttft never goes below 0.4 s: that is the Step 4 prefill-backlog floor (`BASE_TTFT_S = 0.4`), which `max(Little, backlog)` keeps. The raw "within 50 %" looked good only because the 0.35 s trip filled that gap.
- Under load (F2, G, I) the queue term dominates and the estimate tracks better.
- Run J removed the floor, but ran full for the whole run. Its 89.5 % comes from the loaded regime, not from the floor removal.
- **Run M tested the light-load case:** without the floor, 62.4 % within 50 % (trip removed), against 20.0 % to 46.1 % for the light-load runs with the floor. The raw number (0.2 %) is low because the 346 ms trip is about twice the idle engine TTFT.

## Part 8. Proof

_Run data for every answer: `metrics/run_*/summary.json`, `metrics/goodput.json`, the notebook. The 13 answers below are Pavi's (Step 10b), kept in sync with docs/answers_22.md._

**Q8.1 · What is the app, and which tokens are shared vs unique?**

The app is a tutor over the course corpus. One question retrieves the top 5 chunks by BM25, the prompt is built system → chunks → question, and the median prompt is 2,094 tokens. That is Track A: prefill-heavy, TTFT is the SLO. Two tenants share one serve path: `interactive` is a student asking, `batch` is a sweep writing 3 practice questions per chapter.

Sharing is decided by prompt position, because vLLM's prefix cache matches blocks from the left and stops at the first difference. The 431-token system prompt is shared by every request on a worker: same bytes, always first. The ~1,500 chunk tokens are shared only with requests on the same chapter, and only when retrieval returns the same chunks in the same order. The question and the answer are unique to the student.

Measured under 96 users, 70.8 % of queried prefix tokens hit the cache on WorkerA and 67.2 % on WorkerB. The system prompt alone is 21 % of the prompt, so most of that hit rate is repeated chapter chunks: the chunk zone does share when the same chapter is asked again.

**Point at:** `app/prompt.py` prints the token count of each part. `metrics/run_g/` gives the prefix hit rate per worker under 96 users: 70.8 % on WorkerA, 67.2 % on WorkerB. Run H put the two policies on the same chapter-07 traffic: `prefix_then_load` raised the hit rate from 75.0 % to 84.9 % and cut TTFT p95 by 811 ms against `least_loaded`; the chunk zone shares, and routing to it pays.

Evidence: Part 1 "Real prompt length"; `app/prompt.py`.

**Q8.2 · What dies at guardrails vs admit vs place vs queue?**

Each stage says no for a different reason, and the dashboard shows each count.

Stage| What dies| Measured| Cost to the GPU  
---|---|---|---  
guard| Malformed work: prompt over the token limit, unknown model name, `max_tokens` above the cap. A 400.| Run L: 12 of 12 rejected, 4 per reason, 0 engine calls, 3.6 ms each. The model-name check did not exist before Run L; I added `unknown_model`.| 0  
admit| Valid work the system cannot take now, four reasons: `tenant_tokens` (429, stays), `over_cap` (429 for batch; interactive goes to Overflow or a 429 when the limiter is full), `kv_free` (never), `timeout_queue` by estimate.| Run I, 96 users: 731 batch `tenant_tokens`, 496 interactive `over_cap` (40 sent to Overflow), 48 batch `over_cap`, 20 `timeout_queue`. Run J: 1,767 batch `over_cap`. Run M2 at a 3 s deadline: 252 interactive `timeout_queue` at admit.| 0  
place| Nothing, in this cluster. With two workers and a load bound, place never returns Shed; when home is above the bound it picks the other worker and records a hop. Place would shed only if no worker were warm.| 0 sheds at place in every run. 436 hops in Run I (33.5 %), 899 in Run J (49.9 %).| a recompute per hop  
queue| Work whose deadline arrives while it waits for a seat: `timeout_queue` at the queue stage, the backstop behind the estimate.| Run J: 12 interactive after 6 s, 15 batch after 60 s, all in the first 15 s start burst. Run G: 3.| 0, but the client waited  
engine| Nothing died inside vLLM: 0 preemptions, 0 engine 503s, 0 failed Overflow calls.| Every run.|   
  
The cheapest request is the one you never serve: everything above the engine row cost the GPU nothing.

**Point at:** `metrics/run_l_guard/` for the guard counts. Sheds by reason and tenant in `metrics/run_i/summary.json`, `metrics/run_j/summary.json`, `metrics/run_m2/`. `plots/03_sheds_by_reason.png`. Hops in `metrics/hops.json`.

Evidence: sheds by stage and reason in `metrics/run_*/summary.json`; the policy ledger in Part 7.

**Q8.3 · Where do I prevent work that will time out? (`timeout_queue`)**

In admit, before the request takes a seat, and again in the queue as a backstop.

**At admit.** `should_shed` estimates the time to first token from the worker's live state with Little's law: the requests ahead (vLLM waiting plus my queue) divided by the worker's completion rate, plus the idle engine time of 0.15 s. If the estimate is past the tenant's deadline, 6 s interactive or 60 s batch, the request is shed as `timeout_queue` with a `retry_after`. Nothing was prefilled; it cost 0 GPU seconds.

**In the queue.** A request whose deadline arrives while it still waits for a seat is shed there. Run J: 12 interactive after 6 s and 15 batch after 60 s, all from the start burst when the estimate runs low.

**What it saves, measured.** Run M2, 96 users at a 3 s deadline: 252 interactive requests shed at admit, each with an estimate of 3.0 to 3.5 s and 8 to 13 ahead in that worker's queue, all 32 seats in use. Those cost 0 engine-slot seconds. The 99 that were admitted and still missed 3 s cost 809.9 engine-slot seconds for answers that arrived too late. In Run G at the 6 s deadline, 3 late answers cost 15.8 engine-slot seconds. That is the whole argument for the gate.

**The estimator's history, because it matters.** The first version counted prefill only and was 3× too low: Run C shed 0. Little's law replaced it. Run I then shed 20 requests in its first 11 s because the completion rate was divided by time since gateway start, including the 12 s warm-up. Fixed with a test. Run I also showed a 0.4 s floor that an idle engine beats (0.14 to 0.23 s). Removed for Run J: 0 admit-stage sheds in the first 15 s, and 93 % of estimates within 50 % of measured (89.5 % with the network trip removed). What remains is the start burst, when the first 32 answers on each worker finish together and the steady-state seed is wrong for a few seconds.

**Point at:** `control/admit.py`: `est_ttft` and the `timeout_queue` branch. `control/queue.py` for the deadline check. `metrics/run_m2/late_cost.json` for the engine-slot seconds. The estimate-vs-measured plot below, from `metrics/run_j/`.

Evidence: `should_shed` test 2 in `control/admit.py`; `QUEUE_DEADLINE_S` in `control/queue.py`; run G `timeout_queue` counts.

**Q8.4 · Where do I protect KV?**

In three places, from cheapest to last resort.

**Before the GPU, in the guard and the engine flags.** The guard rejects prompts above its token limit, and `--max-model-len 8192` caps any one sequence at 8,192 × 36 KB = 295 MB of KV. No single request can take a large share of the 14.7 GB pool.

**At the door, in admit.** `should_shed` reads the worker's KV usage from the last scrape. Above 85 % it sheds with reason `kv_free` and a `retry_after`. This branch never fired on the normal engine, and the arithmetic says why: the pool holds about 399k tokens, so about 140 sequences at the app's length, while `--max-num-seqs 32` stops the engine at 32. KV peaked at 14.5 % at 48 users. It did fire when I squeezed the engine on purpose: Run F2 ran a pressure profile with 40 % of GPU memory and 64 seats, and `kv_free` shed `run_f2:323` and its neighbours at the door. The gate works; the normal cluster never needs it.

**Inside the engine, as last resort.** vLLM's block manager preempts a running sequence when blocks run out. Preemptions were 0 in every normal run, which is the same fact seen from the other side. In Run F2, with the engine squeezed to 40 % of memory and 64 seats, the door shed 781 and the engine preempted 15: the door fired first and took most of the load, which is the order I want.

So on this GPU, for this app, KV is protected by the seat cap long before the KV check. If I raised `--max-num-seqs` toward 140, the `kv_free` gate would become the one that matters.

**Point at:** `control/admit.py`, the `kv_free` branch of `should_shed`. `DESIGN.md` Part 1 for the 36 KB per token and the 140-sequence number. `metrics/run_b/` for KV at 14.5 % and preemptions 0; `metrics/run_c/` sheds by reason, `kv_free` = 0; `metrics/run_f2/summary.json`: 781 `kv_free`, 15 preemptions.

Evidence: `kv_free` in `control/admit.py`; run F2.

**Q8.5 · Where do I prioritize interactive traffic? (`p99_spread`)**

In my queue, before the engine sees the request. Each worker has one priority queue in the gateway. `interactive` goes ahead of `batch`, and the queue releases at most 32 requests to a worker at a time, the same number as `--max-num-seqs`. vLLM serves what it is given in arrival order, so the order I release is the priority. I do not reorder inside the engine; I decide what reaches it.

Three other settings lean the same way. Deadlines: 6 s for interactive, 60 s for batch, and the `timeout_queue` estimate uses each tenant's own deadline. Token windows: 400,000 tokens per minute per warm worker for interactive, 150,000 for batch. Overflow: only interactive `over_cap` may leave to the Overflow model; batch waits or takes a 429.

Measured: in Run I at 96 users, interactive TTFT p95 was 2,187 ms and batch 5,444 ms; the spread is the priority working. In Run J the spread stayed (4,557 vs 6,624 ms) but batch goodput fell to 7.4 per minute: priority at the queue plus `over_cap` at 8 waiting left batch nothing. Protection for interactive needs a floor for batch too.

**Point at:** `control/queue.py`, the priority comparison and the 32 in-flight bound. `control/admit.py` for deadlines and windows. The per-tenant latency table in the Steps 5 to 8 report.

Evidence: `PRIORITY` in `control/queue.py`; run G TTFT by tenant; capacity overflow in run I.

**Q8.6 · Where do I stop one tenant from owning the GPU?**

In admit, with a token window per tenant, not a request count. `interactive` may spend 400,000 tokens per minute per warm worker, `batch` 150,000; with two warm workers that is 800,000 and 300,000. Baseline B charged 337,000 interactive and 290,000 batch tokens a minute, so the interactive window has about 19 % headroom and the batch window holds batch to about half its demand. A request whose prompt plus `max_tokens` would push its tenant over the window is refused at the door with `429 tenant_tokens` and a `retry_after` jittered by ±30 %, so the retries do not return as one wave. The router never runs for that request; it costs no GPU time.

Tokens, not requests, because one request is not one unit of GPU work: a 2,094-token prompt costs about fifty times a 40-token one in prefill. Counting requests would let a batch sweep of long prompts look as cheap as a student's short question.

A 429 stays home. It is my own limit speaking, so sending it to the Overflow model would hide the limit and bill me for it. Only capacity codes, 503 and 529, may leave.

Measured: Run C refused 248 requests as `tenant_tokens` under 48 users; Run I refused 731 batch requests at 96 users. Run J doubled both windows: `tenant_tokens` went to 0, and batch goodput fell from 130.1 to 7.4 per minute anyway, because `over_cap` took over as the binding gate. The window was never the limit that mattered for batch; the seats were.

**Point at:** `control/admit.py`, the tenant window and the `tenant_tokens` branch. `metrics/run_c/` sheds by reason. Run J comparison table, windows ×1 vs ×2.

Evidence: `tenant_tokens` in `control/admit.py`; runs D and G.

**Q8.7 · Where do I hop, and what is not copied?**

At the place → queue boundary, and nothing is copied.

**Where.** Place knows each chapter's home worker. When the home is above the load bound of 22 of 32, place picks the other worker and the queue writes a hop record: src is the home, dst is the chosen worker, the token count, the prefix id, and the backend, `recompute`. The record is the hop. Modal containers do not share GPU memory and there is no Mooncake between them, so no KV crosses the wire.

**What is not copied.** The home worker's cached blocks for the chapter chunks, about 1,500 tokens, and the question. The destination recomputes them. What the destination reuses is only what it already holds: the 431-token system prompt, always, and any chapter chunks it has seen before. The hop record says what the brief asks: from where, to where, how many tokens, which prefix, which backend.

**What it costs, measured.** Run I: 436 hops, 33.5 % of admitted, 892,209 tokens recomputed. Run J, with every seat full: 899 hops, 49.9 %, and the prefix hit rate fell from about 69 % to 53 % on both workers. Full workers hop more, and each hop is a cold prefill on the destination. The hopped-versus-home split, Runs D to L, 5,382 hops and 8,905,885 tokens recomputed: under `prefix_then_load` hopped requests ran 830 ms p50 against 745 ms at home (4,166 vs 4,025 ms p95); under `least_loaded` 528 vs 531 ms. At light load a hop costs nothing you can see: in Runs D, F, F2 and H2 hopped and home p50 are within 17 ms, because the hop lands on the less busy worker, which pays back the recompute. Under load the gap opens: Run G hopped p50 1,168 ms against 905 ms at home, Run I 1,268 against 1,031 ms. Each hop recomputes a median of 1,668 prompt tokens, everything but the system prompt. The cleanest pair is in section 2: `run_i:504` at home, 0.575 s; `run_i:611` hopped, 1.35 s. Not a controlled test: `prefix_then_load` hops only when the home worker is busy, so hopped requests also met busier queues.

**The counter-example.** Run H2 kept requests at home with a bound of 28 instead of 22: hops fell from 53 % to 42 %, the hit rate rose from 75 % to 85 %, and TTFT p95 fell by 811 ms. Hopping is what the load bound costs; prefix routing is what it buys back.

**Point at:** `control/place.py` (`preferred` vs `chosen`), `control/queue.py` (the record), `metrics/hops.json`, the Router and Hop panels, Run H and Run J tables.

Evidence: `control/hop.py`; `metrics/run_*/hops.jsonl`; Part 6.

**Q8.8 · Where do I evict, and what becomes a ghost if I skip it?**

In the gateway's beliefs, not in GPU memory.

With a `recompute` backend no KV ever crosses workers, so there is no copied block to evict. vLLM evicts its own prefix-cache blocks, least recently used, when its pool fills; it never filled here (KV peaked at 14.5 %). What the gateway can hold stale is affinity: the belief that a worker is warm and holds a chapter's chunks.

**Where I evict.** When a worker fails its warm probe or its container is replaced, the gateway drops that worker's chapter affinity and its warm flag, and counts it in `gateway_affinity_evict_total` . The triggers are a failed warm probe, an unreadable worker, or a restart. The unit test with a fake worker counts 3 evictions; the live runs count 0, because no worker died mid-run.

**The ghost if I skip it.** Place keeps sending chapter 07 to WorkerB because it believes the chunks are cached there. The new container is cold. Every "hit" is a recompute at cold speed, the hit rate falls, and nothing in the placement log shows why. A ghost cache in the router's head.

**It happened today, in a milder form.** The Run J deploy replaced both workers. Their steady probes ran at 229 to 244 ms, against 168 to 192 ms before, on a slower GPU host. A stored warm line of 0.211 s would have called both workers never-warm, and a stored warm flag would have called them warm while cold. The fix in session was to recompute the line from a fresh steady probe: 0.28 s for that deploy. Same lesson as evict: the belief has to be re-earned when the box changes.

**Point at:** `cluster/warm.py`, `control/place.py` for the affinity map, the evict counter and its test from Step 17, and the "Decisions taken" note in the Run J report.

Evidence: `cancel()` and `release()` in `control/queue.py`; `relay()` in `control/gateway.py`.

**Q8.9 · Where does the engine scheduler sit vs my admit / place / queue?**

In a different process, on a different machine, behind an HTTP call. My gateway runs on the laptop at `127.0.0.1:8765`; each worker is a Modal container running vLLM. The gateway answers three questions: does this request enter (guard, admit), on which worker (place), and when is it released (my queue, at most 32 in flight per worker). vLLM's scheduler answers one: of the requests released to it, which tokens run in the next step. It holds them in `waiting`, moves them to `running` when there is a seat under `--max-num-seqs 32` and free KV blocks, fills each step up to `--max-num-batched-tokens 4096` with chunked prefill, and preempts if blocks run out.

I never touch its policy. No FCFS, DRR or chunked-prefill logic lives in `control/`; I set its flags and I shape its input. The boundary is visible as two gauges: my queue depth per worker, and `vllm:num_requests_waiting`.

Measured: Baseline B drove 48 users at the engine before the gateway existed: 32 running, 16 sat in vLLM's `waiting`. With the gateway in front, the excess sits in my queue instead: vLLM's waiting stayed at 0 in Runs G, H and I, while my queue held up to 8 (G, I) and 13 (J) per worker.

**Point at:** `control/` has guard, admit, place and queue, and no scheduler. `cluster/modal_vllm.py` has the engine flags. `metrics/run_b/` next to `metrics/run_d/`: `vllm:num_requests_waiting` beside the gateway queue-depth gauge.

Evidence: Part 7 diagram; `vllm:num_requests_waiting` vs `gateway_queue_depth` in run G.

**Q8.10 · What limited concurrency on this GPU for this app?**

My own seat cap, `--max-num-seqs 32`, and behind it prefill compute. Not KV.

The numbers in order. At 8 users the engine TTFT was 0.15 s and KV sat at 4.1 %. At 48 users, running pinned at 32, 16 more waited, KV reached 14.5 %, preemptions stayed at 0, and TTFT rose with the line: at 96 users (Run G) the engine TTFT mean was 246 to 286 ms and the gateway p50 976 ms with the queue in front of it. The pool could hold about 140 sequences at the app's length, so KV had 6× headroom when the seats ran out.

Why compute runs out first, on paper: a prefill costs about 2 × parameters × tokens. For 3.1B parameters and the 1,660 uncached tokens left after the system prompt hits the cache, that is about 10 TFLOP. An A10G does about 70 TFLOPS in bf16, so one prefill alone takes about 0.15 s, which is the baseline I measured. Under load the engine fits about two such prefills in each 4,096-token step, so it serves roughly six to seven new prompts a second. 48 users asking faster than that fill 32 seats, and the rest wait. That is compute binding, and it shows up as 32 running plus a waiting line, with KV flat.

So the limiter was the seat cap I set to sit at the compute edge. The GPU's memory was never the constraint for this app.

**Point at:** `metrics/run_a/` and `metrics/run_b/`: running, waiting, KV %, preemptions, TTFT. `DESIGN.md` Part 1 for the FLOP estimate. The flags in `cluster/modal_vllm.py`.

Evidence: Part 1 "Expected first limiter"; runs B, G and F2.

**Q8.11 · Four production alerts you would set.**

Each one would have paged on a run I actually did.

Alert| Rule| Which run pages it  
---|---|---  
Interactive TTFT, the SLO| Gateway TTFT p95, interactive, above 3 s for 2 minutes. Half the 6 s deadline.| Run J: p95 4,557 ms. Goodput looked fine; this alert would have said otherwise.  
Starvation by tenant| Any tenant's goodput below 20 % of its offered load for 2 minutes.| Run J: batch 130.1 → 7.4 per minute while total goodput rose 34 %.  
Shed rate| `over_cap` plus `timeout_queue` above 10 % of arrivals for 1 minute, with the reason label on the alert.| Run J: 1,767 batch `over_cap`. Run I: 496 interactive `over_cap`, 93 % of them still a 429 because the limiter took only 40.  
Warm| Any worker not warm for 5 minutes (the ledger's value), or a steady probe more than 1.2 × its last line.| The Run J deploy: probes at 229 to 244 ms against a 0.211 s line. Cold or slow workers get caught before place sends them traffic.  
  
And one quota alert that never fired on the normal engine and must exist anyway: `gpu_cache_usage_perc` above 90 % on any worker, or any preemption at all. It is the "what is scarce here, and does anything measure it?" alert. Today the scarce thing was seats, and the seat alert is the shed rate above.

**Point at:** Metric names in `control/`: gateway TTFT histogram by tenant, goodput by tenant from `metrics/goodput.json`, sheds by reason, the warm flag and probe TTFT, `vllm:gpu_cache_usage_perc`. The policy ledger in `DESIGN.md` Part 7 has an alert column with one rule per stage; these four are the ones that would have paged on a real run.

Evidence: the "alert" column of the policy ledger in Part 7.

**Q8.12 · If I scale, which pool: prefill tokens or decode slots?**

Decode slots first. Seats, not GPUs.

The Class 9 rule: decode slots hot → scale decode; uncached prefill tokens hot → scale prefill. On this cluster the seats were what ran out. Run F, 64 users on 64 seats: 100 % goodput, 0 sheds. Run J, 96 users on the same 64 seats: 30 of 32 in flight on both workers for the whole run, 11 to 13 waiting at each door, interactive TTFT p50 3.5 s, most of it queue wait. And KV sat at 14.3 %. A seat is held for the whole decode, about 150 to 300 answer tokens at 10 to 25 ms a step, so seats fill long before memory does.

So the first move is `--max-num-seqs` from 32 toward 64: more decode slots on the same GPU, at no memory cost, because 64 seats × 8,192 tokens × 36 KB is 19 GB against a 14.7 GB pool only in the worst case, and the app's sequences are a third of that length. Run F2 ran 64 seats and showed the next wall when memory was also cut.

Prefill compute is the wall behind the seats. One uncached prefill of about 1,700 tokens takes the GPU about 0.15 s, so a worker can start at most six or seven prompts a second whatever the seat count. When engine TTFT, not queue wait, drives the SLO, that is the moment to add prefill capacity, which on Modal means a worker in the chapter map, and with it a real hop store so placement stops costing a recompute.

"Add a replica of the same size" is the wrong verb either way: it adds both pools at once, doubles KV reservation and warmup, and hides which one was short.

**Point at:** `DESIGN.md` Part 2, item 3. `metrics/run_f/` and `metrics/run_j/summary.json`: in flight per worker and KV %. `metrics/run_f2/` for 64 seats under memory pressure.

Evidence: runs B, F and F2; Part 2 item 3.

**Q8.13 · What would I change at 10× traffic, and which three knobs are the wrong next move?**

10× is about 480 concurrent students instead of 48. In order:

  1. **Raise`--max-num-seqs` toward 64, then 96,** and measure where interactive TTFT p95 breaks. KV allows about 140, so the seat cap is the first free knob. Expected effect: more throughput per worker at a higher TTFT; the what-if replay bounds it.
  2. **Order chunks by chunk id, not by BM25 score,** so two questions on one chapter share a longer prefix. That cuts uncached prefill tokens, which is the scarce resource. App change, no GPU.
  3. **Then add prefill capacity:** more workers in the chapter map, and with them a real hop store (Mooncake), so chapter affinity stops costing a recompute on every hop.
  4. **Keep the control plane honest at scale:** tenant windows scale with the fleet, the overflow limiter scales with the fleet, and `timeout_queue` keeps shedding work that cannot make its deadline before it burns GPU seconds.



The three wrong knobs:

  * **Raising`--max-model-len`.** Prompts are 2k tokens; a longer limit only shrinks the seats the KV math allows.
  * **Speculative decoding.** It speeds up decode. The SLO here is TTFT, which is prefill.
  * **Routing 429s to Overflow.** A 429 is my limit speaking. Sending it elsewhere hides the signal and bills me for it.

**Point at:** `metrics/whatif.json` and the Step 14 table: first-order goodput change for `LOAD_BOUND` 22 vs 28, the limiter 10 vs 30 per minute, tenant windows ×1 vs ×2. Run J tested them against a real run: the trigger change held in sign and shed count but was 2× too small (closed-loop load); tenant windows ×2 had the wrong sign (batch died at `over_cap` instead); the limiter premise vanished (only 29 attempts). One-knob replays bound, they do not predict.

Evidence: the what-if replay in Part 7; `metrics/goodput.json`.

## Submit
