# The 22 answers

Copied from the presentation page "Tutor Gateway — final project", section 7. Part 8: questions 1 to 13. Part 5: questions 14 to 22. Each answer ends with the file or scrape to point at.


## Part 8 · Q1 (1 of 22). What is the app, and which tokens are shared vs unique?

The app is a tutor over the course corpus. One question retrieves the top 5 chunks by BM25, the prompt is built system → chunks → question, and the median prompt is 2,094 tokens. That is Track A: prefill-heavy, TTFT is the SLO. Two tenants share one serve path: `interactive` is a student asking, `batch` is a sweep writing 3 practice questions per chapter.

Sharing is decided by prompt position, because vLLM's prefix cache matches blocks from the left and stops at the first difference. The 431-token system prompt is shared by every request on a worker: same bytes, always first. The ~1,500 chunk tokens are shared only with requests on the same chapter, and only when retrieval returns the same chunks in the same order. The question and the answer are unique to the student.

Measured under 96 users, 70.8 % of queried prefix tokens hit the cache on WorkerA and 67.2 % on WorkerB. The system prompt alone is 21 % of the prompt, so most of that hit rate is repeated chapter chunks: the chunk zone does share when the same chapter is asked again.

**Point at:** `app/prompt.py` prints the token count of each part. `metrics/run_g/` gives the prefix hit rate per worker under 96 users: 70.8 % on WorkerA, 67.2 % on WorkerB. Run H put the two policies on the same chapter-07 traffic: `prefix_then_load` raised the hit rate from 75.0 % to 84.9 % and cut TTFT p95 by 811 ms against `least_loaded`; the chunk zone shares, and routing to it pays.


## Part 8 · Q2 (2 of 22). What dies at guardrails vs admit vs place vs queue?

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


## Part 8 · Q3 (3 of 22). Where do I prevent work that will time out?

In admit, before the request takes a seat, and again in the queue as a backstop.

**At admit.** `should_shed` estimates the time to first token from the worker's live state with Little's law: the requests ahead (vLLM waiting plus my queue) divided by the worker's completion rate, plus the idle engine time of 0.15 s. If the estimate is past the tenant's deadline, 6 s interactive or 60 s batch, the request is shed as `timeout_queue` with a `retry_after`. Nothing was prefilled; it cost 0 GPU seconds.

**In the queue.** A request whose deadline arrives while it still waits for a seat is shed there. Run J: 12 interactive after 6 s and 15 batch after 60 s, all from the start burst when the estimate runs low.

**What it saves, measured.** Run M2, 96 users at a 3 s deadline: 252 interactive requests shed at admit, each with an estimate of 3.0 to 3.5 s and 8 to 13 ahead in that worker's queue, all 32 seats in use. Those cost 0 engine-slot seconds. The 99 that were admitted and still missed 3 s cost 809.9 engine-slot seconds for answers that arrived too late. In Run G at the 6 s deadline, 3 late answers cost 15.8 engine-slot seconds. That is the whole argument for the gate.

**The estimator's history, because it matters.** The first version counted prefill only and was 3× too low: Run C shed 0. Little's law replaced it. Run I then shed 20 requests in its first 11 s because the completion rate was divided by time since gateway start, including the 12 s warm-up. Fixed with a test. Run I also showed a 0.4 s floor that an idle engine beats (0.14 to 0.23 s). Removed for Run J: 0 admit-stage sheds in the first 15 s, and 93 % of estimates within 50 % of measured (89.5 % with the network trip removed). What remains is the start burst, when the first 32 answers on each worker finish together and the steady-state seed is wrong for a few seconds.

**Point at:** `control/admit.py`: `est_ttft` and the `timeout_queue` branch. `control/queue.py` for the deadline check. `metrics/run_m2/late_cost.json` for the engine-slot seconds. The estimate-vs-measured plot below, from `metrics/run_j/`.


## Part 8 · Q4 (4 of 22). Where do I protect KV?

In three places, from cheapest to last resort.

**Before the GPU, in the guard and the engine flags.** The guard rejects prompts above its token limit, and `--max-model-len 8192` caps any one sequence at 8,192 × 36 KB = 295 MB of KV. No single request can take a large share of the 14.7 GB pool.

**At the door, in admit.** `should_shed` reads the worker's KV usage from the last scrape. Above 85 % it sheds with reason `kv_free` and a `retry_after`. This branch never fired on the normal engine, and the arithmetic says why: the pool holds about 399k tokens, so about 140 sequences at the app's length, while `--max-num-seqs 32` stops the engine at 32. KV peaked at 14.5 % at 48 users. It did fire when I squeezed the engine on purpose: Run F2 ran a pressure profile with 40 % of GPU memory and 64 seats, and `kv_free` shed `run_f2:323` and its neighbours at the door. The gate works; the normal cluster never needs it.

**Inside the engine, as last resort.** vLLM's block manager preempts a running sequence when blocks run out. Preemptions were 0 in every normal run, which is the same fact seen from the other side. In Run F2, with the engine squeezed to 40 % of memory and 64 seats, the door shed 781 and the engine preempted 15: the door fired first and took most of the load, which is the order I want.

So on this GPU, for this app, KV is protected by the seat cap long before the KV check. If I raised `--max-num-seqs` toward 140, the `kv_free` gate would become the one that matters.

**Point at:** `control/admit.py`, the `kv_free` branch of `should_shed`. `DESIGN.md` Part 1 for the 36 KB per token and the 140-sequence number. `metrics/run_b/` for KV at 14.5 % and preemptions 0; `metrics/run_c/` sheds by reason, `kv_free` = 0; `metrics/run_f2/summary.json`: 781 `kv_free`, 15 preemptions.


## Part 8 · Q5 (5 of 22). Where do I prioritise interactive traffic?

In my queue, before the engine sees the request. Each worker has one priority queue in the gateway. `interactive` goes ahead of `batch`, and the queue releases at most 32 requests to a worker at a time, the same number as `--max-num-seqs`. vLLM serves what it is given in arrival order, so the order I release is the priority. I do not reorder inside the engine; I decide what reaches it.

Three other settings lean the same way. Deadlines: 6 s for interactive, 60 s for batch, and the `timeout_queue` estimate uses each tenant's own deadline. Token windows: 400,000 tokens per minute per warm worker for interactive, 150,000 for batch. Overflow: only interactive `over_cap` may leave to the Overflow model; batch waits or takes a 429.

Measured: in Run I at 96 users, interactive TTFT p95 was 2,187 ms and batch 5,444 ms; the spread is the priority working. In Run J the spread stayed (4,557 vs 6,624 ms) but batch goodput fell to 7.4 per minute: priority at the queue plus `over_cap` at 8 waiting left batch nothing. Protection for interactive needs a floor for batch too.

**Point at:** `control/queue.py`, the priority comparison and the 32 in-flight bound. `control/admit.py` for deadlines and windows. The per-tenant latency table in the Steps 5 to 8 report.


## Part 8 · Q6 (6 of 22). Where do I stop one tenant from owning the GPU?

In admit, with a token window per tenant, not a request count. `interactive` may spend 400,000 tokens per minute per warm worker, `batch` 150,000; with two warm workers that is 800,000 and 300,000. Baseline B charged 337,000 interactive and 290,000 batch tokens a minute, so the interactive window has about 19 % headroom and the batch window holds batch to about half its demand. A request whose prompt plus `max_tokens` would push its tenant over the window is refused at the door with `429 tenant_tokens` and a `retry_after` jittered by ±30 %, so the retries do not return as one wave. The router never runs for that request; it costs no GPU time.

Tokens, not requests, because one request is not one unit of GPU work: a 2,094-token prompt costs about fifty times a 40-token one in prefill. Counting requests would let a batch sweep of long prompts look as cheap as a student's short question.

A 429 stays home. It is my own limit speaking, so sending it to the Overflow model would hide the limit and bill me for it. Only capacity codes, 503 and 529, may leave.

Measured: Run C refused 248 requests as `tenant_tokens` under 48 users; Run I refused 731 batch requests at 96 users. Run J doubled both windows: `tenant_tokens` went to 0, and batch goodput fell from 130.1 to 7.4 per minute anyway, because `over_cap` took over as the binding gate. The window was never the limit that mattered for batch; the seats were.

**Point at:** `control/admit.py`, the tenant window and the `tenant_tokens` branch. `metrics/run_c/` sheds by reason. Run J comparison table, windows ×1 vs ×2.


## Part 8 · Q7 (7 of 22). Where do I hop, and what is not copied?

At the place → queue boundary, and nothing is copied.

**Where.** Place knows each chapter's home worker. When the home is above the load bound of 22 of 32, place picks the other worker and the queue writes a hop record: src is the home, dst is the chosen worker, the token count, the prefix id, and the backend, `recompute`. The record is the hop. Modal containers do not share GPU memory and there is no Mooncake between them, so no KV crosses the wire.

**What is not copied.** The home worker's cached blocks for the chapter chunks, about 1,500 tokens, and the question. The destination recomputes them. What the destination reuses is only what it already holds: the 431-token system prompt, always, and any chapter chunks it has seen before. The hop record says what the brief asks: from where, to where, how many tokens, which prefix, which backend.

**What it costs, measured.** Run I: 436 hops, 33.5 % of admitted, 892,209 tokens recomputed. Run J, with every seat full: 899 hops, 49.9 %, and the prefix hit rate fell from about 69 % to 53 % on both workers. Full workers hop more, and each hop is a cold prefill on the destination. The hopped-versus-home split, Runs D to L, 5,382 hops and 8,905,885 tokens recomputed: under `prefix_then_load` hopped requests ran 830 ms p50 against 745 ms at home (4,166 vs 4,025 ms p95); under `least_loaded` 528 vs 531 ms. At light load a hop costs nothing you can see: in Runs D, F, F2 and H2 hopped and home p50 are within 17 ms, because the hop lands on the less busy worker, which pays back the recompute. Under load the gap opens: Run G hopped p50 1,168 ms against 905 ms at home, Run I 1,268 against 1,031 ms. Each hop recomputes a median of 1,668 prompt tokens, everything but the system prompt. The cleanest pair is in section 2: `run_i:504` at home, 0.575 s; `run_i:611` hopped, 1.35 s. Not a controlled test: `prefix_then_load` hops only when the home worker is busy, so hopped requests also met busier queues.

**The counter-example.** Run H2 kept requests at home with a bound of 28 instead of 22: hops fell from 53 % to 42 %, the hit rate rose from 75 % to 85 %, and TTFT p95 fell by 811 ms. Hopping is what the load bound costs; prefix routing is what it buys back.

**Point at:** `control/place.py` (`preferred` vs `chosen`), `control/queue.py` (the record), `metrics/hops.json`, the Router and Hop panels, Run H and Run J tables.


## Part 8 · Q8 (8 of 22). Where do I evict, and what becomes a ghost if I skip it?

In the gateway's beliefs, not in GPU memory.

With a `recompute` backend no KV ever crosses workers, so there is no copied block to evict. vLLM evicts its own prefix-cache blocks, least recently used, when its pool fills; it never filled here (KV peaked at 14.5 %). What the gateway can hold stale is affinity: the belief that a worker is warm and holds a chapter's chunks.

**Where I evict.** When a worker fails its warm probe or its container is replaced, the gateway drops that worker's chapter affinity and its warm flag, and counts it in `gateway_affinity_evict_total` . The triggers are a failed warm probe, an unreadable worker, or a restart. The unit test with a fake worker counts 3 evictions; the live runs count 0, because no worker died mid-run.

**The ghost if I skip it.** Place keeps sending chapter 07 to WorkerB because it believes the chunks are cached there. The new container is cold. Every "hit" is a recompute at cold speed, the hit rate falls, and nothing in the placement log shows why. A ghost cache in the router's head.

**It happened today, in a milder form.** The Run J deploy replaced both workers. Their steady probes ran at 229 to 244 ms, against 168 to 192 ms before, on a slower GPU host. A stored warm line of 0.211 s would have called both workers never-warm, and a stored warm flag would have called them warm while cold. The fix in session was to recompute the line from a fresh steady probe: 0.28 s for that deploy. Same lesson as evict: the belief has to be re-earned when the box changes.

**Point at:** `cluster/warm.py`, `control/place.py` for the affinity map, the evict counter and its test from Step 17, and the "Decisions taken" note in the Run J report.


## Part 8 · Q9 (9 of 22). Where does the engine scheduler sit vs my admit, place, queue?

In a different process, on a different machine, behind an HTTP call. My gateway runs on the laptop at `127.0.0.1:8765`; each worker is a Modal container running vLLM. The gateway answers three questions: does this request enter (guard, admit), on which worker (place), and when is it released (my queue, at most 32 in flight per worker). vLLM's scheduler answers one: of the requests released to it, which tokens run in the next step. It holds them in `waiting`, moves them to `running` when there is a seat under `--max-num-seqs 32` and free KV blocks, fills each step up to `--max-num-batched-tokens 4096` with chunked prefill, and preempts if blocks run out.

I never touch its policy. No FCFS, DRR or chunked-prefill logic lives in `control/`; I set its flags and I shape its input. The boundary is visible as two gauges: my queue depth per worker, and `vllm:num_requests_waiting`.

Measured: Baseline B drove 48 users at the engine before the gateway existed: 32 running, 16 sat in vLLM's `waiting`. With the gateway in front, the excess sits in my queue instead: vLLM's waiting stayed at 0 in Runs G, H and I, while my queue held up to 8 (G, I) and 13 (J) per worker.

**Point at:** `control/` has guard, admit, place and queue, and no scheduler. `cluster/modal_vllm.py` has the engine flags. `metrics/run_b/` next to `metrics/run_d/`: `vllm:num_requests_waiting` beside the gateway queue-depth gauge.


## Part 8 · Q10 (10 of 22). What limited concurrency on this GPU for this app?

My own seat cap, `--max-num-seqs 32`, and behind it prefill compute. Not KV.

The numbers in order. At 8 users the engine TTFT was 0.15 s and KV sat at 4.1 %. At 48 users, running pinned at 32, 16 more waited, KV reached 14.5 %, preemptions stayed at 0, and TTFT rose with the line: at 96 users (Run G) the engine TTFT mean was 246 to 286 ms and the gateway p50 976 ms with the queue in front of it. The pool could hold about 140 sequences at the app's length, so KV had 6× headroom when the seats ran out.

Why compute runs out first, on paper: a prefill costs about 2 × parameters × tokens. For 3.1B parameters and the 1,660 uncached tokens left after the system prompt hits the cache, that is about 10 TFLOP. An A10G does about 70 TFLOPS in bf16, so one prefill alone takes about 0.15 s, which is the baseline I measured. Under load the engine fits about two such prefills in each 4,096-token step, so it serves roughly six to seven new prompts a second. 48 users asking faster than that fill 32 seats, and the rest wait. That is compute binding, and it shows up as 32 running plus a waiting line, with KV flat.

So the limiter was the seat cap I set to sit at the compute edge. The GPU's memory was never the constraint for this app.

**Point at:** `metrics/run_a/` and `metrics/run_b/`: running, waiting, KV %, preemptions, TTFT. `DESIGN.md` Part 1 for the FLOP estimate. The flags in `cluster/modal_vllm.py`.


## Part 8 · Q11 (11 of 22). Four production alerts I would set.

Each one would have paged on a run I actually did.

Alert| Rule| Which run pages it  
---|---|---  
Interactive TTFT, the SLO| Gateway TTFT p95, interactive, above 3 s for 2 minutes. Half the 6 s deadline.| Run J: p95 4,557 ms. Goodput looked fine; this alert would have said otherwise.  
Starvation by tenant| Any tenant's goodput below 20 % of its offered load for 2 minutes.| Run J: batch 130.1 → 7.4 per minute while total goodput rose 34 %.  
Shed rate| `over_cap` plus `timeout_queue` above 10 % of arrivals for 1 minute, with the reason label on the alert.| Run J: 1,767 batch `over_cap`. Run I: 496 interactive `over_cap`, 93 % of them still a 429 because the limiter took only 40.  
Warm| Any worker not warm for 5 minutes (the ledger's value), or a steady probe more than 1.2 × its last line.| The Run J deploy: probes at 229 to 244 ms against a 0.211 s line. Cold or slow workers get caught before place sends them traffic.  
  
And one quota alert that never fired on the normal engine and must exist anyway: `gpu_cache_usage_perc` above 90 % on any worker, or any preemption at all. It is the "what is scarce here, and does anything measure it?" alert. Today the scarce thing was seats, and the seat alert is the shed rate above.

**Point at:** Metric names in `control/`: gateway TTFT histogram by tenant, goodput by tenant from `metrics/goodput.json`, sheds by reason, the warm flag and probe TTFT, `vllm:gpu_cache_usage_perc`. The policy ledger in `DESIGN.md` Part 7 has an alert column with one rule per stage; these four are the ones that would have paged on a real run.


## Part 8 · Q12 (12 of 22). If I scale, which pool: prefill tokens or decode slots?

Decode slots first. Seats, not GPUs.

The Class 9 rule: decode slots hot → scale decode; uncached prefill tokens hot → scale prefill. On this cluster the seats were what ran out. Run F, 64 users on 64 seats: 100 % goodput, 0 sheds. Run J, 96 users on the same 64 seats: 30 of 32 in flight on both workers for the whole run, 11 to 13 waiting at each door, interactive TTFT p50 3.5 s, most of it queue wait. And KV sat at 14.3 %. A seat is held for the whole decode, about 150 to 300 answer tokens at 10 to 25 ms a step, so seats fill long before memory does.

So the first move is `--max-num-seqs` from 32 toward 64: more decode slots on the same GPU, at no memory cost, because 64 seats × 8,192 tokens × 36 KB is 19 GB against a 14.7 GB pool only in the worst case, and the app's sequences are a third of that length. Run F2 ran 64 seats and showed the next wall when memory was also cut.

Prefill compute is the wall behind the seats. One uncached prefill of about 1,700 tokens takes the GPU about 0.15 s, so a worker can start at most six or seven prompts a second whatever the seat count. When engine TTFT, not queue wait, drives the SLO, that is the moment to add prefill capacity, which on Modal means a worker in the chapter map, and with it a real hop store so placement stops costing a recompute.

"Add a replica of the same size" is the wrong verb either way: it adds both pools at once, doubles KV reservation and warmup, and hides which one was short.

**Point at:** `DESIGN.md` Part 2, item 3. `metrics/run_f/` and `metrics/run_j/summary.json`: in flight per worker and KV %. `metrics/run_f2/` for 64 seats under memory pressure.


## Part 8 · Q13 (13 of 22). What would I change at 10× traffic, and which three knobs are the wrong next move?

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


## Part 5 · Q1 (14 of 22). Who sits in my queue vs vLLM's waiting queue?

Two queues, two owners, two meanings.

**My queue** , one per worker in the gateway, holds requests that passed guard, admit and place and have no seat yet: the worker already has 32 in flight. It is ordered interactive first, and it can still say no: `timeout_queue` sheds a request whose estimated wait would miss its deadline, before it costs any GPU time.

**vLLM's waiting** , inside the worker, holds requests I have released that the engine has not started: a seat is free, but this step's 4,096-token budget is spent on other prefills, or a KV block is not free yet. It is first come, first served, and it cannot shed. Once a request is there, it is committed.

So the split is: no seat → mine; seat, no prefill turn yet → vLLM's. Measured: Baseline B, 48 users straight at the engine, had 16 in vLLM's waiting. With the gateway in front, vLLM's waiting stayed at 0 in Runs G, H and I. The line moved into my queue: max 8 and mean 1.6 per worker in Run I, max 13 and mean 11 in Run J. Same users, same GPU; the line now sits where it can be reordered and shed.

**Point at:** `control/queue.py`. The gateway queue-depth gauge beside `vllm:num_requests_waiting` in `metrics/run_b/` and `metrics/run_d/`. Notebook, cell 14.


## Part 5 · Q2 (15 of 22). Waiting, running, swapped or preempted?

Four engine states, and three of them tell the whole story of this cluster.

  * **Running:** the sequences in the current batch. Capped at 32, and pinned at 32 from 48 users upward.
  * **Waiting:** released to the engine, not started. Peaked at 16 in Baseline B, straight at the engine. With the gateway in front it stayed at 0 in Runs G, H and I: my queue releases only when a seat is free.
  * **Swapped:** 0\. This vLLM build preempts by recompute, and no CPU swap space is configured, so nothing is ever swapped out.
  * **Preempted:** 0 in every normal run. Preemption happens when KV blocks run out in the middle of decode, and KV never passed 14.5 % on the normal engine. Run F2 forced it: 40 % of GPU memory, 64 seats, KV at 99.8 %, 15 preemptions behind 781 `kv_free` sheds.



Running full, waiting non-zero, preempted zero: that pattern says seat-bound and compute-bound, not memory-bound. Had preempted been above zero, the right move would have been fewer seats or an earlier `kv_free` gate, never another replica.

**Point at:** `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total` in `metrics/run_b/` and `metrics/run_g/`. Notebook, cell 15.


## Part 5 · Q3 (16 of 22). Queue depth: which pod, under which mix?

My equivalent of `orch_replica_queue_depth` is the gateway queue depth per worker, reported as `gateway_queue_max` and mean in every run summary, next to `gateway_in_flight_mean`. Depth is what waits for a seat; in flight is what holds one.

**Which pod.** Under the shared-chapter mix with `prefix_then_load`, depth follows the chapter map. Chapters 00, 03a, 03b and 05 are at home on WorkerA; 06, 06b, 07, 09 and 10 on WorkerB. B has five chapters to A's four, so with an even spread of questions B's queue grows first. The load bound of 22 of 32 is what stops it: when the home worker is above the bound, place sends the request to the other worker and records a hop. That is why hops were 28 to 49 % of admitted. Under `least_loaded` (Run H1, chapter 07 only) placement split 652 to A and 580 to B with max in flight 32 and 24; under `prefix_then_load` with bound 28 (H2) it skewed to home: 579 to A and 788 to B, in flight 20 and 28, and hops fell from 53 % to 42 %.

**Under which mix.** Interactive-only traffic queues shallow and short, because every request has the same deadline. The mixed tenant runs queue deeper for `batch`: a batch request sees the whole interactive line ahead of it, so its wait is the interactive line's wait plus its own. Peaks: Run I max 8, mean 1.6 per worker with about 17 seats used; Run J max 13, mean 11.4 on A and 11.0 on B with 30 of 32 seats used all run.

Depth is also a scorer, not only an admit input: place reads it through the load bound, and admit reads it through the Little's-law wait estimate that drives `timeout_queue`.

**Point at:** `control/queue.py` for the gauge. `metrics/run_h1/` and `metrics/run_h2/` for depth per worker by policy. `plots/05_queue_depth_by_pod.png`. Notebook, cell 16.


## Part 5 · Q4 (17 of 22). A long retrieve and a short decode are both ready: who goes first, me or the engine?

Both, and in a fixed order: I decide who enters, the engine decides how tokens interleave.

First, the brief's 32k retrieve never reaches the queue. `--max-model-len` is 8,192 and the guard refuses any request whose prompt tokens plus `max_tokens` exceed 8,192, so it dies at the door as `prompt_too_long` with no GPU time spent; Run L shows four of them, 3.6 ms each. The real contest is an 8k retrieve against a short decode request.

**Me.** My queue releases by tenant first, `interactive` ahead of `batch`, then by arrival. It does not look at size. So if the retrieve is interactive and the short request is batch, the retrieve goes first. If both are interactive, the earlier one goes first. What my queue does know about size is the wait: a big prefill ahead inflates the Little's-law estimate for everything behind it, and a request that would miss its deadline is shed as `timeout_queue` rather than released.

**The engine.** Once both are released, vLLM's scheduler owns the step. With chunked prefill, running sequences get their decode token first, then the remaining token budget of 4,096 is filled with prefill chunks. The 8k prefill is cut into about two steps, and the short request's tokens keep flowing between those chunks. So the long retrieve does not stall the short decode; it pays with its own TTFT spread over two steps. I never override that order, and I should not: it is the engine protecting inter-token latency for everyone already running.

**Point at:** `control/guard.py` for the token limit. `control/queue.py` for the tenant order. `--max-num-batched-tokens 4096` in `cluster/modal_vllm.py`. Runs G to J: vLLM waiting stayed at 0 while 30 of 32 seats ran, so released work started at once; the token interleaving inside a step is the engine's chunked-prefill default, a design claim here, not a measured one.


## Part 5 · Q5 (18 of 22). PagedAttention packs KV; prefix cache reuses KV. Which one saved memory on my shared-prefix mix?

Packing saved the memory. Reuse saved the time, and a little memory.

**Packing, the block table.** Without PagedAttention each of the 32 seats would reserve its maximum: 8,192 tokens × 36 KB = 295 MB, so 32 seats = 9.7 GB reserved before any request arrives. With 16-token blocks allocated as the sequence grows, 32 sequences at the app's length need about 32 × 2,400 tokens × 36 KB ≈ 2.8 GB. The measured peak was 14.5 % of the 14.7 GB pool, about 2.1 GB, lower still because shared blocks count once and many answers end early. Packing is the difference between 9.7 GB and 2.1 GB on this GPU.

**Reuse, the prefix cache.** A hit means the engine does not recompute the matching blocks, and resident sequences share them. The 431-token system prompt is 27 blocks, about 15.5 MB, held once per worker instead of once per seat: at 32 seats that is about 0.5 GB not duplicated. The 70 % hit rate at 96 users is mostly a compute saving: about 1,460 of 2,094 prompt tokens skipped prefill on each hit, which is what kept TTFT near the baseline at 8 users and bounded it under load.

So on this shared-prefix mix: packing is why 32 seats fit in 2 GB; reuse is why a 2,094-token prompt costs the GPU like a 630-token one.

**Point at:** `vllm:gpu_cache_usage_perc` at 14.5 % in `metrics/run_b/`; prefix hit rate 70.8 % and 67.2 % in `metrics/run_g/`; the block and byte arithmetic in `DESIGN.md` Part 1. Notebook, cell 18.


## Part 5 · Q6 (19 of 22). Chunked prefill and continuous batching: the engine flags I used and why.

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


## Part 5 · Q7 (20 of 22). KV full after admit: do I shed at the door, or does the engine preempt?

The door, by design. And at these flags the engine cannot preempt at all.

**The door.** Admit reads the worker's `gpu_cache_usage_perc` from the last scrape and sheds with `kv_free` above 85 %, with a `retry_after`. The scrape is 5 s old at worst, so this gate catches a trend, not a race. It never fired on the normal engine: KV peaked at 14.5 %. It did fire in Run F2, on a pressure profile with 40 % of GPU memory and 64 seats, which is the proof the branch is wired.

**The engine.** vLLM preempts a running sequence when blocks run out between my scrapes, and recomputes it later. That costs the preempted request its decode progress and hurts the p99 of work already running. It fired 0 times, and it could not have: 32 seats × 8,192 tokens × 36 KB is 9.7 GB, and the pool is 14.7 GB. With these flags the cache has room for every seat at maximum length.

**Why the door should fire first when it can.** A shed at the door costs 0 GPU seconds and gives the client a time to come back. A preemption throws away work the GPU already did. So the rule is: set the door's threshold so the engine's preemption stays a backstop, not a policy. If I raise seats toward 96 at 10× traffic, 96 × 8,192 × 36 KB is 29 GB, more than the pool, and preemption becomes possible. That is exactly when the 85 % gate starts doing real work, and when I would lower it.

**Point at:** `control/admit.py`, the `kv_free` branch of `should_shed`. `vllm:num_preemptions_total` at 0 in every normal `metrics/run_*/`, and at 15 in `metrics/run_f2/summary.json` next to 781 `kv_free` sheds. The 9.7 GB vs 14.7 GB line in `DESIGN.md` Part 1. Notebook, cell 20.


## Part 5 · Q8 (21 of 22). Client gone: who frees the KV, and how?

The engine frees it. The gateway has to tell it.

**How.** When the client closes the stream, the gateway cancels the upstream vLLM request: it closes the HTTP stream to the worker and cancels the task that was reading it. vLLM aborts that sequence, returns its KV blocks to the pool, and the seat opens for the next request. On the gateway side, `relay()` in `control/gateway.py` closes the upstream, `gateway_aborts_total{stage}` counts it, the in-flight count drops, the request row says `decision=aborted` (code 499 since Step 18; Run L's rows still show 200 because the stream had started), and the tenant window keeps the prompt tokens, because the GPU did spend them on the prefill. The engine's own count is `vllm:request_success_total{finished_reason="abort"}`.

**Measured, Run L.** 20 interactive requests at once; the client closed 10 of them at 1.0 s. vLLM running fell from 19 to 10 and KV from 7.7 % to 4.0 % within about 1.5 s. The 10 closed requests had been prefilled, 42,321 prompt tokens between them, and only the 10 kept requests completed. After 11.6 s running and KV were back at 0.

**What went wrong first.** The first abort script only closed when a token arrived, so busy streams closed late. The fix is to close on the client's disconnect, not on the next token. The first attempt is kept in `run_l_abort/attempt1`.

If the gateway did not forward the cancel, the engine would decode the full answer for nobody, hold the seat and the blocks until it finished, and the dashboard would show 32 running with fewer clients than that. That is the leak the brief is asking about.

**Point at:** The cancel path in `control/` (the stream handler). `metrics/run_l_abort/`: the per-second scrape of running and KV, the plot below, and the `decision=aborted` rows.


## Part 5 · Q9 (22 of 22). After a worker returns: slam it at 100 %, or ramp while p99 holds?

Neither a slam nor a timed ramp. A probe decides when the worker exists; the load bound decides how fast it fills.

**First, it is not back when the weights load.** Modal answers `/health` with 503 while the container cold-starts (129.8 s for Overflow before Run J). Then the server answers, and the first requests are still slow while the prefix cache is empty. So the gateway sends real prompts per chapter group and declares the worker warm only when the probe TTFT is at most 1.2 × the steady probe. Until then place does not choose it, so a cold replica never sets the SLO. The probe also seeds the system prompt into the cache.

**Then, traffic arrives by the bound, not by a switch.** A warm worker joins the chapter map and gets its home chapters as they come. Hops from the other worker land on it only when that worker is above 22 of 32. That is a natural ramp: the new worker fills at the rate the old one overflows.

**Where I did slam, and what it cost.** Every run starts with all users at once. In Run J both workers went to 30 of 32 in flight within seconds, and 12 interactive requests that arrived in the first 0.35 s waited the full 6 s in the queue, because the estimator's steady-state seed does not know that the first 32 answers finish together. That start burst is the slam. The next fix is an admission ramp for the first 15 s after a worker is declared warm: release at most the completion rate per second, so p99 holds while the cache fills.

**And re-measure the line each time.** The Run J deploy probed at 229 to 244 ms against 168 to 192 ms before. A worker that "returns" on a different GPU host is a different worker; the warm line has to come from its own steady probe.

**Point at:** `cluster/warm.py`. The warm table in section 6. The Run J report: first-15-seconds analysis and the "Decisions taken" note on the warm line.
