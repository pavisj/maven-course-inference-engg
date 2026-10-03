# Overnight report: 2026-10-03, 03:35 to about 04:00 IST

**No rehearsal ran, so tonight produced no engine numbers. Nothing here is a baseline.** The brief expected eager-mode workers (`FAST_BOOT=True`) and asked that tonight's numbers be treated as a tooling rehearsal. Two facts changed that:

- **Workers stopped:** `tutor-cluster` has been stopped since 02:30:20 IST. Both worker URLs return HTTP 503. `modal deploy` was ruled out for tonight, so the workers could not come back.
- **Not eager:** the last deploy used `FAST_BOOT = False` (CUDA graphs on), not `FAST_BOOT=True`. A redeploy from the current code also runs with CUDA graphs on.

All tooling is built and was tested offline against a local stub engine. The stub is a fake HTTP server that uses vLLM 0.21.0's real metric names. Stub output stayed in a scratch folder. **No stub data is in `metrics/`, `plots/` or the notebook.**

## What ran

| Task | Result | Commit |
|---|---|---|
| 1. Part 1 arithmetic | Done. `DESIGN.md` Part 1 has every number with its formula. Assumptions are marked **verify**. | `e49c640` |
| 2. `app/` | Done. Ingest: 9 files, 661 chunks, 166,383 tokens. BM25 top 5. Prompt token report. CLI tested on the stub. `DESIGN.md` Part 1 now has the real prompt length. | `95fd325` |
| 3. `control/scrape.py` | Done. All 15 fields resolve against your real WorkerA scrape. Tested on the stub, including a clean stop on SIGTERM. | `d2a4130` |
| 4. `traffic/` | Done. Each mix ran 30 seconds at concurrency 8 **against the stub**: about 380 requests each, 0 failures. | `c2949db` |
| 5. Rehearsal | **Blocked** (see "What failed"). Added `traffic/rehearsal.sh`, which runs Task 5 as one command, tested end to end on the stub. | `5da6bf6` |
| 6. `plots/make_plots.py` | Done. All five PNGs render from stub data. No PNG is in `plots/`, because there is no real data. | `f36e3cf` |
| 7. `notebook/analysis.ipynb` | Done. Nine Part 5 headings, one code cell each. Every cell ran on stub data and on tonight's empty run folder. | `380693b` |
| 8. Stop billing | Already stopped. `modal app stop tutor-cluster` at 22:21:51 UTC replied "App is already stopped. (Stopped at 2026-10-03 02:30:20+05:30 by 'pavithra-sj')". The container list is empty. | none |
| 9. This report | Done. | this commit |

### Key numbers (tokenizer arithmetic, no engine run)

- **Real prompt:** median 2,094 tokens, from 1,698 to 2,317, over 12 tutor questions. That is the 431-token system prompt, 1,242 to 1,857 tokens of chunks, a 10- to 17-token question, and 13 to 14 template tokens.
- **KV length per sequence:** prompt plus 512 output tokens gives 2,606 tokens at the median and 2,829 at the maximum.
- **Capacity:** at the real length, KV holds 140 to 152 sequences against 32 decode slots. Expected first limiter: prefill compute (**verify** under load).
- **Prefix overlap (stub run, retrieval only):** in `shared_chapter`, 33 % of prompts start with the same first chunk, across 16 different first chunks. In `unique`, the top first chunk opens 13 % of prompts, across 102. So `shared_chapter` should reuse more KV than the system prompt alone (**verify** on the engine).

## What failed, and why

1. **Task 5 rehearsal: not run.** `bash traffic/rehearsal.sh` at 22:18:49 UTC got HTTP 503 from both workers and stopped before any traffic. Evidence: `metrics/rehearsal/rehearsal.log` and `metrics/rehearsal/BLOCKED.md`.
2. **Task 4 "test each mix for 30 s": the stub only.** These runs prove the tooling works, not engine behaviour. Their TTFT numbers are invented by the stub.
3. **Task 6 plots: none from real data.** The script works. `plots/` is empty until a real run exists.
4. **Bugs found and fixed during testing:**
   - `app/prompt.py` counted 2 tokens for the full prompt. transformers 5 returns a dict from `apply_chat_template(tokenize=True)`. Fixed by rendering the template to text, then counting.
   - The notebook's cell 9 crashed when `RUN_DIR` sat outside the repo. Fixed.
   - The prefix hit-rate plot used a padded time window that could take in a scrape from the next mix. Fixed to use the nearest scrape at or before the start, and at or after the end.

## Parked questions (I picked the simplest option)

1. **Activations = 1 GB** in Part 1, as you specified. The deploy log implies about 0.39 GiB of non-KV working memory (19.85 − 5.79 − 13.67 GiB). Keep 1 GB, or use the measured 0.39 GiB?
2. **Python 3.10 pins.** The latest matplotlib (3.11.2) needs Python 3.11, and the latest numpy (2.5.3) needs 3.12. I kept the venv on 3.10 and pinned matplotlib 3.10.9 and numpy 2.2.6. Move the venv to 3.12?
3. **Chunking.** Chunks never cross a page and do not overlap, and page tails stay. 37 of the 661 chunks are under 50 tokens, including table-of-contents dot leaders. BM25 ranked one of them second for a Little's Law query. Drop chunks under some size, add overlap, or allow chunks to cross pages?
4. **`app/chunks.jsonl` is git-ignored,** because it holds the full course-note text and `corpus/` is ignored too. Keep it out of the public repo?
5. **Prompt layout.** The system message is the tutor prompt. The user message holds the chunks, then the question. Chunks are sorted by file, page and position, not by BM25 score, so equal chunk sets give equal prefixes. Keep this?
6. **`max_tokens` = 512 for both tenants.** Should batch practice questions get a different budget?
7. **Sampling.** No temperature is set, so the model's defaults apply. Set `temperature: 0` for repeatable runs?
8. **Tenant is a log label only.** It is not sent to the engine. Fine until the gateway exists?
9. **Question bank.** 8 templates × 48 course topics = 384 questions. Some read awkwardly, for example "How would you measure Mooncake on a live cluster?". Replace with a hand-written bank?
10. **`shared_chapter`** limits retrieval to one chapter, default `07` (`--chapter`). Which chapter should the real run use?
11. **`mixed`** alternates interactive and batch, 50/50. What ratio do you want?
12. **Batch prompt title.** Chapters with two files use the first file's title: `03` uses 03a's, and `06` uses 06's. Retrieval still covers both files. Fine?
13. **Load shape.** Closed loop: each of the 8 threads sends its next request when the last one ends. Do you want open-loop arrivals at a fixed rate for Part 8?
14. **Plot label.** You asked for "eager mode, rehearsal", but the next deploy runs with CUDA graphs on. I made `--label` a flag with no default. Which label?
15. **Jupyter** is not in `requirements.txt`, and nothing in the venv opens notebooks. Which runner do you use: JupyterLab, VS Code or another?
16. **Gateway queue file name.** Notebook cell 1 looks for `gateway_queue*.jsonl` for your own queue. Rename it when the gateway exists.
17. **README** does not yet list the `app/`, `traffic/` and `plots/` commands. Add them now, or after the rehearsal?

## Decisions (2026-10-03, 12:07 IST)

| # | Decision | What changed |
|---|---|---|
| 1 | Measured 0.39 GiB as the main line, 1 GB as the conservative case | `DESIGN.md` Part 1. The main line applies 0.90 to total memory first, as vLLM does: 399,461 KV tokens, 0.36 % above vLLM's 398,016. The Task 1 order would give 411,293 (3.3 % above). |
| 2 | Keep Python 3.10 | Nothing. |
| 3 | Drop chunks under 50 tokens, no overlap | `app/ingest.py`: `MIN_CHUNK_TOKENS = 50`. 37 chunks dropped, 624 remain. |
| 4 | Keep `app/chunks.jsonl` out of git | Nothing. |
| 5 | Keep the prompt order | Nothing. |
| 6 | Interactive 512, batch 256 | `app/tutor.py`: `MAX_TOKENS` per tenant; `traffic/run.py` uses it. Each record logs `max_tokens`. |
| 7 | Temperature 0 | `app/tutor.py`: every request sends `temperature: 0`. |
| 8 | Tenant as a label for now | Nothing; Step 4 turns it into a window. |
| 9 | Templates for traffic; 12 real demo questions later | Nothing yet. |
| 10 | Keep chapter 07 | Nothing. |
| 11 | Keep 1:1 | Nothing. |
| 12 | Titles for 03 and 06 OK | Nothing. |
| 13 | Closed loop, add `--users`, run 48 | `traffic/run.py`: `--users` (default 48; `--concurrency` still works). `traffic/rehearsal.sh`: `USERS=48`. |
| 14 | Plot label: flag, no default | Nothing. |
| 15 | Install Jupyter | `requirements.txt` pins jupyter 1.1.1 and its parts. `jupyter nbconvert --execute` ran all 10 code cells with no errors. |
| 16 | Gateway log name at Step 4 | Nothing yet. |
| 17 | README at Step 10 | Nothing yet. |

**New question:** two full-size table-of-contents chunks remain, both on Week 7 page 2 (350 tokens each). The 50-token floor keeps them. BM25 still ranks one of them second for a Little's Law query. Drop Week 7 page 2, or keep it?

## The three plots to look at first

They exist once the rehearsal has run (`bash traffic/rehearsal.sh`, then `python plots/make_plots.py`):

1. **`plots/prefix_hit_rate_by_mix.png`:** does `shared_chapter` reuse more KV than `unique`? This answers Part 5 question 5, "which one saved memory".
2. **`plots/queue_depth.png`:** does `waiting` rise above 0 at 48 users on one A10 (32 decode slots)? This shows where requests queue: in the engine, or nowhere yet. Part 5 questions 1 to 3 depend on it.
3. **`plots/ttft_by_mix.png`:** TTFT p50 and p95 per mix. It is laptop-measured, so it includes about 0.75 to 0.95 seconds of network round trip. Compare it with the engine-side TTFT before quoting an SLO.

## To resume

```bash
modal deploy cluster/modal_vllm.py      # about 3 to 5 minutes; both GPUs bill from here
python cluster/write_workers.py
bash traffic/rehearsal.sh               # about 26 minutes: 3 mixes x 8 min, plus warm probe
python plots/make_plots.py --run-dir metrics/rehearsal --label "rehearsal"
modal app stop tutor-cluster
```

## Housekeeping

- **Not pushed.** Your settings deny `git push`. Tonight's 8 commits, plus the manifest commit `7c6ee42`, are local on `worktree-step1-cluster`.
- **Not committed:** your `.claude/settings.json` change, and `artifacts/`.
- **`caffeinate`** keeps the Mac awake until about 15:37 IST, and only while it is on AC power.
