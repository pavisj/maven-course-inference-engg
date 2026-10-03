# One real request per exit

Each story is the first matching gateway row that arrived at least 30 seconds after the first row of its run. If no match was that late, it is the first match at all.

| exit | id | tenant | chapter | prompt tokens | decision at each stage | est_ttft s | queue_wait_s | ttft_measured s | total_s | final code | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. guard reject | `run_l_guard:1` | interactive | 06 | – | guard 400 prompt_too_long (stops at the guard) | – | – | – | – | 400 | run L guard |
| 2. 429 tenant_tokens | `run_i:475` | batch | 09 | 1954 | guard ok → place WorkerB (home) → admit 429 tenant_tokens (est_ttft 0.4 s) | 0.4 | – | – | – | 429 | run I |
| 3. 429 over_cap (batch) | `run_i:1` | batch | 07 | 2042 | guard ok → place WorkerA (hopped from WorkerB) → admit 429 over_cap (est_ttft 3.11 s) | 3.11 | – | – | – | 429 | run I |
| 4. over_cap → Overflow sent | `run_i:614` | interactive | 05 | 2196 | guard ok → place WorkerA (home) → admit over_cap (est_ttft 4.063 s) → Overflow sent (overflow:qwen2.5-1.5b) | 4.063 | – | 0.9705 | 1.5848 | 200 | run I |
| 5. over_cap → Overflow limited, 429 | `run_i:621` | interactive | 05 | 1950 | guard ok → place WorkerB (hopped from WorkerA) → admit 429 over_cap (est_ttft 2.15 s) → Overflow limited → 429 | 2.15 | – | – | – | 429 | run I |
| 6. 429 timeout_queue | `run_m2:535` | interactive | 09 | 2045 | guard ok → place WorkerA (hopped from WorkerB) → admit 429 timeout_queue (est_ttft 3.126 s, queue_depth 11, in_flight 32) | 3.126 | – | – | – | 429 | run M2 |
| 7. placed on WorkerA at home | `run_i:504` | interactive | 03b | 2104 | guard ok → place WorkerA (home) → admit ok (est_ttft 0.4 s) → queue wait 0.0062 s → engine ok | 0.4 | 0.0062 | 0.5748 | 2.5097 | 200 | run I |
| 8. placed on WorkerB, hopped | `run_i:611` | interactive | 05 | 2314 | guard ok → place WorkerB (hopped from WorkerA) → admit ok (est_ttft 0.4 s) → queue wait 0.0058 s → engine ok | 0.4 | 0.0058 | 1.3495 | 3.8921 | 200 | run I |
| 9. retried_once (503 → other worker) | none | – | – | – | never happened: no worker answered 503 or 529 in runs C to M2. Closest test: `test_classify_stay_and_leave` in tests/test_overflow.py. It checks only that status 503 and 529 classify as leave. No test covers the full path 503 → other worker → Overflow. | – | – | – | – | – | runs C to M2 |
| 10. 503 → failure Overflow | none | – | – | – | never happened: no worker answered 503 or 529 in runs C to M2. Closest test: `test_classify_stay_and_leave` in tests/test_overflow.py. It checks only that status 503 and 529 classify as leave. No test covers the full path 503 → other worker → Overflow. | – | – | – | – | – | runs C to M2 |
| 11. aborted | `run_l_abort:1` | interactive | 00 | 2190 | guard ok → place WorkerA (home) → admit ok (est_ttft 0.15 s) → queue wait 0.0023 s → engine aborted (client_gone) | 0.15 | 0.0023 | – | 0.9024 | 200 | run L abort |
| 12. 429 kv_free | `run_f2:323` | batch | 00 | 2016 | guard ok → place WorkerA (home) → admit 429 kv_free (est_ttft 0.4 s) | 0.4 | – | – | – | 429 | run F2 |

kv_free on the normal engine: never fired, needs ≈140 sequences, cap is 32. It fired only in run F2 (pressure2 engine: 0.4 GPU memory, 64 seats).

No match arrived 30 seconds after the first row, so the first match was used: `run_l_guard:1` (exit 1), `run_i:1` (exit 3), `run_l_abort:1` (exit 11).
Run L's aborted row shows code 200: the gateway then recorded 200 once the stream started. Since step 18 an aborted row records 499.
The guard row has no prompt tokens because the gateway records them only after the guard passes.
