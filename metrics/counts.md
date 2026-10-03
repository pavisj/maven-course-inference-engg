# Request counts per run

| run | client requests | 200 | 429 total | 429 tenant_tokens | 429 over_cap | 429 timeout_queue | 429 kv_free | 503 | 400 (guard) | aborted |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_a` | 1,771 | 1,771 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| `baseline_b` | 3,232 | 3,232 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| `run_c` | 1,286 | 758 | 528 | 248 | 280 | 0 | 0 | 0 | 0 | 0 |
| `run_d` | 8,027 | 2,005 | 6,022 | 6,018 | 0 | 4 | 0 | 0 | 0 | 0 |
| `run_e` | 954 | 305 | 649 | 635 | 0 | 14 | 0 | 0 | 0 | 0 |
| `run_f` | 1,528 | 1,528 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `run_f2` | 2,599 | 1,818 | 781 | 0 | 0 | 0 | 781 | 0 | 0 | 0 |
| `run_g` | 2,747 | 1,317 | 1,430 | 770 | 636 | 24 | 0 | 0 | 0 | 0 |
| `run_h1` | 2,074 | 1,232 | 842 | 809 | 33 | 0 | 0 | 0 | 0 | 0 |
| `run_h2` | 1,887 | 1,367 | 520 | 520 | 0 | 0 | 0 | 0 | 0 | 0 |
| `run_i` | 2,637 | 1,342 | 1,295 | 731 | 544 | 20 | 0 | 0 | 0 | 0 |
| `run_j` | 3,626 | 1,813 | 1,813 | 0 | 1,786 | 27 | 0 | 0 | 0 | 0 |
| `run_l_guard` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 |
| `run_l_abort` | 20 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| `run_m` | 1,537 | 1,537 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `run_m2` | 3,300 | 1,359 | 1,941 | 0 | 1,561 | 380 | 0 | 0 | 0 | 0 |

503: no run had one.
Client counts come from the traffic files. Run L guard and run L abort have none, so their counts come from the gateway rows.
In run L abort, 10 requests were aborted and 10 ended ok. The aborted ones are not counted under 200.
The baselines have no gateway. Their aborted cell shows – and their 400 cell is the client-side count.
