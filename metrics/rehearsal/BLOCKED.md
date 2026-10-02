# Rehearsal: blocked on 2026-10-03

The overnight pipeline rehearsal (Task 5) did not run. No scrape, traffic or warm-up data from it exists.

- **Why:** `tutor-cluster` was stopped at about 02:30 IST on 2026-10-03, after Step 1. Both worker URLs return HTTP 503 (no containers). The overnight instructions ruled out `modal deploy`, so the workers could not come back.
- **Evidence:** `rehearsal.log` in this folder: `bash traffic/rehearsal.sh` at 2026-10-02T22:18:49Z, both `/health` checks returned 503, and the script stopped before any traffic.
- **Premise check:** the overnight brief said the workers run with `FAST_BOOT=True`. The last deploy used `FAST_BOOT = False` (commit "Turn off FAST_BOOT after first deploy"). A redeploy from the current code runs with CUDA graphs on.

To run it once the workers are up:

```bash
modal deploy cluster/modal_vllm.py
python cluster/write_workers.py
bash traffic/rehearsal.sh          # about 26 minutes: 3 mixes x 8 min, plus warm probe
python plots/make_plots.py --run-dir metrics/rehearsal --label "rehearsal"   # label: see OVERNIGHT.md, parked question
modal app stop tutor-cluster
```
