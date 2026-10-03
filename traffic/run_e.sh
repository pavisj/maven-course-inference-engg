#!/usr/bin/env bash
# Run E: hop and warm proof. Run from the repo root with both workers up:
#
#     bash traffic/run_e.sh
#
# 1. Cold-start WorkerB only: stop its container; Modal starts a fresh one (cluster/cold_start.py).
# 2. cluster/warm.py on both workers right away: WorkerB is cold, WorkerA is warm.
#    Saved to metrics/run_e/warm_proof.json and copied to metrics/warm_proof.json.
# 3. 2 minutes of shared_chapter traffic on chapter 07 (home: WorkerB) through the gateway.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-metrics/run_e}"
mkdir -p "$OUT"

"$PY" cluster/cold_start.py --worker WorkerB 2>&1 | tee "$OUT/cold_start.log"
"$PY" cluster/warm.py --out "$OUT/warm_proof.json" --label "right after a cold start of WorkerB only" 2>&1 \
  | grep -v "PyTorch was not found" | tee "$OUT/warm_proof.log"
cp "$OUT/warm_proof.json" metrics/warm_proof.json
RUN=run_e OUT="$OUT" MIXES=shared_chapter CHAPTER=07 DURATION=120 USERS=48 bash traffic/gateway_run.sh
