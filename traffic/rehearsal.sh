#!/usr/bin/env bash
# Pipeline rehearsal (overnight Task 5), as one command. Run from the repo root
# once both workers are up:
#
#     bash traffic/rehearsal.sh
#
# 1. Check /health is 200 on every worker in $WORKERS. Stop if not.
# 2. Start control/scrape.py in the background.
# 3. Run each mix for $MIX_SECONDS with $USERS closed-loop users against WorkerA only.
#    48 users is more than the engine's 32 decode slots, so requests queue.
# 4. Stop the scrape.
# 5. Run cluster/warm_probe.py once more (files: warmup_<worker>_$RUN.json).
# Everything goes to $OUT.
#
# Override any setting with an environment variable, e.g. MIX_SECONDS=30 bash traffic/rehearsal.sh
set -euo pipefail

PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-cluster/workers.json}"
OUT="${OUT:-metrics/rehearsal}"
MIX_SECONDS="${MIX_SECONDS:-480}"
USERS="${USERS:-48}"
MIXES="${MIXES:-unique shared_chapter mixed}"
RUN="${RUN:-rehearsal}"   # names the scrape file and the warm-probe files

mkdir -p "$OUT"
LOG="$OUT/$RUN.log"
echo "rehearsal start $(date -u +%Y-%m-%dT%H:%M:%SZ): workers $WORKERS, $MIX_SECONDS s per mix, $USERS users" | tee -a "$LOG"

# 1. health
"$PY" - "$WORKERS" <<'EOF' | tee -a "$LOG"
import json, sys, httpx
bad = []
for name, url in json.load(open(sys.argv[1])).items():
    try:
        code = httpx.get(f"{url}/health", timeout=30).status_code
    except httpx.HTTPError as e:
        code = type(e).__name__
    print(f"{name} /health -> {code}")
    if code != 200:
        bad.append(name)
if bad:
    sys.exit(f"not healthy: {', '.join(bad)}. Stop.")
EOF

TARGET="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['WorkerA'])" "$WORKERS")"

# 2. scrape in the background; stop it on any exit
"$PY" -m control.scrape --workers "$WORKERS" --run "$RUN" --out-dir "$OUT" >> "$LOG" 2>&1 &
SCRAPE_PID=$!
trap 'kill -TERM "$SCRAPE_PID" 2>/dev/null || true' EXIT

# 3. mixes, WorkerA only
for MIX in $MIXES; do
  "$PY" -m traffic.run --mix "$MIX" --target "$TARGET" --duration "$MIX_SECONDS" \
    --users "$USERS" --out-dir "$OUT" 2>&1 | grep -v "PyTorch was not found" | tee -a "$LOG"
done

# 4. stop the scrape
kill -TERM "$SCRAPE_PID"
wait "$SCRAPE_PID" || true
trap - EXIT

# 5. warm probe once more
"$PY" cluster/warm_probe.py --workers "$WORKERS" --out-dir "$OUT" --label "$RUN" 2>&1 | tee -a "$LOG"
echo "rehearsal end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
