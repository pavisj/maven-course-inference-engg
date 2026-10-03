#!/usr/bin/env bash
# One traffic run through the gateway, as one command. Run from the repo root
# once both workers are up:
#
#     RUN=run_c OUT=metrics/run_c bash traffic/gateway_run.sh
#
# 1. Start control/gateway.py on 127.0.0.1:8765, logging to $OUT/gateway_requests.jsonl.
# 2. Start control/scrape.py on both workers and the gateway's /metrics.
# 3. Run $MIX for $DURATION seconds with $USERS closed-loop users through the gateway.
# 4. Stop the scrape and the gateway.
# Everything goes to $OUT.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-cluster/workers.json}"
RUN="${RUN:-run_c}"
OUT="${OUT:-metrics/$RUN}"
MIX="${MIX:-mixed}"
DURATION="${DURATION:-240}"   # not SECONDS: bash sets SECONDS itself
USERS="${USERS:-48}"
GATEWAY="http://127.0.0.1:8765"

mkdir -p "$OUT"
LOG="$OUT/$RUN.log"
echo "$RUN start $(date -u +%Y-%m-%dT%H:%M:%SZ): $MIX, $USERS users, $DURATION s, through $GATEWAY" | tee -a "$LOG"

# 1. gateway
"$PY" -m control.gateway --workers "$WORKERS" --log "$OUT/gateway_requests.jsonl" >> "$LOG" 2>&1 &
GATEWAY_PID=$!
SCRAPE_PID=""
trap 'kill -TERM $SCRAPE_PID $GATEWAY_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "$GATEWAY/health" > /dev/null && break
  sleep 1
done
curl -sf "$GATEWAY/health" | tee -a "$LOG"; echo | tee -a "$LOG"

# 2. scrape workers and gateway
"$PY" -m control.scrape --workers "$WORKERS" --gateway "$GATEWAY" --run "$RUN" --out-dir "$OUT" >> "$LOG" 2>&1 &
SCRAPE_PID=$!

# 3. traffic through the gateway
"$PY" -m traffic.run --mix "$MIX" --base-url "$GATEWAY" --duration "$DURATION" \
  --users "$USERS" --out-dir "$OUT" 2>&1 | grep -v "PyTorch was not found" | tee -a "$LOG"

# 4. stop scrape, then gateway
sleep 6                       # one more scrape after the last request
kill -TERM "$SCRAPE_PID"; wait "$SCRAPE_PID" || true
kill -TERM "$GATEWAY_PID"; wait "$GATEWAY_PID" || true
trap - EXIT
echo "$RUN end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
