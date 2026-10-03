#!/usr/bin/env bash
# Traffic runs through the gateway, as one command. Run from the repo root
# once both workers are up:
#
#     RUN=run_d MIXES="unique shared_chapter mixed" DURATION=240 bash traffic/gateway_run.sh
#
# 1. Start control/gateway.py on 127.0.0.1:8765, writing its logs to $OUT.
# 2. Wait until /health reports every worker warm (at most $WARM_WAIT seconds).
# 3. Start control/scrape.py on both workers and the gateway's /metrics.
# 4. Run each mix in $MIXES for $DURATION seconds with $USERS closed-loop users through the gateway.
# 5. Stop the scrape and the gateway.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
WORKERS="${WORKERS:-cluster/workers.json}"
RUN="${RUN:-run_c}"
OUT="${OUT:-metrics/$RUN}"
MIXES="${MIXES:-mixed}"
CHAPTER="${CHAPTER:-07}"                 # for the shared_chapter mix
DURATION="${DURATION:-240}"              # not SECONDS: bash sets SECONDS itself
USERS="${USERS:-48}"
POLICY="${PLACE_POLICY:-${POLICY:-prefix_then_load}}"   # PLACE_POLICY=least_loaded|p2c|prefix_then_load
LOAD_BOUND="${LOAD_BOUND:-22}"                          # prefix_then_load hops at this many in flight
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-32}"
TENANT_SCALE="${TENANT_SCALE:-1}"         # test lever: multiplies both tenant windows (1 = as designed)
WARM_WAIT="${WARM_WAIT:-600}"
GATEWAY="http://127.0.0.1:8765"

mkdir -p "$OUT"
LOG="$OUT/$RUN.log"
echo "$RUN start $(date -u +%Y-%m-%dT%H:%M:%SZ): $MIXES, $USERS users, $DURATION s each, policy $POLICY, load bound $LOAD_BOUND, $MAX_IN_FLIGHT seats" | tee -a "$LOG"

# 1. gateway
"$PY" -m control.gateway --workers "$WORKERS" --out-dir "$OUT" --policy "$POLICY" --load-bound "$LOAD_BOUND" --max-in-flight "$MAX_IN_FLIGHT" \
  --tenant-window-scale "$TENANT_SCALE" >> "$LOG" 2>&1 &
GATEWAY_PID=$!
SCRAPE_PID=""
trap 'kill -TERM $SCRAPE_PID $GATEWAY_PID 2>/dev/null || true' EXIT

# 2. wait for every worker to be warm
"$PY" - "$GATEWAY" "$WARM_WAIT" <<'EOF' | tee -a "$LOG"
import sys, time, httpx
url, limit = sys.argv[1], float(sys.argv[2])
start = time.time()
while time.time() - start < limit:
    try:
        h = httpx.get(f"{url}/health", timeout=5).json()
        if h["warm"] and all(h["warm"].values()):
            print(f"all warm after {time.time() - start:.0f} s: {h['warm']}")
            sys.exit(0)
    except httpx.HTTPError:
        pass
    time.sleep(2)
sys.exit(f"not all warm after {limit:.0f} s")
EOF

# 3. scrape workers and gateway
"$PY" -m control.scrape --workers "$WORKERS" --gateway "$GATEWAY" --run "$RUN" --out-dir "$OUT" >> "$LOG" 2>&1 &
SCRAPE_PID=$!

# 4. traffic through the gateway
for MIX in $MIXES; do
  "$PY" -m traffic.run --mix "$MIX" --chapter "$CHAPTER" --base-url "$GATEWAY" --duration "$DURATION" \
    --users "$USERS" --out-dir "$OUT" 2>&1 | grep -v "PyTorch was not found" | tee -a "$LOG"
done

# 5. stop scrape, then gateway
sleep 6                       # one more scrape after the last request
kill -TERM "$SCRAPE_PID"; wait "$SCRAPE_PID" || true
kill -TERM "$GATEWAY_PID"; wait "$GATEWAY_PID" || true
trap - EXIT
echo "$RUN end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
