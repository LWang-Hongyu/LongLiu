#!/bin/bash
# ============================================================================
# Experiment C — Scale Ladder Runner (one regime × one arm)
# ============================================================================
# Launches N emulator job-pairs across 10.1 (client) and 226 (server),
# plus the allocation daemon on 10.1. Waits for completion, archives CSVs.
#
# Usage: bash run_expC.sh <regime> <arm> [round_num]
#   regime: deep_scarcity | transition | ample
#   arm:    longliu | static | fair
#   round_num: 1-3 (default 1)
#
# Prereqs:
#   1. epoch_emulator compiled on BOTH nodes (emulator/epoch_emulator)
#   2. scenarios.json configured
#   3. Solo calibration done (T_target files in /tmp/expC_ttarget_<job_id>.json)
# ============================================================================
set -uo pipefail  # no -e: handle errors explicitly

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_C_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATOR="$EXP_C_DIR/emulator/epoch_emulator"
SCENARIOS="$EXP_C_DIR/scenarios/scenarios.json"
DAEMON="$EXP_C_DIR/daemon/alloc_daemon.py"

REGIME=${1:?Usage: bash run_expC.sh <regime> <arm> [round_num]}
ARM=${2:?Usage: bash run_expC.sh <regime> <arm> [round_num]}
ROUND=${3:-1}

# Nodes
NODE_226="10.157.197.107"
RDMA_226=""  # filled from scenarios or hardcoded
RDMA_10="192.10.10.110"
DEV="mlx5_0"
BASE_PORT=31000

# Experiment params (read from scenarios.json via python)
read -r NUM_EPOCHS ITERS_PER_EPOCH JITTER_PCT RUNTIME_S <<<$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
p = c['experiment_params']
print(p['num_epochs'], p['iters_per_epoch'], p['jitter_pct'], p['runtime_s'])
")

# Jobs for this regime
JOB_IDS=$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
jobs = c['regimes']['$REGIME']['jobs']
print(' '.join(str(j['job_id']) for j in jobs))
")

# Per-job params
declare -A JOB_CI JOB_PAYLOAD JOB_SLEEP JOB_TTARGET
while IFS='|' read -r jid ci payload_kb sleep_us ttarget_ms; do
    JOB_CI[$jid]=$ci
    JOB_PAYLOAD[$jid]=$((payload_kb * 1024))
    JOB_SLEEP[$jid]=$sleep_us
    JOB_TTARGET[$jid]=$ttarget_ms
done < <(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
for j in c['regimes']['$REGIME']['jobs']:
    ttf = f'/tmp/expC_ttarget_{j[\"job_id\"]}.json'
    import os
    if os.path.exists(ttf):
        d = json.load(open(ttf))
        tt = d.get('target_comm_time_ms', j.get('t_target_ms_est', 1.0))
    else:
        tt = j.get('t_target_ms_est', 1.0)
    print(f'{j[\"job_id\"]}|{j[\"c_i\"]}|{j[\"payload_kb\"]}|{j[\"sleep_us\"]}|{tt}')
")

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$EXP_C_DIR/data/${REGIME}_${ARM}_r${ROUND}_${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "================================================================"
echo "Experiment C — Regime: $REGIME | Arm: $ARM | Round: $ROUND"
echo "  Jobs: $JOB_IDS"
echo "  Epochs: $NUM_EPOCHS × $ITERS_PER_EPOCH iters"
echo "  Jitter: ${JITTER_PCT}%"
echo "  Date: $(date -Iseconds)"
echo "  Data: $RUN_DIR"
echo "================================================================"

# ---- Cleanup any leftover emulator processes ----
echo "[cleanup] killing leftover emulator processes..."
for jid in $JOB_IDS; do
    pgrep -f "epoch_emulator.*--job-id $jid" | xargs -r kill 2>/dev/null || true
done
ssh -o ConnectTimeout=10 "$NODE_226" "pkill -f 'epoch_emulator.*--server'" 2>/dev/null || true
rm -f /tmp/expC_stats_*.csv /tmp/expC_dscp_* /tmp/expC_daemon_*.log 2>/dev/null || true
ssh -o ConnectTimeout=10 "$NODE_226" "rm -f /tmp/expC_stats_*.csv" 2>/dev/null || true
sleep 2

# ---- Sync emulator binary to 226 ----
echo "[sync] copying epoch_emulator to 226..."
scp -o ConnectTimeout=10 "$EMULATOR" "$NODE_226:/tmp/epoch_emulator" 2>/dev/null || {
    echo "[ERR] failed to copy emulator to 226"
    exit 1
}

# ---- Warmup ----
echo "[warmup] 60s idle to stabilize NIC..."
sleep 60

# ---- Launch SERVER processes on 226 ----
echo "[launch] Starting servers on 226..."
for jid in $JOB_IDS; do
    PORT=$((BASE_PORT + jid))
    DATA_SIZE=${JOB_PAYLOAD[$jid]}
    SLEEP_US=${JOB_SLEEP[$jid]}
    echo "  Server job $jid on port $PORT (data=${DATA_SIZE}B sleep=${SLEEP_US}us)..."
    ssh -n -o ConnectTimeout=10 "$NODE_226" \
        "/tmp/epoch_emulator --server --port $PORT --job-id $jid \
         --num-epochs $NUM_EPOCHS --iters-per-epoch $ITERS_PER_EPOCH \
         --sleep-us $SLEEP_US --data-size $DATA_SIZE --device $DEV" \
        > "$RUN_DIR/server_${jid}.log" 2>&1 &
    sleep 0.5
done

sleep 3  # wait for servers to bind

# ---- Launch DAEMON on 10.1 ----
echo "[launch] Starting allocation daemon (arm=$ARM)..."
STATIC_FLAG=""
if [[ "$ARM" == "static" || "$ARM" == "fair" ]]; then
    STATIC_FLAG="--static-priority 4"
fi
python3 "$DAEMON" \
    --config "$SCENARIOS" \
    --regime "$REGIME" \
    --arm "$ARM" \
    $STATIC_FLAG \
    --runtime-s $RUNTIME_S \
    > "$RUN_DIR/daemon.log" 2>&1 &
DAEMON_PID=$!
echo "  Daemon PID: $DAEMON_PID"
sleep 2

# ---- Launch CLIENT processes on 10.1 ----
echo "[launch] Starting clients on 10.1..."
for jid in $JOB_IDS; do
    PORT=$((BASE_PORT + jid))
    DATA_SIZE=${JOB_PAYLOAD[$jid]}
    SLEEP_US=${JOB_SLEEP[$jid]}
    CI=${JOB_CI[$jid]}

    echo "  Client job $jid: port=$PORT data=${DATA_SIZE}B sleep=${SLEEP_US}us c_i=$CI"
    "$EMULATOR" --client --host "$NODE_226" --port $PORT --job-id $jid \
        --data-size $DATA_SIZE --sleep-us $SLEEP_US \
        --num-epochs $NUM_EPOCHS --iters-per-epoch $ITERS_PER_EPOCH \
        --jitter-pct $JITTER_PCT --device $DEV \
        > "$RUN_DIR/client_${jid}.log" 2>&1 &
    sleep 0.5
done

echo "[launch] All clients + daemon running. Waiting for completion..."

# ---- Wait for all clients to finish ----
wait
echo "[done] All emulator processes finished."

# Kill daemon if still running
kill "$DAEMON_PID" 2>/dev/null || true

# ---- Cleanup servers on 226 ----
ssh -o ConnectTimeout=10 "$NODE_226" "pkill -f 'epoch_emulator.*--server'" 2>/dev/null || true

# ---- Archive stats files ----
echo "[archive] Collecting stats..."
for jid in $JOB_IDS; do
    cp "/tmp/expC_stats_${jid}.csv" "$RUN_DIR/job${jid}_stats.csv" 2>/dev/null || true
done
cp /tmp/expC_daemon_${REGIME}.log "$RUN_DIR/daemon_epoch.csv" 2>/dev/null || true

# Copy server-side stats from 226
for jid in $JOB_IDS; do
    scp -o ConnectTimeout=10 "$NODE_226:/tmp/expC_stats_${jid}.csv" \
        "$RUN_DIR/job${jid}_server_stats.csv" 2>/dev/null || true
done

# ---- Manifest ----
cat > "$RUN_DIR/manifest.txt" <<EOF
regime=$REGIME
arm=$ARM
round=$ROUND
run_ts=$RUN_TS
jobs=$JOB_IDS
num_epochs=$NUM_EPOCHS
iters_per_epoch=$ITERS_PER_EPOCH
jitter_pct=$JITTER_PCT
link_bw_gbps=50
topology=dumbbell
EOF

# Save scenarios snapshot
cp "$SCENARIOS" "$RUN_DIR/scenarios_snapshot.json"
md5sum "$SCENARIOS" > "$RUN_DIR/scenarios.md5"

echo ""
echo "================================================================"
echo "Round complete: $RUN_DIR"
echo "================================================================"
ls -la "$RUN_DIR/"
echo "================================================================"
