#!/bin/bash
# ============================================================================
# Experiment C v2 — Scale Ladder Runner (one regime × one arm)
# ============================================================================
# Launches N emulator job-pairs across 10.1 (client) and 226 (server),
# plus the allocation daemon on 10.1. Waits for completion, archives CSVs.
#
# v2 changes:
#   - Reads from scenarios_v2.json with scenario/regime/d_scale
#   - Supports c_policy/c_eval via daemon v2
#   - Static arm: premium→P6, standard→P2
#   - Logs sleep_us in stats for iteration-level slowdown
#   - More epochs (200) and rounds (5) for statistical robustness
#
# Usage: bash run_expC_v2.sh <scenario> <regime> <arm> [round_num]
#   scenario: S1 | S2
#   regime:   S1_ample | S1_moderate | S1_deep | S1_very_deep | S2_starvation
#   arm:      longliu | static | fair
#   round_num: 1-5 (default 1)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_C_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATOR="$EXP_C_DIR/emulator/epoch_emulator"
SCENARIOS="$EXP_C_DIR/scenarios/scenarios_v2.json"
DAEMON="$EXP_C_DIR/daemon/alloc_daemon_v2.py"
DATA_DIR="$EXP_C_DIR/data_v2"

SCENARIO=${1:?Usage: bash run_expC_v2.sh <scenario> <regime> <arm> [round]}
REGIME=${2:?Usage: bash run_expC_v2.sh <scenario> <regime> <arm> [round]}
ARM=${3:?Usage: bash run_expC_v2.sh <scenario> <regime> <arm> [round]}
ROUND=${4:-1}

# Nodes
NODE_226="10.157.197.107"
DEV="mlx5_0"
BASE_PORT=31000

# Experiment params (read from scenarios_v2.json)
read -r NUM_EPOCHS ITERS_PER_EPOCH JITTER_PCT RUNTIME_S POLL_INTERVAL <<<$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
p = c['experiment_params']
print(p['num_epochs'], p['iters_per_epoch'], p['jitter_pct'],
      p['runtime_s'], p.get('daemon_poll_interval_s', 0.1))
")

# Resolve job IDs from scenario config
# All actual parameters (payload_bytes, sleep_us, c_policy, c_eval, tier, T_target)
# are read from calibration files, which are the source of truth.
JOB_IDS=$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
sc = c['scenarios']['$SCENARIO']
for bj in sc['base_jobs']:
    print(bj['job_id'], end=' ')
print()
")
JOB_IDS=$(echo $JOB_IDS | xargs)  # trim

# Read all per-job parameters from calibration files (source of truth)
declare -A JOB_TTARGET JOB_SLEEP JOB_PAYLOAD JOB_TIER JOB_CP JOB_CE
for jid in $JOB_IDS; do
    ttf="/tmp/expC_ttarget_${jid}.json"
    if [ -f "$ttf" ]; then
        eval $(python3 -c "
import json
d = json.load(open('$ttf'))
print(f\"JOB_TTARGET[$jid]={d['target_comm_time_ms']}\")
print(f\"JOB_SLEEP[$jid]={d.get('sleep_us_adjusted', 5000)}\")
print(f\"JOB_PAYLOAD[$jid]={d.get('payload_bytes', d.get('payload_kb',1024)*1024)}\")
print(f\"JOB_TIER[$jid]={d.get('tier','?')}\")
print(f\"JOB_CP[$jid]={d.get('c_policy',1.35)}\")
print(f\"JOB_CE[$jid]={d.get('c_eval',1.5)}\")
")
    else
        JOB_TTARGET[$jid]="N/A"
        echo "[WARN] No calibration file for job $jid! Run calib_solo_v2.sh first."
    fi
done

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DATA_DIR/${REGIME}_${ARM}_r${ROUND}_${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "================================================================"
echo "Experiment C v2 — Scenario: $SCENARIO | Regime: $REGIME | Arm: $ARM | Round: $ROUND"
echo "  Jobs: $JOB_IDS"
echo "  Epochs: $NUM_EPOCHS × $ITERS_PER_EPOCH iters"
echo "  Jitter: ${JITTER_PCT}%"
echo "  Daemon poll: ${POLL_INTERVAL}s"
echo "  Date: $(date -Iseconds)"
echo "  Data: $RUN_DIR"
echo "================================================================"
echo "Job details:"
for jid in $JOB_IDS; do
    echo "  J$jid (${JOB_TIER[$jid]}): c_policy=${JOB_CP[$jid]}, c_eval=${JOB_CE[$jid]}, payload=${JOB_PAYLOAD[$jid]}B, sleep=${JOB_SLEEP[$jid]}us, T_target=${JOB_TTARGET[$jid]}ms"
done

# ---- Verify T_target files exist ----
MISSING=0
for jid in $JOB_IDS; do
    if [ ! -f "/tmp/expC_ttarget_${jid}.json" ]; then
        echo "[ERR] Missing T_target for job $jid! Run calib_solo_v2.sh first."
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo "[ERR] Aborting: missing calibration files."
    exit 1
fi

# ---- Cleanup any leftover emulator processes ----
echo "[cleanup] Killing leftover emulator processes..."
for jid in $JOB_IDS; do
    pgrep -f "epoch_emulator.*--job-id $jid " | xargs -r kill 2>/dev/null || true
done
ssh -n -o ConnectTimeout=10 "$NODE_226" "pgrep -f 'epoch_emulator.*--server' | xargs -r kill" 2>/dev/null || true
rm -f /tmp/expC_stats_*.csv /tmp/expC_dscp_* /tmp/expC_daemon_*.log 2>/dev/null || true
ssh -n -o ConnectTimeout=10 "$NODE_226" "rm -f /tmp/expC_stats_*.csv" 2>/dev/null || true
sleep 2

# ---- Sync emulator binary to 226 ----
echo "[sync] Copying epoch_emulator to 226..."
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
echo "[launch] Starting allocation daemon v2 (arm=$ARM)..."
python3 "$DAEMON" \
    --config "$SCENARIOS" \
    --scenario "$SCENARIO" \
    --regime "$REGIME" \
    --arm "$ARM" \
    --runtime-s $RUNTIME_S \
    --poll-interval-s $POLL_INTERVAL \
    > "$RUN_DIR/daemon.log" 2>&1 &
DAEMON_PID=$!
echo "  Daemon PID: $DAEMON_PID"
sleep 2

# ---- Launch CLIENT processes on 10.1 ----
echo "[launch] Starting clients on 10.1..."
CLIENT_PIDS=()
for jid in $JOB_IDS; do
    PORT=$((BASE_PORT + jid))
    DATA_SIZE=${JOB_PAYLOAD[$jid]}
    SLEEP_US=${JOB_SLEEP[$jid]}
    CP=${JOB_CP[$jid]}

    echo "  Client job $jid: port=$PORT data=${DATA_SIZE}B sleep=${SLEEP_US}us c_policy=$CP"
    "$EMULATOR" --client --host "$NODE_226" --port $PORT --job-id $jid \
        --data-size $DATA_SIZE --sleep-us $SLEEP_US \
        --num-epochs $NUM_EPOCHS --iters-per-epoch $ITERS_PER_EPOCH \
        --jitter-pct $JITTER_PCT --device $DEV \
        > "$RUN_DIR/client_${jid}.log" 2>&1 &
    CLIENT_PIDS+=($!)
    sleep 0.5
done

echo "[launch] All clients + daemon running. Waiting for completion..."

# ---- Wait for all CLIENT processes to finish ----
# NOTE: We only wait for client PIDs, NOT the daemon (which has a 900s timeout).
# The daemon is killed separately after clients finish.
for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done
echo "[done] All emulator processes finished."

# Kill daemon if still running
kill "$DAEMON_PID" 2>/dev/null || true

# ---- Cleanup servers on 226 ----
ssh -n -o ConnectTimeout=10 "$NODE_226" "pgrep -f 'epoch_emulator.*--server' | xargs -r kill" 2>/dev/null || true

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
scenario=$SCENARIO
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
version=v2
c_policy_eval_split=yes
slowdown_formula=iter_level
EOF

# Save scenarios snapshot
cp "$SCENARIOS" "$RUN_DIR/scenarios_v2_snapshot.json"
md5sum "$SCENARIOS" > "$RUN_DIR/scenarios_v2.md5"

# Save T_target calibration files
for jid in $JOB_IDS; do
    cp "/tmp/expC_ttarget_${jid}.json" "$RUN_DIR/ttarget_${jid}.json" 2>/dev/null || true
done

echo ""
echo "================================================================"
echo "Round complete: $RUN_DIR"
echo "================================================================"
ls -la "$RUN_DIR/"
echo "================================================================"
