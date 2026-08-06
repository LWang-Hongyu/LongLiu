#!/bin/bash
# ============================================================================
# Experiment C — Solo Calibration
# ============================================================================
# Runs each job solo to measure T_target (solo comm time) for each payload size.
# Writes /tmp/expC_ttarget_<job_id>.json for the daemon to use.
#
# Usage: bash calib_solo_expC.sh <regime>
#   regime: deep_scarcity | transition | ample
# ============================================================================
set -uo pipefail  # no -e: handle errors explicitly in loop

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_C_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATOR="$EXP_C_DIR/emulator/epoch_emulator"
SCENARIOS="$EXP_C_DIR/scenarios/scenarios.json"

REGIME=${1:?Usage: bash calib_solo_expC.sh <regime>}
NODE_226="10.157.197.107"
DEV="mlx5_0"
BASE_PORT=31000

# Sync emulator to 226
scp -o ConnectTimeout=10 "$EMULATOR" "$NODE_226:/tmp/epoch_emulator" 2>/dev/null || true

# Get unique payload sizes and their jobs
echo "=== Solo calibration for regime: $REGIME ==="

# Read jobs from scenarios.json
JOBS_JSON=$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
for j in c['regimes']['$REGIME']['jobs']:
    print(f'{j[\"job_id\"]} {j[\"payload_kb\"]} {j[\"sleep_us\"]} {j[\"c_i\"]}')
")

echo "$JOBS_JSON" | while IFS=' ' read -r jid payload_kb sleep_us ci; do
    [ -z "$jid" ] && continue
    PORT=$((BASE_PORT + jid))
    DATA_SIZE=$((payload_kb * 1024))
    CALIB_EPOCHS=10
    CALIB_ITERS=20

    echo ""
    echo "[calib] Job $jid: payload=${payload_kb}KB, sleep=${sleep_us}us, c_i=$ci"

    # Start server on 226 (pass sleep-us + data-size for duration estimation)
    # Use -n to redirect ssh stdin from /dev/null (prevents consuming pipe stdin)
    ssh -n -o ConnectTimeout=10 "$NODE_226" \
        "/tmp/epoch_emulator --server --port $PORT --job-id $jid \
         --num-epochs $CALIB_EPOCHS --iters-per-epoch $CALIB_ITERS \
         --sleep-us $sleep_us --data-size $DATA_SIZE --device $DEV" \
        > "/tmp/expC_calib_server_${jid}.log" 2>&1 &
    sleep 2

    # Start client on 10.1
    "$EMULATOR" --client --host "$NODE_226" --port $PORT --job-id $jid \
        --data-size $DATA_SIZE --sleep-us $sleep_us \
        --num-epochs $CALIB_EPOCHS --iters-per-epoch $CALIB_ITERS \
        --device $DEV \
        > "/tmp/expC_calib_client_${jid}.log" 2>&1
    RC=$?

    # Kill server
    ssh -n -o ConnectTimeout=10 "$NODE_226" "pkill -f 'epoch_emulator.*--job-id $jid'" 2>/dev/null || true

    if [ $RC -ne 0 ]; then
        echo "[ERR] Job $jid calibration failed (RC=$RC)"
        continue
    fi

    # Parse stats: compute avg comm time from epochs 3-9 (skip first 3 warmup)
    STATS_FILE="/tmp/expC_stats_${jid}.csv"
    if [ ! -f "$STATS_FILE" ]; then
        echo "[ERR] Stats file not found: $STATS_FILE"
        continue
    fi

    python3 -c "
import csv, json
stats_file = '$STATS_FILE'
ttarget_file = '/tmp/expC_ttarget_${jid}.json'
ci = $ci
payload_kb = $payload_kb

# Read per-iter stats
iters = []
with open(stats_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = int(row['epoch'])
        if ep < 3:  # skip warmup
            continue
        iters.append(float(row['comm_us']))

if not iters:
    print('[ERR] no calibration data for job $jid')
    exit(1)

# T_target = avg comm time per iter (in ms)
avg_comm_us = sum(iters) / len(iters)
t_target_ms = avg_comm_us / 1000.0

d = {
    'job_id': $jid,
    'c_i': ci,
    'payload_kb': payload_kb,
    'payload_bytes': payload_kb * 1024,
    'target_comm_time_ms': t_target_ms,
    'unit': 'per_iter_ms',
    'source': 'solo_calibration',
    'num_epochs': 10,
    'iters_per_epoch': 20,
    'calib_epochs_used': len(iters) // 20,
    'solo_bw_gbps': (payload_kb * 1024 * 8) / (t_target_ms / 1000) / 1e9,
    'timestamp': '$(date -Iseconds)'
}
json.dump(d, open(ttarget_file, 'w'), indent=2)
print(f'[ok] Job $jid: T_target={t_target_ms:.3f}ms, solo_bw={d[\"solo_bw_gbps\"]:.1f}Gbps → {ttarget_file}')
"

    rm -f "$STATS_FILE"
    sleep 2  # NIC cooldown between calibrations
done

echo ""
echo "=== Calibration complete ==="
echo "T_target files:"
ls -la /tmp/expC_ttarget_*.json 2>/dev/null
