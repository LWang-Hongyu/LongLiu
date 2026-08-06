#!/bin/bash
# ============================================================================
# Experiment C v2.1 — Solo Calibration
# ============================================================================
# Runs each job solo to measure T_target (solo comm time per iter) for each
# payload size. Computes sleep_us from measured Tcomm and target φ.
#
# v2.1: Uses payload_kb directly from scenario (not derived from d_per_epoch_mb).
#       Sleep_us is computed from measured Tcomm and phi_target AFTER calibration.
#       The run script reads the computed sleep_us from the ttarget JSON files.
#
# Usage: bash calib_solo_v2.sh <scenario> <regime>
#   scenario: S1 | S2
#   regime:   S1_moderate | S1_ample | S1_deep | S1_very_deep | S2_starvation
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_C_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMULATOR="$EXP_C_DIR/emulator/epoch_emulator"
SCENARIOS="$EXP_C_DIR/scenarios/scenarios_v2.json"

SCENARIO=${1:?Usage: bash calib_solo_v2.sh <scenario> <regime>}
REGIME=${2:?Usage: bash calib_solo_v2.sh <scenario> <regime>}
NODE_226="10.157.197.107"
DEV="mlx5_0"
BASE_PORT=31000
CALIB_EPOCHS=10
CALIB_ITERS=20

# Sync emulator to 226
echo "[sync] Copying epoch_emulator to 226..."
scp -o ConnectTimeout=10 "$EMULATOR" "$NODE_226:/tmp/epoch_emulator" 2>/dev/null || {
    echo "[ERR] Failed to copy emulator to 226"
    exit 1
}

# Get jobs for this regime (python resolves d_scale and payload)
echo "=== Solo calibration for scenario=$SCENARIO regime=$REGIME ==="

JOBS_JSON=$(python3 -c "
import json
c = json.load(open('$SCENARIOS'))
sc = c['scenarios']['$SCENARIO']
regime = sc['regimes']['$REGIME']
d_scale = regime.get('d_scale', 1.0)
for bj in sc['base_jobs']:
    # Scale payload by d_scale
    payload_kb = int(bj['payload_kb'] * d_scale)
    # Use a temporary sleep_us for calibration (will be adjusted after)
    temp_sleep_us = 5000  # 5ms placeholder
    print(f\"{bj['job_id']}|{payload_kb}|{temp_sleep_us}|{bj['c_policy']}|{bj['c_eval']}|{bj['tier']}|{bj['phi_target']}\")
")

echo "$JOBS_JSON" | while IFS='|' read -r jid payload_kb sleep_us c_policy c_eval tier phi_target; do
    [ -z "$jid" ] && continue
    PORT=$((BASE_PORT + jid))
    DATA_SIZE=$((payload_kb * 1024))

    echo ""
    echo "[calib] Job $jid ($tier): payload=${payload_kb}KB, sleep=${sleep_us}us, c_policy=$c_policy, c_eval=$c_eval, φ_target=$phi_target"

    # Start server on 226
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
    ssh -n -o ConnectTimeout=10 "$NODE_226" "pgrep -f 'epoch_emulator.*--job-id $jid' | xargs -r kill" 2>/dev/null || true

    if [ $RC -ne 0 ]; then
        echo "[ERR] Job $jid calibration failed (RC=$RC)"
        continue
    fi

    # Parse stats and compute T_target + adjusted sleep_us
    STATS_FILE="/tmp/expC_stats_${jid}.csv"
    if [ ! -f "$STATS_FILE" ]; then
        echo "[ERR] Stats file not found: $STATS_FILE"
        continue
    fi

    python3 -c "
import csv, json
stats_file = '$STATS_FILE'
ttarget_file = '/tmp/expC_ttarget_${jid}.json'
c_policy = $c_policy
c_eval = $c_eval
payload_kb = $payload_kb
phi_target = $phi_target
tier = '$tier'

# Read per-iter stats (skip first 3 epochs warmup)
iters = []
sleep_vals = []
with open(stats_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = int(row['epoch'])
        if ep < 3:
            continue
        iters.append(float(row['comm_us']))
        if 'sleep_us' in row:
            sleep_vals.append(int(row['sleep_us']))

if not iters:
    print('[ERR] no calibration data for job $jid')
    exit(1)

# T_target = avg comm time per iter (in ms)
avg_comm_us = sum(iters) / len(iters)
t_target_ms = avg_comm_us / 1000.0

# Solo bandwidth
solo_bw_gbps = (payload_kb * 1024 * 8) / (t_target_ms / 1000) / 1e9

# Compute adjusted sleep_us from measured Tcomm and target φ
# φ = Tcomm/(Tcomp+Tcomm) → Tcomp = Tcomm * (1-φ)/φ
tcomm_solo_ms = t_target_ms
if phi_target > 0 and phi_target < 1:
    tcomp_required_ms = tcomm_solo_ms * (1 - phi_target) / phi_target
    sleep_us_adjusted = int(tcomp_required_ms * 1000)
else:
    sleep_us_adjusted = 5000  # fallback

# Verify with adjusted sleep
phi_with_adjusted = tcomm_solo_ms / (tcomp_required_ms + tcomm_solo_ms)

# Epoch duration estimate with adjusted params
iters_per_epoch = $CALIB_ITERS
epoch_duration_ms = (tcomp_required_ms + tcomm_solo_ms) * iters_per_epoch

# b̄ estimate
b_bar_gbps = (payload_kb * 1024 * 8) / ((tcomp_required_ms + tcomm_solo_ms) / 1000) / 1e9

d = {
    'job_id': $jid,
    'tier': tier,
    'c_policy': c_policy,
    'c_eval': c_eval,
    'payload_kb': payload_kb,
    'payload_bytes': payload_kb * 1024,
    'target_comm_time_ms': t_target_ms,
    'unit': 'per_iter_ms',
    'source': 'solo_calibration_v2.1',
    'num_epochs': $CALIB_EPOCHS,
    'iters_per_epoch': $CALIB_ITERS,
    'solo_bw_gbps': round(solo_bw_gbps, 2),
    'tcomm_solo_per_iter_ms': tcomm_solo_ms,
    'phi_target': phi_target,
    'sleep_us_adjusted': sleep_us_adjusted,
    'tcomp_per_iter_ms': tcomp_required_ms,
    'phi_with_adjusted': round(phi_with_adjusted, 4),
    'epoch_duration_ms': round(epoch_duration_ms, 1),
    'b_bar_gbps': round(b_bar_gbps, 2),
    'timestamp': '$(date -Iseconds)'
}
json.dump(d, open(ttarget_file, 'w'), indent=2)
print(f'[ok] Job $jid ({tier}): Tcomm_solo={tcomm_solo_ms:.3f}ms, '
      f'solo_bw={solo_bw_gbps:.1f}Gbps, '
      f'sleep_us_adjusted={sleep_us_adjusted}, '
      f'φ_adj={phi_with_adjusted:.4f} (target={phi_target}), '
      f'b̄={b_bar_gbps:.1f}G, '
      f'epoch≈{epoch_duration_ms:.0f}ms → {ttarget_file}')
"

    rm -f "$STATS_FILE"
    sleep 3  # NIC cooldown
done

echo ""
echo "=== Calibration complete ==="
echo "T_target files:"
ls -la /tmp/expC_ttarget_*.json 2>/dev/null

# Summary with adjusted sleep_us
echo ""
echo "=== Adjusted Parameters Summary ==="
python3 -c "
import json, glob
files = sorted(glob.glob('/tmp/expC_ttarget_*.json'))
total_b_bar = 0
premium_batt = []
for f in files:
    d = json.load(open(f))
    if 'sleep_us_adjusted' not in d:
        continue
    b_bar = d.get('b_bar_gbps', 0)
    total_b_bar += b_bar
    tier = d.get('tier', '?')
    c_policy = d.get('c_policy', 0)
    b_att = c_policy * b_bar
    if tier == 'premium':
        premium_batt.append(b_att)
    n = len(files)
    bn = 50.0 / n if n > 0 else 0
    print(f'  Job {d[\"job_id\"]} ({tier}): payload={d[\"payload_kb\"]}KB, '
          f'sleep_us={d[\"sleep_us_adjusted\"]}, '
          f'Tcomm={d[\"tcomm_solo_per_iter_ms\"]:.3f}ms, '
          f'φ={d[\"phi_with_adjusted\"]:.4f}, '
          f'b̄={b_bar:.1f}G, b^att={b_att:.1f}G, '
          f'epoch≈{d[\"epoch_duration_ms\"]:.0f}ms')

print(f'\\n  Σb̄ = {total_b_bar:.1f}G ({total_b_bar/50:.2f}B of 50G link)')
if premium_batt:
    print(f'  Premium b^att = {premium_batt}, B/N = {50/n:.1f}G, '
          f'b^att ≥ 1.5×B/N? {all(b >= 1.5*50/n for b in premium_batt)}')
"
