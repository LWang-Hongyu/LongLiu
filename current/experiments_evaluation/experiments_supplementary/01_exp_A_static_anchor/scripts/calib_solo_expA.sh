#!/bin/bash
# ============================================================================
# Experiment A: Solo BW Calibration for 768MB payload
# ============================================================================
# Produces /tmp/ttarget_expA_jobA_768.json and /tmp/ttarget_expA_jobB_768.json
# (1024MB reuses V5 calibration; 512MB uses theoretical anchor for hold-out S5)
# ============================================================================
set -e
source "$(dirname "$0")/expA_config.sh"
ensure_dirs

PAYLOAD=${1:-768}
JOB=${2:-A}
TTARGET_FILE="/tmp/ttarget_expA_job${JOB}_${PAYLOAD}.json"

echo "================================================================"
echo "[calib] payload=${PAYLOAD}MB job=${JOB}"
echo "[calib] output: $TTARGET_FILE"
echo "================================================================"

# Cleanup any orphan processes
cleanup_jobs
sleep 2

# Launch rank 1 on 226 (background)
ssh -o ConnectTimeout=30 "$NODE_226_SSH" "
    cd $P4_DIR && \
    $NCCL_ENV_COMMON $NCCL_ENV_226 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL:\$LD_LIBRARY_PATH \
    PYTHONPATH=$PYTHONPATH_SLO:\$PYTHONPATH \
    MASTER_ADDR=$RDMA_10 MASTER_PORT=29998 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
    timeout 180 python3 -u $JOB_SCRIPT \
    --job $JOB --mode longliu --phase calibrate \
    --ttarget-file $TTARGET_FILE \
    --payload-mb $PAYLOAD --ci-phase1 1.2 --ci-phase2 1.2 \
    --sleep-us $SLEEP_US --reverse-epoch $REVERSE_EPOCH_OFF \
    --num-iters 100 --iters-per-epoch 20 --calib-epochs 5
" > "$LOGS_DIR/calib_${JOB}_${PAYLOAD}MB_226.log" 2>&1 &
RANK1_PID=$!
echo "[calib] launched rank 1 on 226 (pid=$RANK1_PID)"
sleep 3

# Launch rank 0 on 10.1 (foreground)
cd "$P4_DIR"
env $NCCL_ENV_COMMON $NCCL_ENV_10 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL \
    PYTHONPATH=$PYTHONPATH_SLO \
    MASTER_ADDR=$RDMA_10 MASTER_PORT=29998 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
    timeout 180 python3 -u "$JOB_SCRIPT" \
    --job "$JOB" --mode longliu --phase calibrate \
    --ttarget-file "$TTARGET_FILE" \
    --payload-mb "$PAYLOAD" --ci-phase1 1.2 --ci-phase2 1.2 \
    --sleep-us "$SLEEP_US" --reverse-epoch "$REVERSE_EPOCH_OFF" \
    --num-iters 100 --iters-per-epoch 20 --calib-epochs 5 \
    > "$LOGS_DIR/calib_${JOB}_${PAYLOAD}MB_10.log" 2>&1
RC=$?

wait $RANK1_PID 2>/dev/null || true
cleanup_jobs

echo "[calib] rank 0 exit=$RC"
if [ $RC -ne 0 ]; then
    echo "[FAIL] calibration failed for ${JOB}@${PAYLOAD}MB"
    echo "--- 10.1 log tail ---"
    tail -30 "$LOGS_DIR/calib_${JOB}_${PAYLOAD}MB_10.log"
    echo "--- 226 log tail ---"
    tail -30 "$LOGS_DIR/calib_${JOB}_${PAYLOAD}MB_226.log"
    exit 1
fi

verify_ttarget "$TTARGET_FILE" || exit 1

# Compute and report solo BW
python3 -c "
import json
d = json.load(open('$TTARGET_FILE'))
t = d['target_comm_time_ms']
pay = d['payload_mb']
bw = (pay * 1024 * 1024 * 8.0 / 1e9) * 0.5 / (t / 1000 / 20)
print(f'[ok] {pay}MB solo -> T_target_epoch={t:.1f}ms, T_target_per_iter={t/20:.2f}ms, BW={bw:.2f}Gbps ({bw/50*100:.1f}% of 50G)')
"
