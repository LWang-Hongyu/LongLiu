#!/bin/bash
# run_solo_bw.sh — simple solo bandwidth test
# Launches a single calibration pair (Job A only, 2 ranks)
# Usage: bash run_solo_bw.sh <payload_mb>
set -e

PAYLOAD_MB=${1:-1024}
EXP_DIR=/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo
TTARGET_FILE=/tmp/ttarget_bw_${PAYLOAD_MB}MB.json

# Kill old & wait
pgrep -f p4_job_reverse | xargs -r kill 2>/dev/null || true
ssh 192.10.10.226 "pgrep -f p4_job_reverse | xargs -r kill" 2>/dev/null || true
sleep 3

echo "Testing ${PAYLOAD_MB}MB solo calibration..."
echo "Launching..."

# Rank 1 on 226 (bg)
ssh 192.10.10.226 "cd ${EXP_DIR} && \
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp59s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src:\$PYTHONPATH \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29998 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
timeout 120 python3 -u ${EXP_DIR}/p4_job_reverse.py \
--job A --mode longliu --phase calibrate \
--ttarget-file $TTARGET_FILE \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000" \
>/tmp/solo_bw_226.log 2>&1 &
sleep 3

# Rank 0 on 10.1 (blocking)
cd ${EXP_DIR}
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp130s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29998 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
timeout 120 python3 -u p4_job_reverse.py \
--job A --mode longliu --phase calibrate \
--ttarget-file $TTARGET_FILE \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000

RC=$?
echo "exit=$RC"
echo "---"

if [ -f $TTARGET_FILE ]; then
  python3 -c "
import json
d=json.load(open('$TTARGET_FILE'))
t=d['target_comm_time_ms']
pay=d['payload_mb']
bw=(pay*1024*1024*8.0/1e9)*0.5/(t/1000/20)
print(f'OK: {pay}MB solo -> T_target_epoch={t:.1f}ms, T_target_per_iter={t/20:.1f}ms, BW={bw:.1f}Gbps ({bw/50*100:.0f}% of 50G)')"
else
  echo "FAIL: no T_target file generated"
  echo "=== 226 log ==="
  cat /tmp/solo_bw_226.log
fi
