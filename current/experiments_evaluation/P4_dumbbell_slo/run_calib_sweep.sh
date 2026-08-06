#!/bin/bash
# Solo calibration sweep: measure solo BW at different payload sizes
# Uses the experiment's multi_comm_wrapper -> LongLiu NCCL 2.30.7
set -euo pipefail

EXP_DIR=/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo
TTARGET_DIR=/tmp/ttarget_sweep
mkdir -p $TTARGET_DIR

for MB in 512 1024 2048 3072 4096; do
  echo ""
  echo "=========================================="
  echo "  Calibrating ${MB}MB..."
  echo "=========================================="
  
  # Clean up
  pgrep -f p4_job_reverse | xargs -r kill 2>/dev/null || true
  ssh 192.10.10.226 "pgrep -f p4_job_reverse | xargs -r kill" 2>/dev/null || true
  sleep 3
  
  TTFILE_A="${TTARGET_DIR}/jobA_${MB}MB_calib.json"
  TTFILE_B="${TTARGET_DIR}/jobB_${MB}MB_calib.json"
  
  # Launch Job B on 226 first (will wait)
  ssh 192.10.10.226 "cd ${EXP_DIR} && \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src:\$PYTHONPATH \
    MASTER_ADDR=192.10.10.110 MASTER_PORT=29999 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
    python3 -u ${EXP_DIR}/p4_job_reverse.py --job B --mode longliu --phase calibrate \
      --ttarget-file $TTFILE_B \
      --payload-mb $MB --ci-phase1 1.2 --ci-phase2 3.0 \
      --sleep-us 30000" > /tmp/sweep_${MB}MB_jobB.log 2>&1 &
  sleep 3
  
  # Launch Job A on 10.1 (blocking)
  cd ${EXP_DIR} && \
  NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src \
    MASTER_ADDR=192.10.10.110 MASTER_PORT=29999 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
    python3 -u p4_job_reverse.py --job A --mode longliu --phase calibrate \
      --ttarget-file $TTFILE_A \
      --payload-mb $MB --ci-phase1 1.2 --ci-phase2 3.0 \
      --sleep-us 30000 \
      2>&1
  
  echo "[${MB}MB] exit=$?"
  
  # Extract result
  if [ -f $TTFILE_A ]; then
    echo "  -> Job A: $(python3 -c "import json; d=json.load(open('$TTFILE_A')); print(f'T_target={d[\"target_comm_time_ms\"]}ms, bw={1024*8/1000/(d[\"target_comm_time_ms\"]/1000):.1f}Gbps')")"
  fi
  if [ -f $TTFILE_B ]; then
    echo "  -> Job B: $(python3 -c "import json; d=json.load(open('$TTFILE_B')); print(f'T_target={d[\"target_comm_time_ms\"]}ms, bw={1024*8/1000/(d[\"target_comm_time_ms\"]/1000):.1f}Gbps')")"
  fi
  
  sleep 5
done

echo ""
echo "=== CALIBRATION SWEEP SUMMARY ==="
echo "Payload | Job A T_target(ms) | Job A BW(Gbps) | Job B T_target(ms) | Job B BW(Gbps)"
echo "--------|--------------------|----------------|--------------------|---------------"
for MB in 512 1024 2048 3072 4096; do
  TTFILE_A="${TTARGET_DIR}/jobA_${MB}MB_calib.json"
  TTFILE_B="${TTARGET_DIR}/jobB_${MB}MB_calib.json"
  if [ -f $TTFILE_A ] && [ -f $TTFILE_B ]; then
    python3 -c "
import json
a = json.load(open('$TTFILE_A'))
b = json.load(open('$TTFILE_B'))
t_a = a['target_comm_time_ms']
t_b = b['target_comm_time_ms']
pay = a['payload_mb']
bw_a = pay*8/1000/(t_a/1000)
bw_b = pay*8/1000/(t_b/1000)
print(f'{pay:>5}MB | {t_a:>18.1f} | {bw_a:>14.1f} | {t_b:>18.1f} | {bw_b:>14.1f}')"
  fi
done
