#!/bin/bash
# Quick single-payload solo bandwidth test (full 4-process setup)
PAYLOAD_MB=${1:-2048}
EXP_DIR=/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo
TTARGET_DIR=/tmp/ttarget_q
mkdir -p $TTARGET_DIR

# Kill stale
pgrep -f p4_job_reverse|xargs -r kill 2>/dev/null
ssh 192.10.10.226 "pgrep -f p4_job_reverse|xargs -r kill" 2>/dev/null
sleep 3

TTFILE_A="${TTARGET_DIR}/jobA_${PAYLOAD_MB}MB.json"
TTFILE_B="${TTARGET_DIR}/jobB_${PAYLOAD_MB}MB.json"

# Job B rank 1 on 226
ssh 192.10.10.226 "cd ${EXP_DIR} && \
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp59s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src:\$PYTHONPATH \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29501 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
timeout 120 python3 -u ${EXP_DIR}/p4_job_reverse.py \
--job B --mode longliu --phase calibrate \
--ttarget-file $TTFILE_B \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000" \
>/tmp/bwtest_jobB_226.log 2>&1 &

# Job A rank 1 on 226
ssh 192.10.10.226 "cd ${EXP_DIR} && \
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp59s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src:\$PYTHONPATH \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
timeout 120 python3 -u ${EXP_DIR}/p4_job_reverse.py \
--job A --mode longliu --phase calibrate \
--ttarget-file $TTFILE_A \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000" \
>/tmp/bwtest_jobA_226.log 2>&1 &

sleep 3

# Job B rank 0 on 10.1
cd ${EXP_DIR}
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp130s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29501 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
timeout 120 python3 -u p4_job_reverse.py \
--job B --mode longliu --phase calibrate \
--ttarget-file $TTFILE_B \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000 \
>/tmp/bwtest_jobB_10.log 2>&1 &
sleep 1

# Job A rank 0 on 10.1 (blocking)
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp130s0f0np0 \
NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
PYTHONPATH=${EXP_DIR}/../../multi_comm_slo/src \
MASTER_ADDR=192.10.10.110 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
timeout 120 python3 -u p4_job_reverse.py \
--job A --mode longliu --phase calibrate \
--ttarget-file $TTFILE_A \
--payload-mb $PAYLOAD_MB --ci-phase1 1.2 --ci-phase2 3.0 --sleep-us 30000

echo "exit=$?"

# Results
echo ""
echo "=== RESULTS ==="
for J in A B; do
  F="${TTARGET_DIR}/job${J}_${PAYLOAD_MB}MB.json"
  if [ -f $F ]; then
    python3 -c "
import json
d=json.load(open('$F'))
t=d['target_comm_time_ms']
pay=d['payload_mb']
bw=pay*8/1000/(t/1000)
print(f'Job {J}: {pay}MB -> T_target={t:.1f}ms, BW={bw:.1f}Gbps ({bw/50*100:.0f}% of 50G)')"
  else
    echo "Job $J: no result file"
  fi
done
