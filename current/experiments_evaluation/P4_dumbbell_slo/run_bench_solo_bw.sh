#!/bin/bash
# Bandwidth sweep: test solo BW at different payload sizes
set -e

cd "$(dirname "$0")"
SCRIPT="bench_solo_bw.py"
REMOTE_SCRIPT="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo/bench_solo_bw.py"

# Kill stale processes
pgrep -f bench_solo_bw | xargs -r kill 2>/dev/null || true
ssh 192.10.10.226 "pgrep -f bench_solo_bw | xargs -r kill 2>/dev/null" || true
sleep 2

for MB in 1024 2048 3072 4096; do
  echo ""
  echo "=========================================="
  echo "  Testing ${MB}MB payload..."
  echo "=========================================="
  
  # Launch 226 in background
  ssh 192.10.10.226 "NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_DEBUG=WARN MASTER_ADDR=192.10.10.110 MASTER_PORT=29999 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 timeout 120 python3 ${REMOTE_SCRIPT} ${MB}" > /tmp/bw_${MB}_226.log 2>&1 &
  sleep 3
  
  # Launch 10.1 (blocking)
  NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_DEBUG=WARN MASTER_ADDR=192.10.10.110 MASTER_PORT=29999 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 timeout 120 python3 /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo/${SCRIPT} ${MB} > /tmp/bw_${MB}_10.log 2>&1
  
  RC=$?
  echo "[${MB}MB] exit=$RC"
  grep -E "RESULT|ERROR|error|Traceback" /tmp/bw_${MB}_10.log || echo "(no RESULT line)"
  
  # Short delay
  sleep 5
done

echo ""
echo "=== BANDWIDTH SWEEP SUMMARY ==="
echo "Payload | Comm(ms) | BW(Gbps) | LineUtil"
echo "--------|----------|----------|---------"
for MB in 1024 2048 3072 4096; do
  if [ -f /tmp/bw_${MB}_10.log ]; then
    grep "^=== RESULT" /tmp/bw_${MB}_10.log
    grep "Line util" /tmp/bw_${MB}_10.log
  fi
done
echo ""
echo "=== DETAILED LOGS ==="
for MB in 1024 2048 3072 4096; do
  echo "--- ${MB}MB 10.1 ---"
  cat /tmp/bw_${MB}_10.log 2>/dev/null
  echo "--- ${MB}MB 226 ---"
  cat /tmp/bw_${MB}_226.log 2>/dev/null | grep -v "^$"
done
