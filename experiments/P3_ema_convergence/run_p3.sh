#!/bin/bash
#
# P3: EMA Bandwidth Convergence Experiment
# Launches two independent NCCL jobs across 10.1 (node_rank=0) and 226 (node_rank=1)
#
# BOTH jobs are launched in parallel. Job 1 runs epochs 1-10 immediately.
# Job 2 waits for Job 1's sync signal (epoch 5 done), then runs epochs 6-10.
# This ensures epochs 6-10 are truly contested.
#
# Network: RDMA fabric (192.10.10.x) via mlx5_0 / RoCE
#   10.1: enp130s0f0np0 @ 192.10.10.110
#   226:  enp59s0f0np0  @ 192.10.10.226
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_ADDR="192.10.10.110"
NODE_226="192.10.10.226"
NODE_226_IF="enp59s0f0np0"
NODE_101_IF="enp130s0f0np0"

# LD_PRELOAD paths for DSCP-modified NCCL library
LD_PRELOAD_101="/home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2"
LD_PRELOAD_226="/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"

PAYLOAD_MB="${P3_PAYLOAD_MB:-256}"
ITERS="${P3_ITERS:-20}"
export P3_PAYLOAD_MB="$PAYLOAD_MB"
export P3_ITERS="$ITERS"

# Clean stale files
rm -f "$SCRIPT_DIR/.sync_ready"

echo "============================================"
echo "P3: EMA Bandwidth Convergence Experiment"
echo "Network:  RDMA fabric ($MASTER_ADDR ↔ $NODE_226)"
echo "Payload:  ${PAYLOAD_MB}MB x ${ITERS} iters = $((PAYLOAD_MB * ITERS))MB/epoch"
echo "============================================"

# ─────────────────────────────────────────
# Launch ALL 4 processes in parallel
# ─────────────────────────────────────────

echo "[Launcher] Starting Job 1 (node_rank=0) locally..."
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME=$NODE_101_IF \
LD_PRELOAD=$LD_PRELOAD_101 \
P3_PAYLOAD_MB=$PAYLOAD_MB P3_ITERS=$ITERS \
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
      --master_addr="$MASTER_ADDR" --master_port=29510 \
      "$SCRIPT_DIR/p3_job1.py" \
    > /tmp/p3_job1_node101.log 2>&1 &
PID_J1_101=$!

echo "[Launcher] Starting Job 1 (node_rank=1) on 226..."
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_226" \
    "cd '$SCRIPT_DIR' && \
     export PATH=\"\$HOME/.local/bin:\$PATH\" && \
     NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
     NCCL_SOCKET_IFNAME=$NODE_226_IF \
     LD_PRELOAD=$LD_PRELOAD_226 \
     P3_PAYLOAD_MB=$PAYLOAD_MB P3_ITERS=$ITERS \
     torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
       --master_addr=$MASTER_ADDR --master_port=29510 \
       p3_job1.py" \
    > /tmp/p3_job1_node226.log 2>&1 &
PID_J1_226=$!

# Give Job 1 a moment to init before launching Job 2
sleep 3

echo "[Launcher] Starting Job 2 (node_rank=0) locally..."
NCCL_P2P_DISABLE=1 \
NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME=$NODE_101_IF \
LD_PRELOAD=$LD_PRELOAD_101 \
P3_PAYLOAD_MB=$PAYLOAD_MB P3_ITERS=$ITERS \
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
      --master_addr="$MASTER_ADDR" --master_port=29511 \
      "$SCRIPT_DIR/p3_job2.py" \
    > /tmp/p3_job2_node101.log 2>&1 &
PID_J2_101=$!

echo "[Launcher] Starting Job 2 (node_rank=1) on 226..."
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_226" \
    "cd '$SCRIPT_DIR' && \
     export PATH=\"\$HOME/.local/bin:\$PATH\" && \
     CUDA_VISIBLE_DEVICES=1 \
     NCCL_P2P_DISABLE=1 \
     NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
     NCCL_SOCKET_IFNAME=$NODE_226_IF \
     LD_PRELOAD=$LD_PRELOAD_226 \
     P3_PAYLOAD_MB=$PAYLOAD_MB P3_ITERS=$ITERS \
     torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
       --master_addr=$MASTER_ADDR --master_port=29511 \
       p3_job2.py" \
    > /tmp/p3_job2_node226.log 2>&1 &
PID_J2_226=$!

echo ""
echo "[Launcher] All 4 processes launched. Waiting for completion..."
echo "  Job 1 (10.1):  PID $PID_J1_101"
echo "  Job 1 (226):   PID $PID_J1_226"
echo "  Job 2 (10.1):  PID $PID_J2_101"
echo "  Job 2 (226):   PID $PID_J2_226"
echo ""

# Wait for all processes
wait $PID_J1_101 || echo "[WARN] Job 1 (101) exited with code $?"
wait $PID_J1_226 || echo "[WARN] Job 1 (226) exited with code $?"
wait $PID_J2_101 || echo "[WARN] Job 2 (101) exited with code $?"
wait $PID_J2_226 || echo "[WARN] Job 2 (226) exited with code $?"

echo ""
echo "============================================"
echo "P3 data collection complete!"
echo ""
echo "Output files:"
ls -la "$SCRIPT_DIR"/p3_job*_rank*.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Logs:"
echo "  Job 1 (101): /tmp/p3_job1_node101.log"
echo "  Job 1 (226): /tmp/p3_job1_node226.log"
echo "  Job 2 (101): /tmp/p3_job2_node101.log"
echo "  Job 2 (226): /tmp/p3_job2_node226.log"
echo "============================================"
echo ""
echo "Next: python3 plot_p3.py"
