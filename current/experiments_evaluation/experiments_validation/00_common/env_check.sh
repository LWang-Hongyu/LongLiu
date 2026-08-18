#!/bin/bash
# ============================================================================
# env_check.sh — 实验前环境核实（双端一致性）
# ============================================================================
# 采集：主机/时间/GPU/NIC 型号固件/链路速率/QoS 映射/NCCL·CUDA 版本/IB 状态
# 输出：<exp_dir>/data/env/env_<timestamp>.txt
#
# Usage:
#   bash env_check.sh [label]
# ============================================================================
set -uo pipefail

LABEL=${1:-env}
NODE_226="192.10.10.226"
IFACE_10="enp130s0f0np0"
IFACE_226="enp59s0f0np0"

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
OUTDIR="$DATA_DIR/env"
mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$OUTDIR/${LABEL}_${TS}.txt"

{
    echo "=== LongLiu 环境核实 $(date -Iseconds) ==="
    echo ""

    echo "### 10.1 (guolab-10) ###"
    echo "-- host --"; hostname; uname -r
    echo "-- GPU --"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
    echo "-- CUDA --"; nvcc --version 2>/dev/null | tail -1
    echo "-- torch/nccl --"
    python3 -c "import torch; print('torch', torch.__version__, 'nccl', torch.version.nccl)" 2>/dev/null || echo "torch unavailable"
    echo "-- NIC (${IFACE_10}) --"
    ethtool -i "$IFACE_10" 2>/dev/null | grep -E 'driver|firmware|version'
    ethtool "$IFACE_10" 2>/dev/null | grep -E 'Speed|Link detected'
    cat /sys/class/infiniband/mlx5_0/device/vendor 2>/dev/null | xargs -I{} echo -n "vendor_id: {}  "
    cat /sys/class/infiniband/mlx5_0/device/device 2>/dev/null | xargs -I{} echo "device_id: {}"
    echo "-- QoS (mlnx_qos) --"
    mlnx_qos -i "$IFACE_10" 2>/dev/null || echo "mlnx_qos not available"
    echo "-- IB port state --"
    cat /sys/class/infiniband/mlx5_0/ports/1/state 2>/dev/null
    cat /sys/class/infiniband/mlx5_0/ports/1/rate 2>/dev/null

    echo ""
    echo "### 226 (guolab-226) ###"
    ssh "$NODE_226" "
        echo '-- host --'; hostname; uname -r
        echo '-- GPU --'
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
        echo '-- torch/nccl --'
        python3 -c \"import torch; print('torch', torch.__version__, 'nccl', torch.version.nccl)\" 2>/dev/null || echo 'torch unavailable'
        echo '-- NIC (${IFACE_226}) --'
        ethtool -i ${IFACE_226} 2>/dev/null | grep -E 'driver|firmware|version'
        ethtool ${IFACE_226} 2>/dev/null | grep -E 'Speed|Link detected'
        cat /sys/class/infiniband/mlx5_0/device/vendor 2>/dev/null | xargs -I{} echo -n 'vendor_id: {}  '
        cat /sys/class/infiniband/mlx5_0/device/device 2>/dev/null | xargs -I{} echo 'device_id: {}'
        echo '-- IB port state --'
        cat /sys/class/infiniband/mlx5_0/ports/1/state 2>/dev/null
        cat /sys/class/infiniband/mlx5_0/ports/1/rate 2>/dev/null
    "

    echo ""
    echo "=== end ==="
} > "$OUT" 2>&1

echo "[env_check] saved -> $OUT"
