#!/bin/bash
# ============================================================================
# run_test2_p3share.sh — 实验2 测试2：P3 内部带宽共享验证（含重试机制）
# ============================================================================
# 场景：同一节点上 3 个逻辑作业均映射到 P3（同一 DSCP=16 → 同一 tc:2 队列）。
#   因硬件只能区分到 TC，P3 内部无法再细分优先级 → 3 流在队列内按 FIFO 争抢。
# 测量：各流带宽分配 → 验证 FIFO/近似公平程度，量化"无法细分优先级导致的性能慢度"。
#
# 重试机制：与 test1 相同 —— 10.1 防火墙对部分端口 REJECT 导致 NCCL listen
#   偶发挂死，solo 与并发阶段均自动重试直到成功。
#
# Usage:
#   bash run_test2_p3share.sh [round]
# ============================================================================
set -uo pipefail

ROUND=${1:-1}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp2_test2_r${ROUND}_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
REMOTE_EXP_DIR="/home/why/LongLiu_rebuild/experiments_validation/02_exp_hw_quantization/scripts"
REMOTE_DATA_BASE="/home/why/LongLiu_rebuild/experiments_validation/02_exp_hw_quantization/data"
REMOTE_OUTDIR="$REMOTE_DATA_BASE/$RUN_ID"
SCHED_10="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
SCHED_226="/home/why/LongLiu_rebuild/multi_comm_slo/src"

PAYLOAD_MB=256
SOLO_ITERS=20
MAIN_ITERS=60
SLEEP_US=10000
PORT_BASE=$((29810 + ROUND * 10))
MON_DURATION=200
TIMEOUT_SOLO=40
TIMEOUT_MAIN=60
MAX_RETRY_SOLO=5
MAX_RETRY_MAIN=5

echo "================================================================"
echo "Exp2 测试2: 3×P3 内部共享 — Round $ROUND (含重试)"
echo "  3 个作业均 P3 (DSCP=16/tc:2, 同一 TC 队列)"
echo "  payload=${PAYLOAD_MB}MB, sleep=${SLEEP_US}us, main_iters=${MAIN_ITERS}"
echo "  timeout: solo=${TIMEOUT_SOLO}s main=${TIMEOUT_MAIN}s"
echo "  输出: $OUTDIR"
echo "================================================================"

cleanup_jobs() {
    pkill -f 'fixed_prio_job.py' 2>/dev/null
    ssh "$NODE_226" "pkill -f 'fixed_prio_job.py' 2>/dev/null; true"
    sleep 2
}

launch_10() {  # <port> <label> <prio> <iters> <timeout> <suffix>
    local port=$1 label=$2 prio=$3 iters=$4 to=$5 suffix=$6
    CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
        MASTER_PORT=$port MULTI_COMM_PORT=$port MULTI_COMM_SRC=$SCHED_10 \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        timeout $to python3 -u "$BASE_DIR/scripts/fixed_prio_job.py" --label "$label" --priority "$prio" \
            --payload-mb $PAYLOAD_MB --num-iters "$iters" --sleep-us $SLEEP_US \
            --outdir "$OUTDIR" --solo-bw "${SOLO_BW:-0}" \
            > "$OUTDIR/exp2_${label}_rank0_${suffix}.log" 2>&1 &
    echo $!
}

launch_226() {  # <port> <label> <prio> <iters> <timeout> <suffix>
    local port=$1 label=$2 prio=$3 iters=$4 to=$5 suffix=$6
    ssh "$NODE_226" "mkdir -p '$REMOTE_OUTDIR' && cd $REMOTE_EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
        MASTER_PORT=$port MULTI_COMM_PORT=$port MULTI_COMM_SRC=$SCHED_226 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
        NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        timeout $to python3 -u fixed_prio_job.py --label $label --priority $prio \
            --payload-mb $PAYLOAD_MB --num-iters $iters --sleep-us $SLEEP_US \
            --outdir $REMOTE_OUTDIR --solo-bw ${SOLO_BW:-0} \
            > $REMOTE_OUTDIR/exp2_${label}_rank1_${suffix}.log 2>&1" &
    echo $!
}

check_solo_ok() {
    grep -q '平均带宽 = ' "$OUTDIR/exp2_solo_rank0_solocalib.log" 2>/dev/null
}

check_main_ok() {  # 3 个流都有 iter 数据
    local ok=1
    for i in 1 2 3; do
        [[ "$(wc -l < "$OUTDIR/exp2_p3flow${i}_rank0_iter.csv" 2>/dev/null || echo 0)" -gt 5 ]] || ok=0
    done
    [[ $ok -eq 1 ]]
}

# ------------------------------------------------------------------
# Step 1: solo 校准（P3 solo，重试）→ solo 参考带宽
# ------------------------------------------------------------------
echo "=== Step 1: solo 校准 (P3 solo, 最多 ${MAX_RETRY_SOLO} 次) ==="
SOLO_BW=0
SOLO_OK=0
for s in $(seq 1 $MAX_RETRY_SOLO); do
    cleanup_jobs
    P=$((PORT_BASE + 1))
    rm -f "$OUTDIR/exp2_solo_rank0_solocalib.log"
    echo "  [solo attempt $s] port=$P"
    PID10=$(launch_10 $P solo 3 $SOLO_ITERS $TIMEOUT_SOLO solocalib)
    PID26=$(launch_226 $P solo 3 $SOLO_ITERS $TIMEOUT_SOLO solocalib)
    wait $PID10; E10=$?
    wait $PID26; E26=$?
    if check_solo_ok; then
        SOLO_BW=$(grep -oP '平均带宽 = \K[\d.]+' "$OUTDIR/exp2_solo_rank0_solocalib.log" | tail -1 || echo 0)
        echo "  solo 校准成功: ${SOLO_BW} Gbps"
        SOLO_OK=1
        break
    fi
    echo "  solo 失败(r0=$E10 r1=$E26)，重试"
done
if [[ $SOLO_OK -ne 1 ]]; then
    echo "ERROR: solo 校准 ${MAX_RETRY_SOLO} 次均失败，退出"
    exit 1
fi
sleep 3

# ------------------------------------------------------------------
# Step 2: 3×P3 并发（整体重试，最多 MAX_RETRY_MAIN 次）
# ------------------------------------------------------------------
echo "=== Step 2: 3×P3 并发 (最多 ${MAX_RETRY_MAIN} 次) ==="
MAIN_OK=0
for m in $(seq 1 $MAX_RETRY_MAIN); do
    cleanup_jobs
    echo "  [main attempt $m]"
    for i in 1 2 3; do
        rm -f "$OUTDIR/exp2_p3flow${i}_rank0_iter.csv"
    done
    bash "$COMMON_DIR/monitor_nic.sh" "conc" 300 1 "$OUTDIR" \
        > "$OUTDIR/monitor_conc_${m}.log" 2>&1 &
    MON_PID=$!

    declare -a PIDS=()
    for i in 1 2 3; do
        PORT=$((PORT_BASE + 1 + i))
        PID10=$(launch_10 $PORT "p3flow$i" 3 $MAIN_ITERS $TIMEOUT_MAIN main)
        PID26=$(launch_226 $PORT "p3flow$i" 3 $MAIN_ITERS $TIMEOUT_MAIN main)
        PIDS+=("$PID10" "$PID26")
        sleep 3   # 依次启动，观察 FIFO 对先到流的影响
    done

    for PID in "${PIDS[@]}"; do
        wait $PID 2>/dev/null
    done
    echo "  3 个 P3 作业结束"

    kill "$MON_PID" 2>/dev/null
    mv -f "$OUTDIR/conc/nic_10.csv" "$OUTDIR/nic_10_conc.csv" 2>/dev/null
    mv -f "$OUTDIR/conc/nic_226.csv" "$OUTDIR/nic_226_conc.csv" 2>/dev/null
    rmdir "$OUTDIR/conc" 2>/dev/null

    # 归档 rank1 数据
    for i in 1 2 3; do
        scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_p3flow${i}_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
    done
    scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_solo_rank1_solocalib.log" "$OUTDIR/" 2>/dev/null || true

    if check_main_ok; then
        echo "  并发成功：3 个 P3 流均有数据"
        MAIN_OK=1
        break
    fi
    echo "  并发未成功，重试"
done
if [[ $MAIN_OK -ne 1 ]]; then
    echo "ERROR: 并发 ${MAX_RETRY_MAIN} 次均未成功，退出"
    exit 1
fi

echo ""
echo "================================================================"
echo "测试2 Round $ROUND 完成"
echo "  solo 参考: ${SOLO_BW} Gbps"
echo "  数据: $OUTDIR"
echo "================================================================"
