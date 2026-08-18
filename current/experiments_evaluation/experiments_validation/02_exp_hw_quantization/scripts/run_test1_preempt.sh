#!/bin/bash
# ============================================================================
# run_test1_preempt.sh — 实验2 测试1：P6 严格抢占 P3 验证（含重试机制）
# ============================================================================
# 场景：同一节点上两个逻辑作业（NCCL Multi-comm 机制），分别映射到 P3 与 P6。
#   作业 A：固定 P3（DSCP=16 / tc:2）
#   作业 B：固定 P6（DSCP=8 / tc:0，SP 最高优先级）
# 预期：P6 完全抢占 → P6 带宽 ≈ solo，P3 带宽 ≈ 0（饿死）。
#
# 重试机制：10.1 防火墙对入站目标端口 20000-30000 / 60000-65000 段 REJECT，
#   NCCL listen 端口（32768-60999）偶发落进 60000-60999 → connect EHOSTUNREACH
#   → comm init 挂死（~20-25%）。solo 与并发阶段均自动重试直到成功。
# 饿死设计：P3 被 P6 严格抢占后每 iter 通信时间暴增，不会在 iters 内完成，
#   由 timeout 90 兜底；判定以 P6 完成 + P3 有数据为准。
#
# Usage:
#   bash run_test1_preempt.sh [round]
# ============================================================================
set -uo pipefail

ROUND=${1:-1}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp2_test1_r${ROUND}_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
REMOTE_EXP_DIR="/home/why/LongLiu_rebuild/experiments_validation/02_exp_hw_quantization/scripts"
REMOTE_DATA_BASE="/home/why/LongLiu_rebuild/experiments_validation/02_exp_hw_quantization/data"
REMOTE_OUTDIR="$REMOTE_DATA_BASE/$RUN_ID"
SCHED_10="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
SCHED_226="/home/why/LongLiu_rebuild/multi_comm_slo/src"

PAYLOAD_MB=512
SOLO_ITERS=20
MAIN_ITERS_A=150    # P3 长作业（正常 ~50s；被饿死时由 timeout 兜底）
MAIN_ITERS_B=60     # P6 短作业（solo ~7s），中途插入抢占
SLEEP_US=10000
PORT_BASE=$((29710 + ROUND * 10))
MON_DURATION=180
TIMEOUT_SOLO=60
TIMEOUT_A=90
TIMEOUT_B=40
MAX_RETRY_SOLO=5
MAX_RETRY_MAIN=5

echo "================================================================"
echo "Exp2 测试1: P6 抢占 P3 — Round $ROUND (含重试)"
echo "  作业A: P3 (DSCP=16/tc:2), 作业B: P6 (DSCP=8/tc:0)"
echo "  payload=${PAYLOAD_MB}MB, sleep=${SLEEP_US}us, iters_A=${MAIN_ITERS_A}, iters_B=${MAIN_ITERS_B}"
echo "  timeout: solo=${TIMEOUT_SOLO}s jobA=${TIMEOUT_A}s jobB=${TIMEOUT_B}s"
echo "  输出: $OUTDIR"
echo "================================================================"

# 清理两端残留 job
cleanup_jobs() {
    pkill -f 'fixed_prio_job.py' 2>/dev/null
    ssh "$NODE_226" "pkill -f 'fixed_prio_job.py' 2>/dev/null; true"
    sleep 2
}

# 公共环境前缀函数（10.1）
launch_10() {  # launch_10 <port> <label> <prio> <iters> <timeout> <out_suffix>
    local port=$1 label=$2 prio=$3 iters=$4 to=$5 suffix=$6
    CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
        MASTER_PORT=$port MULTI_COMM_PORT=$port MULTI_COMM_SRC=$SCHED_10 \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        timeout $to python3 -u "$BASE_DIR/scripts/fixed_prio_job.py" --label "$label" --priority "$prio" \
            --payload-mb $PAYLOAD_MB --num-iters "$iters" --sleep-us $SLEEP_US \
            --outdir "$OUTDIR" --solo-bw "${SOLO_BW:-0}" \
            > "$OUTDIR/exp2_${label}_rank0_${suffix}.log" 2>&1 &
    LAST_PID=$!
}

launch_226() {  # launch_226 <port> <label> <prio> <iters> <timeout> <out_suffix>
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
    LAST_PID=$!
}

check_solo_ok() {  # rank0 solo 日志含 平均带宽
    grep -q '平均带宽 = ' "$OUTDIR/exp2_soloB_rank0_solocalib.log" 2>/dev/null
}
check_main_ok() {  # JobB(P6) 完成 + JobA(P3) 有数据
    grep -q '平均带宽 = ' "$OUTDIR/exp2_jobB_rank0_main.log" 2>/dev/null && \
    [[ "$(wc -l < "$OUTDIR/exp2_jobA_rank0_iter.csv" 2>/dev/null || echo 0)" -gt 5 ]]
}

# ------------------------------------------------------------------
# Step 1: solo 校准（P6 solo，20 iters，重试）→ solo 参考带宽
# ------------------------------------------------------------------
echo "=== Step 1: solo 校准 (P6 solo, 最多 ${MAX_RETRY_SOLO} 次) ==="
SOLO_BW=0
SOLO_OK=0
for s in $(seq 1 $MAX_RETRY_SOLO); do
    cleanup_jobs
    P=$((PORT_BASE + 1))
    rm -f "$OUTDIR/exp2_soloB_rank0_solocalib.log"
    echo "  [solo attempt $s] port=$P"
    launch_10 $P soloB 6 $SOLO_ITERS $TIMEOUT_SOLO solocalib; PID10=$LAST_PID
    launch_226 $P soloB 6 $SOLO_ITERS $TIMEOUT_SOLO solocalib; PID26=$LAST_PID
    wait $PID10; E10=$?
    wait $PID26; E26=$?
    if check_solo_ok; then
        SOLO_BW=$(grep -oP '平均带宽 = \K[\d.]+' "$OUTDIR/exp2_soloB_rank0_solocalib.log" | tail -1 || echo 0)
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

# 并发期间 NIC prio 计数器采样（覆盖争抢窗口，验证 226 是否分类）
# 输出 $MAIN_SUB/nic_10_conc.csv / nic_226_conc.csv
start_conc_monitor() {
    local sub=$1
    bash "$COMMON_DIR/monitor_nic.sh" "conc" 600 1 "$sub" \
        > "$sub/monitor_conc.log" 2>&1 &
    CONC_MON_PID=$!
}
stop_conc_monitor() {
    [[ -n "${CONC_MON_PID:-}" ]] && kill "$CONC_MON_PID" 2>/dev/null
    # monitor_nic 以 run_id=conc 写入 $sub/conc/，移回 attempt 顶层
    local sub=$1
    mv -f "$sub/conc/nic_10.csv" "$sub/nic_10_conc.csv" 2>/dev/null
    mv -f "$sub/conc/nic_226.csv" "$sub/nic_226_conc.csv" 2>/dev/null
    rmdir "$sub/conc" 2>/dev/null
    CONC_MON_PID=
}

# ------------------------------------------------------------------
# Step 2: 并发 P3 vs P6（整体重试，最多 MAX_RETRY_MAIN 次）
# ------------------------------------------------------------------
echo "=== Step 2: 并发 P3 vs P6 (最多 ${MAX_RETRY_MAIN} 次) ==="
MAIN_OK=0
for m in $(seq 1 $MAX_RETRY_MAIN); do
    cleanup_jobs
    # 并发轮次子目录：保留每次尝试
    MAIN_SUB="$OUTDIR/attempt${m}"
    mkdir -p "$MAIN_SUB"
    echo "  [main attempt $m]"
    PA=$((PORT_BASE + 2))
    PB=$((PORT_BASE + 3))
    # 作业 A (P3) 先启动，作业 B (P6) 后启动 3s
    start_conc_monitor "$MAIN_SUB"
    launch_10 $PA jobA 3 $MAIN_ITERS_A $TIMEOUT_A main; PID10_A=$LAST_PID
    launch_226 $PA jobA 3 $MAIN_ITERS_A $TIMEOUT_A main; PID26_A=$LAST_PID
    sleep 3
    launch_10 $PB jobB 6 $MAIN_ITERS_B $TIMEOUT_B main; PID10_B=$LAST_PID
    launch_226 $PB jobB 6 $MAIN_ITERS_B $TIMEOUT_B main; PID26_B=$LAST_PID

    wait $PID10_A; echo "  JobA(P3) 10.1 exit=$?"
    wait $PID26_A; echo "  JobA(P3) 226 exit=$?"
    wait $PID10_B; echo "  JobB(P6) 10.1 exit=$?"
    wait $PID26_B; echo "  JobB(P6) 226 exit=$?"
    stop_conc_monitor "$MAIN_SUB"

    # 归档本次尝试产物
    cp -f "$OUTDIR"/exp2_jobA_rank0_main.log   "$MAIN_SUB/" 2>/dev/null
    cp -f "$OUTDIR"/exp2_jobB_rank0_main.log   "$MAIN_SUB/" 2>/dev/null
    cp -f "$OUTDIR"/exp2_jobA_rank0_iter.csv   "$MAIN_SUB/" 2>/dev/null
    cp -f "$OUTDIR"/exp2_jobB_rank0_iter.csv   "$MAIN_SUB/" 2>/dev/null
    scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobA_rank1_iter.csv" "$MAIN_SUB/" 2>/dev/null || true
    scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobB_rank1_iter.csv" "$MAIN_SUB/" 2>/dev/null || true

    if check_main_ok; then
        echo "  并发成功：P6 完成且 P3 有数据"
        MAIN_OK=1
        break
    fi
    echo "  并发未成功（P6 或 P3 未产出），重试"
done
if [[ $MAIN_OK -ne 1 ]]; then
    echo "ERROR: 并发 ${MAX_RETRY_MAIN} 次均未成功，退出"
    exit 1
fi

# ------------------------------------------------------------------
# Step 3: 事后监控 180s（稳定性参考；并发窗口证据见 attemptN/nic_*_conc.csv）
# ------------------------------------------------------------------
echo "=== Step 3: 监控（NIC）${MON_DURATION}s ==="
bash "$COMMON_DIR/monitor_nic.sh" "$RUN_ID" "$MON_DURATION" 1 "$DATA_DIR" \
    > "$OUTDIR/monitor_nic.log" 2>&1
echo "  监控完成"

# 最终归档（成功轮数据归入 OUTDIR 顶层）
LAST_SUB=$(ls -d "$OUTDIR"/attempt* 2>/dev/null | tail -1)
if [[ -n "$LAST_SUB" ]]; then
    cp -f "$LAST_SUB"/exp2_job* "$OUTDIR/" 2>/dev/null
fi
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobA_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobB_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_soloB_rank1_solocalib.log" "$OUTDIR/" 2>/dev/null || true

echo ""
echo "================================================================"
echo "测试1 Round $ROUND 完成"
echo "  solo 参考: ${SOLO_BW} Gbps"
echo "  数据: $OUTDIR"
echo "================================================================"
