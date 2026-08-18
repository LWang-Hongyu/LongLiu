#!/bin/bash
# ============================================================================
# run_test3_sp_strict.sh — 实验2 测试3：SP 队列严格性判定（受控双流·连续通信）
# ============================================================================
# 目的：判定 SP（严格优先级）队列是否为 per-packet 严格抢占。
#   test1（sleep 10ms）测得抢占度 58.2%±0.2%，两种解释：
#     (a) P6 流量突发/未持续占满队列 → SP 实际严格
#     (b) 队列调度非严格 per-packet（WRR/交换机行为）
#   test3 用 **连续通信**（sleep=0）排除 (a)：P6 流量持续饱和。
#   若 SP 严格 → P6 抢占度应接近 100%（P6≈solo，P3≈0）；
#   若仍 ~58% → 队列调度非严格 per-packet，58% 即硬件物理上限。
#
# 与 test1 的差异（仅机制性差异，其余完全复用）：
#   * --sleep-us 0（连续通信，无 compute 间隔）
#   * payload 512MB（iter 更快，连续模式 ~0.15s/iter）
#   * jobA=P3 100 iters / jobB=P6 200 iters（连续模式迭代快）
#
# Usage:
#   bash run_test3_sp_strict.sh [round]
# ============================================================================
set -uo pipefail

ROUND=${1:-1}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_PREFIX=${RUN_PREFIX:-exp2_test3}     # 数据目录前缀（对照实验可覆盖，避免混入主实验分析）
RUN_ID="${RUN_PREFIX}_r${ROUND}_${TS}"
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
SLEEP_US=0              # ★ 连续通信（本实验核心变量）
SOLO_ITERS=20
MAIN_ITERS_A=100        # 作业A（被饿死时由 timeout 兜底）
MAIN_ITERS_B=200        # 作业B（连续模式 iter 快）
PRIO_A=${PRIO_A:-3}     # 作业A 优先级（默认 P3/DSCP16/tc:2）
PRIO_B=${PRIO_B:-6}     # 作业B 优先级（默认 P6/DSCP8/tc:0）
SOLO_PRIO=${SOLO_PRIO:-6}  # solo 校准优先级（对照实验同优先级时设与作业一致）
dscp_of() { case "$1" in 6) echo 8;; 3) echo 16;; 4) echo 0;; *) echo "?";; esac; }
PORT_BASE=$((29810 + ROUND * 10))
MON_DURATION=120
TIMEOUT_SOLO=60
TIMEOUT_A=120
TIMEOUT_B=90
MAX_RETRY_SOLO=5
MAX_RETRY_MAIN=5

echo "================================================================"
echo "Exp2 测试3: SP 严格性判定（连续通信）— Round $ROUND"
echo "  作业A: P${PRIO_A} (DSCP=$(dscp_of $PRIO_A)), 作业B: P${PRIO_B} (DSCP=$(dscp_of $PRIO_B))"
echo "  payload=${PAYLOAD_MB}MB sleep=${SLEEP_US}us（连续）"
echo "  iters_A=${MAIN_ITERS_A}, iters_B=${MAIN_ITERS_B}"
echo "  timeout: solo=${TIMEOUT_SOLO}s jobA=${TIMEOUT_A}s jobB=${TIMEOUT_B}s"
echo "  输出: $OUTDIR"
echo "================================================================"

# 清理两端残留 job（注意避免 pkill 匹配到本脚本/ssh 自身）
cleanup_jobs() {
    pkill -f 'fixed_prio_job.py' 2>/dev/null
    ssh "$NODE_226" "pkill -f 'fixed_prio_job.py' 2>/dev/null; true"
    sleep 2
}

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
# Step 1: solo 校准（P6 solo，连续模式 20 iters，重试）→ solo 参考带宽
# ------------------------------------------------------------------
echo "=== Step 1: solo 校准 (P6 solo 连续, 最多 ${MAX_RETRY_SOLO} 次) ==="
SOLO_BW=0
SOLO_OK=0
for s in $(seq 1 $MAX_RETRY_SOLO); do
    cleanup_jobs
    P=$((PORT_BASE + 1))
    rm -f "$OUTDIR/exp2_soloB_rank0_solocalib.log"
    echo "  [solo attempt $s] port=$P"
    launch_10 $P soloB $SOLO_PRIO $SOLO_ITERS $TIMEOUT_SOLO solocalib; PID10=$LAST_PID
    launch_226 $P soloB $SOLO_PRIO $SOLO_ITERS $TIMEOUT_SOLO solocalib; PID26=$LAST_PID
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

# 并发期间直接采样两端 NIC tx_prio 计数器（判定链路饱和/队列分配）
start_conc_monitor() {
    local sub=$1
    ( for i in $(seq 1 240); do
        ts=$(date +%s.%3N)
        t10=$(ethtool -S enp130s0f0np0 2>/dev/null | grep -E 'tx_prio[0-7]_bytes' | awk '{printf "%s,", $2}')
        t26=$(ssh "$NODE_226" "ethtool -S enp59s0f0np0 2>/dev/null | grep -E 'tx_prio[0-7]_bytes' | awk '{printf \"%s,\", \$2}'")
        echo "$ts,$t10$t26" >> "$sub/nic_prio_conc.csv"
        sleep 1
    done ) &
    CONC_MON_PID=$!
}
stop_conc_monitor() {
    [[ -n "${CONC_MON_PID:-}" ]] && kill "$CONC_MON_PID" 2>/dev/null
    CONC_MON_PID=
}

# ------------------------------------------------------------------
# Step 2: 并发 P3 vs P6（连续通信；整体重试）
# ------------------------------------------------------------------
echo "=== Step 2: 并发 P3 vs P6 连续通信 (最多 ${MAX_RETRY_MAIN} 次) ==="
MAIN_OK=0
for m in $(seq 1 $MAX_RETRY_MAIN); do
    cleanup_jobs
    MAIN_SUB="$OUTDIR/attempt${m}"
    mkdir -p "$MAIN_SUB"
    echo "  [main attempt $m]"
    PA=$((PORT_BASE + 2))
    PB=$((PORT_BASE + 3))
    start_conc_monitor "$MAIN_SUB"
    launch_10 $PA jobA $PRIO_A $MAIN_ITERS_A $TIMEOUT_A main; PID10_A=$LAST_PID
    launch_226 $PA jobA $PRIO_A $MAIN_ITERS_A $TIMEOUT_A main; PID26_A=$LAST_PID
    sleep 3
    launch_10 $PB jobB $PRIO_B $MAIN_ITERS_B $TIMEOUT_B main; PID10_B=$LAST_PID
    launch_226 $PB jobB $PRIO_B $MAIN_ITERS_B $TIMEOUT_B main; PID26_B=$LAST_PID

    wait $PID10_A; echo "  JobA(P$PRIO_A) 10.1 exit=$?"
    wait $PID26_A; echo "  JobA(P$PRIO_A) 226 exit=$?"
    wait $PID10_B; echo "  JobB(P$PRIO_B) 10.1 exit=$?"
    wait $PID26_B; echo "  JobB(P$PRIO_B) 226 exit=$?"
    stop_conc_monitor "$MAIN_SUB"

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
# Step 3: 事后监控（NIC 计数器参考）
# ------------------------------------------------------------------
echo "=== Step 3: 监控（NIC）${MON_DURATION}s ==="
bash "$COMMON_DIR/monitor_nic.sh" "$RUN_ID" "$MON_DURATION" 1 "$DATA_DIR" \
    > "$OUTDIR/monitor_nic.log" 2>&1
echo "  监控完成"

# 最终归档
LAST_SUB=$(ls -d "$OUTDIR"/attempt* 2>/dev/null | tail -1)
if [[ -n "$LAST_SUB" ]]; then
    cp -f "$LAST_SUB"/exp2_job* "$OUTDIR/" 2>/dev/null
fi
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobA_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_jobB_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp2_soloB_rank1_solocalib.log" "$OUTDIR/" 2>/dev/null || true

echo ""
echo "================================================================"
echo "测试3 Round $ROUND 完成"
echo "  solo 参考: ${SOLO_BW} Gbps（连续通信模式）"
echo "  数据: $OUTDIR"
echo "================================================================"
