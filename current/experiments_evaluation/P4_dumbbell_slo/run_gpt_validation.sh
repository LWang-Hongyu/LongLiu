#!/bin/bash
# ============================================================================
# GPT 真实训练验证 — 按序运行 solo / fair / longliu 并汇总 loss 下降
# ============================================================================
# 前提：
#   1. 在 10.1 节点的真实 shell（非沙箱）中执行，且 10.1/226 GPU 空闲
#   2. 已能 ssh 免密到 192.10.10.226
# 流程：
#   Phase 0: 同步最新代码到 226
#   Phase 1: solo  — JOB1 单作业标准 NCCL（基线，验证模型能收敛）
#   Phase 2: fair  — 双作业标准 NCCL（无优先级，验证 datatype 修复后 loss 正常下降）
#   Phase 3: longliu — 双作业 MultiComm + DSCP 调度（核心实验，看 JOB1 受保护 + loss 下降）
# 注意：每个 run 内部有 70s TIME_WAIT 等待 + 300/200 iters，全程约 15-20 分钟。
# ============================================================================

set -e
cd "$(dirname "$0")"

echo "===== Phase 0: 同步代码到 226 ====="
bash sync_to_226.sh

for sched in solo fair longliu; do
    echo ""
    echo "############################################################"
    echo "# [${sched}] 开始运行 train_gpt"
    echo "############################################################"
    bash run_p4.sh train_gpt "$sched"

    echo ""
    echo "--- [${sched}] JOB1 loss 前 5 / 后 5 ---"
    if [ -f "/tmp/p4_train_JOB1_${sched}_rank0.csv" ]; then
        awk -F, 'NR>1{print "  iter "$1"  loss "$6}' "/tmp/p4_train_JOB1_${sched}_rank0.csv" | head -5
        echo "  ..."
        awk -F, 'NR>1{print "  iter "$1"  loss "$6}' "/tmp/p4_train_JOB1_${sched}_rank0.csv" | tail -5
    else
        echo "  WARNING: /tmp/p4_train_JOB1_${sched}_rank0.csv 不存在！"
    fi

    if [[ "$sched" != "solo" ]]; then
        echo "--- [${sched}] JOB2 loss 前 5 / 后 5 ---"
        if [ -f "/tmp/p4_train_JOB2_${sched}_rank0.csv" ]; then
            awk -F, 'NR>1{print "  iter "$1"  loss "$6}' "/tmp/p4_train_JOB2_${sched}_rank0.csv" | head -5
            echo "  ..."
            awk -F, 'NR>1{print "  iter "$1"  loss "$6}' "/tmp/p4_train_JOB2_${sched}_rank0.csv" | tail -5
        else
            echo "  WARNING: /tmp/p4_train_JOB2_${sched}_rank0.csv 不存在！"
        fi
    fi
done

echo ""
echo "===== loss 下降汇总 ====="
python3 - <<'EOF'
import csv, glob
for f in sorted(glob.glob('/tmp/p4_train_JOB*_rank0.csv')):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        print(f"{f.split('/')[-1]:35s} EMPTY")
        continue
    first = float(rows[0]['loss']); last = float(rows[-1]['loss'])
    drop = (first - last) / first * 100
    # JOB1 在 fair 下受竞争影响 loss 下降可能变慢，但不应上升
    print(f"{f.split('/')[-1]:35s} loss {first:.3f} -> {last:.3f}  ({drop:+.1f}%)")
EOF

echo ""
echo "===== 验证完成 ====="
echo "结果 CSV: /tmp/p4_train_JOB*_rank0.csv"
echo "NCCL 日志: /tmp/nccl_j1_101_*.log /tmp/nccl_j1_226_*.log"
