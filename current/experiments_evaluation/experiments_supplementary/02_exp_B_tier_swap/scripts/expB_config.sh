#!/bin/bash
# ============================================================================
# Experiment B: Hardware Tier Swap — Shared Configuration
# ============================================================================
# Source this file from other scripts: source expB_config.sh
# ============================================================================

# ---- Workload parameters (from LongLiu_补充实验方案.md §B.1) ----
PAYLOAD_MB=1024          # Same as V5/V6 (reuse V5 T_target)
CI_PREMIUM=1.2           # Tight SLO (premium tier)
CI_STANDARD=2.0          # Loose SLO (standard tier)
SLEEP_US=30000           # 30ms compute per iter (same as V5/V6)

# ---- Iteration / Epoch layout ----
# 25 epochs × 20 iters/epoch = 500 iters total
# At ~15-20s/epoch (with contention), total ~6-8 min per arm
ITERS_PER_EPOCH=20
NUM_EPOCHS=25
NUM_ITERS=$((NUM_EPOCHS * ITERS_PER_EPOCH))   # 500
REVERSE_EPOCH=8           # T_swap: c_i swap at epoch 8

# ---- Measurement windows (epoch-based, approximating 100s time windows) ----
# W1 = [T_swap−100s, T_swap]       → epochs 3-7 (5 epochs before swap)
# W2 = [T_swap, T_swap+100s]       → epochs 8-12 (5 epochs just after swap)
# Gap = [T_swap+100s, T_swap+200s] → epochs 13-17 (5 epochs transition)
# W3 = [T_swap+200s, T_swap+300s]  → epochs 18-22 (5 epochs later)
# Buffer: epochs 23-24 (cleanup, not analyzed)
W1_START=3;  W1_END=7
W2_START=8;  W2_END=12    # W2_START = REVERSE_EPOCH
W3_START=18; W3_END=22

# ---- CRUX static priorities (label once at start, never change) ----
# A starts as premium → assign higher static priority
# B starts as standard → assign lower static priority
# After swap, these DO NOT change (this is the "lost lock" behavior)
CRUX_PRIO_A=4             # P4 (DSCP=0, tc:1 — second highest)
CRUX_PRIO_B=3             # P3 (DSCP=16, tc:2 — third)
# LongLiu initial priority (both jobs start at P3, same as V6)
LL_INITIAL_PRIORITY=3

# ---- Background flow (from V6 calibration: 6G DSCP=P3) ----
BG_NUM_FLOWS=12
BG_RATE_MBPS=500          # 12 × 500M = 6G total
BG_PORT_START=6200
BG_PORT_END=6211
BG_TOS=64                 # DSCP=16 (P3) << 2 = 64
BG_DURATION=900           # 15 min (covers full arm with contention margin)

# ---- T_target files (reuse V5 calibration, 1024MB payload) ----
TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"

# ---- Network / node config ----
NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
PORT_JA=29520
PORT_JB=29521

# ---- Directory layout ----
EXP_SUP_DIR="/home/why/LongLiu_rebuild/experiments_supplementary"
EXP_B_DIR="${EXP_SUP_DIR}/02_exp_B_tier_swap"
P4_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
JOB_SCRIPT="${P4_DIR}/p4_job_reverse.py"

# ---- Round definitions (4 rounds, alternation) ----
# Dimensions: {job start order} × {arm run order}
# Round 1: A→B,  LL→CX
# Round 2: A→B,  CX→LL
# Round 3: B→A,  LL→CX
# Round 4: B→A,  CX→LL
declare -A ROUND_JOB_ORDER=( [1]="AB" [2]="AB" [3]="BA" [4]="BA" )
declare -A ROUND_ARM_ORDER=( [1]="ll_cx" [2]="cx_ll" [3]="ll_cx" [4]="cx_ll" )

# ---- Helper: safe process cleanup (no pkill -9 -f, per project_memory) ----
cleanup_p4_jobs() {
    local label="$1"
    # Local cleanup
    for PID in $(pgrep -f "p4_job_reverse.py --job [AB]" 2>/dev/null); do
        echo "  [$label] Killing local PID $PID"
        kill "$PID" 2>/dev/null || true
    done
    # Remote cleanup (226)
    ssh -o ConnectTimeout=10 "$NODE_226" \
        "for PID in \$(pgrep -f 'p4_job_reverse.py --job [AB]' 2>/dev/null); do
             echo '  [$label] Killing 226 PID '\$PID
             kill \$PID 2>/dev/null
         done" 2>/dev/null || true
    sleep 2
}

# ---- Helper: verify T_target files exist on both nodes ----
verify_ttarget() {
    local ok=0  # 0 = success (bash convention)
    for f in "$TTARGET_A" "$TTARGET_B"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: T_target file missing on 10.1: $f"
            ok=1
        fi
        # Check unit field (must be per_epoch_ms, per project_memory)
        local unit
        unit=$(python3 -c "import json; print(json.load(open('$f')).get('unit','MISSING'))" 2>/dev/null)
        if [[ "$unit" != "per_epoch_ms" ]]; then
            echo "ERROR: T_target unit='$unit' (expected 'per_epoch_ms') in $f"
            ok=1
        fi
    done
    # Ensure files exist on 226 too
    ssh -o ConnectTimeout=10 "$NODE_226" "test -f $TTARGET_A && test -f $TTARGET_B" 2>/dev/null || {
        echo "Copying T_target files to 226..."
        scp -q "$TTARGET_A" "$NODE_226:$TTARGET_A"
        scp -q "$TTARGET_B" "$NODE_226:$TTARGET_B"
    }
    return $ok
}

# ---- Helper: md5 checksum of job script (for reproducibility) ----
record_md5() {
    local outfile="$1"
    md5sum "$JOB_SCRIPT" > "$outfile"
    ssh -o ConnectTimeout=10 "$NODE_226" "md5sum $JOB_SCRIPT" >> "$outfile" 2>/dev/null
}
