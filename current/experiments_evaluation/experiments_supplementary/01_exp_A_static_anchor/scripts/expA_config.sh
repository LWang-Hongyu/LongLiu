#!/bin/bash
# ============================================================================
# Experiment A: Static Anchor — Shared Configuration
# ============================================================================
# Source this file from other scripts: source expA_config.sh
# ============================================================================
# Design rationale (see README.md §Design decisions):
#   - "Capacity" dimension is realized via payload size modulation, since
#     per-QP rate limiting is unavailable (no sudo, no API in multi_comm_slo).
#   - Solo BW is calibrated per payload; the simulator mirrors the same
#     payload × sleep × c_i, so the HW-vs-Sim comparison is bit-for-bit fair.
#   - 6 scenarios sample the {jobs: 2 / 2+bg} × {capacity: 50G / 35G / 25G}
#     × {c_i: 1.2 / 1.5} grid; S5 is the hold-out (theoretical anchor).
# ============================================================================

# ---- Workload parameters ----
SLEEP_US=30000              # 30ms compute per iter (same as V5/V6)
ITERS_PER_EPOCH=20
NUM_EPOCHS=25
NUM_ITERS=$((NUM_EPOCHS * ITERS_PER_EPOCH))   # 500
# Static mode: reverse-epoch set beyond experiment horizon → no swap
REVERSE_EPOCH_OFF=999

# ---- Payloads → capacity mapping (calibrated via solo bench) ----
PAYLOAD_50G=1024            # 1024MB → solo BW ≈ 40-50G (link limited)
PAYLOAD_35G=768             # 768MB  → solo BW ≈ 30-35G
PAYLOAD_25G=512             # 512MB  → solo BW ≈ 20-25G

# ---- c_i values ----
CI_TIGHT=1.2
CI_LOOSE=1.5

# ---- Background flow (reuse V6 calibration: 6G DSCP=P3) ----
BG_NUM_FLOWS=12
BG_RATE_MBPS=500            # 12 × 500M = 6G total
BG_PORT_START=6200
BG_PORT_END=6211
BG_TOS=64                   # DSCP=16 (P3) << 2 = 64
BG_DURATION=600             # 10 min (covers one scenario)

# ---- Network / node config ----
NODE_226_SSH="10.157.197.107"           # SSH reachable IP
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
NIC_10="enp130s0f0np0"
NIC_226="enp59s0f0np0"
PORT_JA=29530
PORT_JB=29531

# ---- Paths ----
EXP_SUP_DIR="/home/why/LongLiu_rebuild/experiments_supplementary"
EXP_A_DIR="${EXP_SUP_DIR}/01_exp_A_static_anchor"
SCRIPTS_DIR="${EXP_A_DIR}/scripts"
DATA_DIR="${EXP_A_DIR}/data"
LOGS_DIR="${EXP_A_DIR}/logs"
ANALYSIS_DIR="${EXP_A_DIR}/analysis"
P4_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
JOB_SCRIPT="${P4_DIR}/p4_job_reverse.py"
SIM_DIR="/home/why/LongLiu/simulation"

# ---- T_target files (per payload; 1024MB reuses V5) ----
TTARGET_1024="/tmp/ttarget_v5_jobA.json"        # Reuse V5 (already calibrated)
TTARGET_768_A="/tmp/ttarget_expA_jobA_768.json"
TTARGET_768_B="/tmp/ttarget_expA_jobB_768.json"
TTARGET_512_A="/tmp/ttarget_expA_jobA_512.json"
TTARGET_512_B="/tmp/ttarget_expA_jobB_512.json"

# ---- NCCL env (common) ----
NCCL_ENV_COMMON="NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN"
NCCL_ENV_10="NCCL_SOCKET_IFNAME=${NIC_10}"
NCCL_ENV_226="NCCL_SOCKET_IFNAME=${NIC_226}"
LD_LIBRARY_PATH_NCCL="/home/why/LongLiu_rebuild/nccl-master/build/lib"
PYTHONPATH_SLO="/home/why/LongLiu_rebuild/multi_comm_slo/src"

# ---- Helper: ensure dirs ----
ensure_dirs() {
    mkdir -p "$SCRIPTS_DIR" "$DATA_DIR" "$LOGS_DIR" "$ANALYSIS_DIR"
}

# ---- Helper: cleanup orphan processes (precise, no wide pkill) ----
cleanup_jobs() {
    local pids
    pids=$(pgrep -f "p4_job_reverse.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[cleanup] killing local pids: $pids"
        kill $pids 2>/dev/null || true
    fi
    pids=$(ssh -o ConnectTimeout=10 "$NODE_226_SSH" "pgrep -f p4_job_reverse.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[cleanup] killing 226 pids: $pids"
        ssh -o ConnectTimeout=10 "$NODE_226_SSH" "kill $pids" 2>/dev/null || true
    fi
}

# ---- Helper: cleanup iperf3 ----
cleanup_bg() {
    local pids
    pids=$(pgrep -f "iperf3.*-c.*${RDMA_226}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[bg] stopping local iperf3 clients: $pids"
        kill $pids 2>/dev/null || true
    fi
    ssh -o ConnectTimeout=10 "$NODE_226_SSH" "pkill -x iperf3" 2>/dev/null || true
}

# ---- Helper: start iperf3 servers on 226 ----
start_bg_servers() {
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        ssh -o ConnectTimeout=10 "$NODE_226_SSH" \
            "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null || true
    done
    sleep 2
}

# ---- Helper: start iperf3 UDP clients on 10.1 (background) ----
start_bg_clients() {
    BG_CLIENT_PIDS=""
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        iperf3 -c "$RDMA_226" -u -b "${BG_RATE_MBPS}M" -t "$BG_DURATION" \
            --tos "$BG_TOS" -p "$PORT" -f g -l 8900 \
            > "$LOGS_DIR/bgflow_${PORT}.log" 2>&1 &
        BG_CLIENT_PIDS="$BG_CLIENT_PIDS $!"
    done
    echo "[bg] started ${BG_NUM_FLOWS} iperf3 UDP clients (rate=${BG_RATE_MBPS}M, tos=${BG_TOS}, dur=${BG_DURATION}s)"
    sleep 3   # let flows ramp up
}

# ---- Helper: verify T_target file ----
verify_ttarget() {
    local f=$1
    if [ ! -f "$f" ]; then
        echo "[ERR] T_target file missing: $f"
        return 1
    fi
    python3 -c "
import json, sys
d = json.load(open('$f'))
assert d.get('unit') == 'per_epoch_ms', f'unit mismatch: {d.get(\"unit\")}'
assert d.get('target_comm_time_ms', 0) > 0, 'non-positive target_comm_time_ms'
print(f'[ok] {\"$f\"}: payload={d[\"payload_mb\"]}MB, T_target_epoch={d[\"target_comm_time_ms\"]:.1f}ms')
" || return 1
    return 0
}
