#!/bin/bash
# ============================================================================
# Experiment A: HW Runner — 6 static scenarios
# ============================================================================
# Runs each scenario as 2 concurrent NCCL jobs (Job A + Job B) in static mode
# (no tier swap). Optional iperf3 background flow for S3/S4.
#
# Startup pattern (mirrors run_expB_round.sh, proven to work):
#   1. Launch FIRST job: rank 1 on 226 → wait 5s → rank 0 on 10.1
#   2. Wait 15s for FIRST job NCCL init to complete
#   3. Launch SECOND job: rank 1 on 226 → wait 5s → rank 0 on 10.1
#   4. Wait for both to finish
#
# Usage: bash run_expA_hw.sh [scenario_id]
#   no arg  → run all 6 scenarios sequentially
#   S1      → run only S1
# ============================================================================
set -e
source "$(dirname "$0")/expA_config.sh"
ensure_dirs

SCENARIO_FILTER=${1:-ALL}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DATA_DIR/run_${RUN_TS}"
mkdir -p "$RUN_DIR"
echo "[run] data dir: $RUN_DIR"

# ---- Per-scenario HW runner (mirrors run_expB_round.sh pattern) ----
run_scenario() {
    local SID=$1
    local LABEL=$2
    local JOBS=$3
    local BG=$4
    local PAYLOAD=$5
    local CI=$6
    local TTARGET_A=$7
    local TTARGET_B=$8

    local SCEN_DIR="$RUN_DIR/${SID}_${LABEL}"
    mkdir -p "$SCEN_DIR"
    echo ""
    echo "================================================================"
    echo "[SCENARIO $SID] $LABEL"
    echo "  payload=${PAYLOAD}MB  c_i=${CI}  bg=${BG}  jobs=${JOBS}"
    echo "  ttarget_a=${TTARGET_A}  ttarget_b=${TTARGET_B}"
    echo "================================================================"

    # Verify T_target files; generate theoretical for hold-out S5
    if [ "$TTARGET_A" = "THEORETICAL" ]; then
        TTARGET_A="/tmp/ttarget_expA_theoretical_${PAYLOAD}MB_A.json"
        TTARGET_B="/tmp/ttarget_expA_theoretical_${PAYLOAD}MB_B.json"
        python3 -c "
import json
payload_mb = $PAYLOAD
link_bw_gbps = 50.0
payload_bits = payload_mb * 1024 * 1024 * 8
per_iter_ms = (payload_bits / (link_bw_gbps * 1e9)) * 1000
per_epoch_ms = per_iter_ms * $ITERS_PER_EPOCH
for job in ['A', 'B']:
    d = {'job': job, 'mode': 'longliu', 'payload_mb': payload_mb,
         'c_i_calib': $CI, 'sleep_us': $SLEEP_US,
         'calib_epochs': 0, 'iters_per_epoch': $ITERS_PER_EPOCH,
         'target_comm_time_ms': per_epoch_ms, 'unit': 'per_epoch_ms',
         'source': 'theoretical', 'timestamp': '2026-07-28'}
    out = '/tmp/ttarget_expA_theoretical_${PAYLOAD}MB_' + job + '.json'
    json.dump(d, open(out, 'w'), indent=2)
    print(f'[theoretical] {out}: T_target_epoch={per_epoch_ms:.1f}ms')
"
    fi
    verify_ttarget "$TTARGET_A" || { echo "[ERR] missing ttarget_a"; return 1; }
    verify_ttarget "$TTARGET_B" || { echo "[ERR] missing ttarget_b"; return 1; }

    # Sync T_target files to 226 (filesystem not shared)
    for TF in "$TTARGET_A" "$TTARGET_B"; do
        scp -o ConnectTimeout=10 "$TF" "$NODE_226_SSH:$TF" 2>/dev/null || true
    done

    # Cleanup orphans + stale CSVs
    cleanup_jobs
    cleanup_bg
    rm -f "$P4_DIR"/p4_job{A,B}_reverse_longliu_rank{0,1}_{iter,epoch}.csv 2>/dev/null || true
    ssh -o ConnectTimeout=10 "$NODE_226_SSH" \
        "rm -f $P4_DIR/p4_job{A,B}_reverse_longliu_rank{0,1}_{iter,epoch}.csv 2>/dev/null" 2>/dev/null || true
    sleep 2

    # 3-min warmup (ENVIRONMENT DRIFT mitigation)
    echo "[warmup] 3 min idle to stabilize NIC/system state..."
    sleep 180

    # Start bg flow if needed
    if [ "$BG" = "true" ]; then
        echo "[bg] starting iperf3 background flow..."
        start_bg_servers
        start_bg_clients
    fi

    # Save scenario config snapshot
    cp "$SCRIPTS_DIR/expA_scenarios.json" "$SCEN_DIR/scenarios_snapshot.json"
    md5sum "$SCRIPTS_DIR/expA_scenarios.json" > "$SCEN_DIR/scenarios.md5"

    # ---- Common args ----
    local COMMON_ARGS="--phase main --payload-mb $PAYLOAD \
        --ci-phase1 $CI --ci-phase2 $CI \
        --reverse-epoch $REVERSE_EPOCH_OFF \
        --num-iters $NUM_ITERS --iters-per-epoch $ITERS_PER_EPOCH \
        --sleep-us $SLEEP_US --initial-priority 3"

    # ============================================================
    # Launch FIRST job (Job A) — staggered startup pattern
    # ============================================================
    echo "[launch] Job A first (port $PORT_JA)..."
    REMOTE_LOG_A="/tmp/expA_${SID}_jobA_226.log"
    REMOTE_LOG_B="/tmp/expA_${SID}_jobB_226.log"

    # Job A rank 1 on 226
    ssh -o ConnectTimeout=30 "$NODE_226_SSH" "cd $P4_DIR && \
        env CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JA \
        WORLD_SIZE=2 RANK=1 MULTI_COMM_PORT=$PORT_JA \
        $NCCL_ENV_COMMON $NCCL_ENV_226 \
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL:\$LD_LIBRARY_PATH \
        PYTHONPATH=$PYTHONPATH_SLO:\$PYTHONPATH \
        timeout 900 python3 -u $JOB_SCRIPT --job A --mode longliu \
            --ttarget-file $TTARGET_A $COMMON_ARGS \
            > $REMOTE_LOG_A 2>&1" \
        > "$SCEN_DIR/jobA_226_runner.log" 2>&1 &
    local JA_226_PID=$!
    echo "  Job A rank 1 on 226 (pid=$JA_226_PID), waiting 5s..."
    sleep 5

    # Job A rank 0 on 10.1
    cd "$P4_DIR"
    env CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JA \
    WORLD_SIZE=2 RANK=0 MULTI_COMM_PORT=$PORT_JA \
    $NCCL_ENV_COMMON $NCCL_ENV_10 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL \
    PYTHONPATH=$PYTHONPATH_SLO \
    timeout 900 python3 -u "$JOB_SCRIPT" --job A --mode longliu \
        --ttarget-file "$TTARGET_A" $COMMON_ARGS \
        > "$SCEN_DIR/jobA_10.log" 2>&1 &
    local JA_10_PID=$!

    echo "  Job A launched. Waiting 15s for NCCL init..."
    sleep 15

    # ============================================================
    # Launch SECOND job (Job B) — after first job init complete
    # ============================================================
    echo "[launch] Job B (port $PORT_JB)..."

    # Job B rank 1 on 226
    ssh -o ConnectTimeout=30 "$NODE_226_SSH" "cd $P4_DIR && \
        env CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JB \
        WORLD_SIZE=2 RANK=1 MULTI_COMM_PORT=$PORT_JB \
        $NCCL_ENV_COMMON $NCCL_ENV_226 \
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL:\$LD_LIBRARY_PATH \
        PYTHONPATH=$PYTHONPATH_SLO:\$PYTHONPATH \
        timeout 900 python3 -u $JOB_SCRIPT --job B --mode longliu \
            --ttarget-file $TTARGET_B $COMMON_ARGS \
            > $REMOTE_LOG_B 2>&1" \
        > "$SCEN_DIR/jobB_226_runner.log" 2>&1 &
    local JB_226_PID=$!
    echo "  Job B rank 1 on 226 (pid=$JB_226_PID), waiting 5s..."
    sleep 5

    # Job B rank 0 on 10.1
    env CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JB \
    WORLD_SIZE=2 RANK=0 MULTI_COMM_PORT=$PORT_JB \
    $NCCL_ENV_COMMON $NCCL_ENV_10 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH_NCCL \
    PYTHONPATH=$PYTHONPATH_SLO \
    timeout 900 python3 -u "$JOB_SCRIPT" --job B --mode longliu \
        --ttarget-file "$TTARGET_B" $COMMON_ARGS \
        > "$SCEN_DIR/jobB_10.log" 2>&1 &
    local JB_10_PID=$!

    echo "[launch] Both jobs running. BG=$BG"
    echo "  PIDs: JA_10=$JA_10_PID JB_10=$JB_10_PID JA_226=$JA_226_PID JB_226=$JB_226_PID"

    # ---- Wait for completion ----
    wait "$JA_10_PID";  local RC_JA=$?
    wait "$JB_10_PID";  local RC_JB=$?
    wait "$JA_226_PID" 2>/dev/null || true
    wait "$JB_226_PID" 2>/dev/null || true

    echo "[done] $SID: RC_JA=$RC_JA RC_JB=$RC_JB"

    # Cleanup
    cleanup_jobs
    if [ "$BG" = "true" ]; then
        cleanup_bg
    fi

    # ---- Archive CSVs ----
    for J in A B; do
        for SUFFIX in rank0_iter rank0_epoch; do
            F="${P4_DIR}/p4_job${J}_reverse_longliu_${SUFFIX}.csv"
            [ -f "$F" ] && cp "$F" "$SCEN_DIR/job${J}_${SUFFIX}.csv" 2>/dev/null || true
        done
    done
    # scp rank1 logs from 226
    scp -o ConnectTimeout=10 "$NODE_226_SSH:$REMOTE_LOG_A" "$SCEN_DIR/jobA_226.log" 2>/dev/null || true
    scp -o ConnectTimeout=10 "$NODE_226_SSH:$REMOTE_LOG_B" "$SCEN_DIR/jobB_226.log" 2>/dev/null || true

    # Manifest
    cat > "$SCEN_DIR/manifest.txt" <<EOF
scenario_id=$SID
label=$LABEL
payload_mb=$PAYLOAD
c_i=$CI
bg_flow=$BG
ttarget_a=$TTARGET_A
ttarget_b=$TTARGET_B
run_ts=$RUN_TS
rc_ja=$RC_JA
rc_jb=$RC_JB
EOF

    # Quick sanity: check if epoch CSV has data
    for J in A B; do
        local CSV="$SCEN_DIR/job${J}_rank0_epoch.csv"
        if [ -f "$CSV" ]; then
            local NROWS=$(wc -l < "$CSV")
            echo "[archive] job${J}_rank0_epoch.csv: ${NROWS} lines"
        else
            echo "[WARN] job${J}_rank0_epoch.csv MISSING — check logs"
        fi
    done
    echo "[archived] $SCEN_DIR"
    return 0
}

# ---- Scenario dispatch ----
run_S1() { run_scenario S1 "2job_50G_ci1.2" 2 false $PAYLOAD_50G $CI_TIGHT \
    /tmp/ttarget_v5_jobA.json /tmp/ttarget_v5_jobB.json; }
run_S2() { run_scenario S2 "2job_50G_ci1.5" 2 false $PAYLOAD_50G $CI_LOOSE \
    /tmp/ttarget_v5_jobA.json /tmp/ttarget_v5_jobB.json; }
run_S3() { run_scenario S3 "2job_bg_44G_ci1.2" 2 true $PAYLOAD_50G $CI_TIGHT \
    /tmp/ttarget_v5_jobA.json /tmp/ttarget_v5_jobB.json; }
run_S4() { run_scenario S4 "2job_bg_44G_ci1.5" 2 true $PAYLOAD_50G $CI_LOOSE \
    /tmp/ttarget_v5_jobA.json /tmp/ttarget_v5_jobB.json; }
run_S5() { run_scenario S5 "2job_25G_ci1.2_HOLDOUT" 2 false $PAYLOAD_25G $CI_TIGHT \
    THEORETICAL THEORETICAL; }
run_S6() { run_scenario S6 "2job_35G_ci1.5" 2 false $PAYLOAD_35G $CI_LOOSE \
    /tmp/ttarget_expA_jobA_768.json /tmp/ttarget_expA_jobB_768.json; }

# ---- Main ----
case "$SCENARIO_FILTER" in
    ALL)
        for SID in S1 S2 S3 S4 S5 S6; do
            run_$SID || { echo "[ERR] $SID failed, continuing..."; }
        done
        ;;
    S1|S2|S3|S4|S5|S6)
        run_$SCENARIO_FILTER || { echo "[ERR] $SCENARIO_FILTER failed"; exit 1; }
        ;;
    *)
        echo "Usage: $0 [ALL|S1|S2|S3|S4|S5|S6]"
        exit 1
        ;;
esac

echo ""
echo "[run] all scenarios complete. data: $RUN_DIR"
echo "$RUN_DIR" > "$DATA_DIR/latest_run.txt"
