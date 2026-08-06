#!/bin/bash
# ============================================================================
# Experiment C v2 — Master Runner
# ============================================================================
# Calibrates and runs all regimes for v2 experiments.
# For each regime: calibrate → run 3 arms × 5 rounds.
# Skips runs that already have data (for resume after interruption).
#
# Usage: bash run_all_v2.sh [scenario]
#   scenario: S1 | S2 | all (default: all)
#
# Total: S1 (4 regimes × 3 arms × 5 rounds = 60) + S2 (1 × 3 × 5 = 15) = 75 runs
# Estimated time: ~3-4 hours
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_C_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$EXP_C_DIR/data_v2"

SCENARIO_FILTER=${1:-all}

# Define the full schedule
declare -a SCHEDULE
# S1 regimes
for regime in S1_ample S1_moderate S1_deep S1_very_deep; do
    for arm in longliu static fair; do
        for round in $(seq 1 5); do
            SCHEDULE+=("S1|$regime|$arm|$round")
        done
    done
done
# S2 regimes
for arm in longliu static fair; do
    for round in $(seq 1 5); do
        SCHEDULE+=("S2|S2_starvation|$arm|$round")
    done
done

echo "================================================================"
echo "Experiment C v2 — Master Runner"
echo "  Total runs: ${#SCHEDULE[@]}"
echo "  Scenario filter: $SCENARIO_FILTER"
echo "  Start time: $(date -Iseconds)"
echo "================================================================"

RUN=0
SKIPPED=0
FAILED=0
CALIBRATED_REGIMES=""

for entry in "${SCHEDULE[@]}"; do
    IFS='|' read -r scenario regime arm round <<< "$entry"

    # Apply scenario filter
    if [[ "$SCENARIO_FILTER" != "all" && "$SCENARIO_FILTER" != "$scenario" ]]; then
        continue
    fi

    # Skip if run already exists (resume support)
    EXISTING=$(ls -d "$DATA_DIR/${regime}_${arm}_r${round}_"* 2>/dev/null | head -1)
    if [[ -n "$EXISTING" ]]; then
        # Verify data integrity: check for stats files
        if [[ -f "$EXISTING/manifest.txt" ]] && ls "$EXISTING"/job*_stats.csv &>/dev/null; then
            SKIPPED=$((SKIPPED+1))
            continue
        fi
    fi

    # Calibrate once per regime (before its first run)
    if [[ "$CALIBRATED_REGIMES" != *"$regime"* ]]; then
        echo ""
        echo "============================================"
        echo "[calibrate] $scenario/$regime"
        echo "============================================"
        bash "$SCRIPT_DIR/calib_solo_v2.sh" "$scenario" "$regime" 2>&1 | grep -E '^\[ok\]|^\[ERR\]|Summary|Σb̄'
        CALIBRATED_REGIMES="$CALIBRATED_REGIMES $regime"
        echo "[warmup] 30s cooldown after calibration..."
        sleep 30
    fi

    echo ""
    echo "============================================"
    echo "[run $((RUN+1))/${#SCHEDULE[@]}] $scenario/$regime/$arm/r$round"
    echo "  Time: $(date -Iseconds)"
    echo "============================================"

    bash "$SCRIPT_DIR/run_expC_v2.sh" "$scenario" "$regime" "$arm" "$round" 2>&1 | tail -5
    RC=$?

    if [[ $RC -eq 0 ]]; then
        RUN=$((RUN+1))
        echo "[ok] Run $RUN completed successfully"
    else
        FAILED=$((FAILED+1))
        echo "[FAIL] Run failed with RC=$RC"
    fi

    # Brief cooldown between runs
    sleep 10
done

echo ""
echo "================================================================"
echo "Experiment C v2 — All runs complete"
echo "  Successful: $RUN"
echo "  Skipped (already exists): $SKIPPED"
echo "  Failed: $FAILED"
echo "  End time: $(date -Iseconds)"
echo "================================================================"

# Run analysis
echo ""
echo "[analyze] Running v2 analysis..."
python3 "$EXP_C_DIR/analysis/analyze_expC_v2.py" \
    --data-dir "$DATA_DIR" \
    --scenarios "$EXP_C_DIR/scenarios/scenarios_v2.json" \
    --output "$EXP_C_DIR/analysis" 2>&1

echo ""
echo "[done] Analysis complete. See $EXP_C_DIR/analysis/expC_v2_analysis.md"
