#!/bin/bash
# ============================================================================
# Experiment B: Hardware Tier Swap — Full Multi-Round Orchestrator
# ============================================================================
# Runs all 4 rounds with alternation:
#   Round 1: A→B,  LL→CX
#   Round 2: A→B,  CX→LL
#   Round 3: B→A,  LL→CX
#   Round 4: B→A,  CX→LL
#
# Usage: bash run_expB_all.sh [skip_bg=0] [start_round=1]
#   skip_bg:     1 = skip background iperf3 flow
#   start_round: resume from this round (1-4)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/expB_config.sh"

SKIP_BG=${1:-0}
START_ROUND=${2:-1}

echo "================================================================"
echo "Experiment B — Full Multi-Round Orchestrator"
echo "  Rounds    : 4 (alternating job order × arm order)"
echo "  Skip BG   : $SKIP_BG"
echo "  Start from: Round $START_ROUND"
echo "  Est. time : ~60-80 min (4 rounds × 2 arms × ~8 min + warmups)"
echo "  Date      : $(date -Iseconds)"
echo "================================================================"

# Verify prerequisites
verify_ttarget || exit 1

# Run each round
for ROUND in $(seq "$START_ROUND" 4); do
    echo ""
    echo "########################################################"
    echo "#                                                      #"
    echo "#  Starting Round $ROUND / 4                            #"
    echo "#                                                      #"
    echo "########################################################"
    
    bash "$SCRIPT_DIR/run_expB_round.sh" "$ROUND" "$SKIP_BG"
    
    if [[ "$ROUND" -lt 4 ]]; then
        echo ""
        echo ">>> Inter-round cooldown: 180s (NIC state stabilization)"
        sleep 180
    fi
done

echo ""
echo "================================================================"
echo "All 4 rounds complete!"
echo "================================================================"
echo "Data directories:"
for ROUND in 1 2 3 4; do
    JOB_ORD="${ROUND_JOB_ORDER[$ROUND]}"
    ARM_ORD="${ROUND_ARM_ORDER[$ROUND]}"
    LABEL="round${ROUND}_${JOB_ORD}_${ARM_ORD}"
    echo "  Round $ROUND ($LABEL): ${EXP_B_DIR}/data/${LABEL}/"
done
echo ""
echo "Next step: run analysis"
echo "  python3 ${SCRIPT_DIR}/analyze_expB.py"
echo "================================================================"
