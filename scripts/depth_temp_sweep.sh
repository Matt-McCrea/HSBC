#!/bin/bash
# depth_temp_sweep.sh — test whether a depth-channel temperature (kappa) unfreezes the
# cheap deterministic sampler (DDIM_10 eta=0) by restoring the marketable-order tail.
#
# Mechanism: training clamped depth >= 0, so the model piles depth output near 0 (passive).
# Only sampling variance spilling below 0 makes orders marketable (cross spread -> execute ->
# move price). Deterministic few-step DDIM doesn't spill -> freeze. Scaling the decoded depth
# z-score by kappa>1 widens the distribution so a fraction crosses into negative depth,
# reproducing DDPM's spillover on a 10-step deterministic sampler. If some kappa gives
# DDPM-like movement (unique mids >> 6, exec ~7%, depth_pre_drop neg>0) at 10 steps, that's a
# fast working sampler AND confirmation of the depth-spillover mechanism.
#
# Run AFTER night_run finishes (shares the GPU). Training is OFF by default here (--do-train
# to launch it after the sweep).
#
# Usage:
#   bash scripts/depth_temp_sweep.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv
#
# Results: depth_temp_results/<timestamp>/summary.md

set -uo pipefail

TICKER="INTC"; DATE="20150130"; START="09:30:00"; END="10:00:00"
REAL_PATH=""
CKPT="0.681"
OUT_DIR="depth_temp_results/$(date +%Y%m%d_%H%M%S)"
KAPPAS="1.0 1.5 2.0 2.5 3.0"
DO_TRAIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)     REAL_PATH="$2"; shift 2 ;;
        --id)       CKPT="$2";      shift 2 ;;
        --kappas)   KAPPAS="$2";    shift 2 ;;   # space-separated
        --ticker)   TICKER="$2";    shift 2 ;;
        --date)     DATE="$2";      shift 2 ;;
        --start)    START="$2";     shift 2 ;;
        --end)      END="$2";       shift 2 ;;
        --out-dir)  OUT_DIR="$2";   shift 2 ;;
        --do-train) DO_TRAIN=1;     shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"
echo "# Depth-temp sweep — $(date '+%F %T')" > "$SUMMARY"
echo "Window: $DATE $START-$END | ckpt: $CKPT | real: ${REAL_PATH:-none}" >> "$SUMMARY"
echo "" >> "$SUMMARY"

run_sim () {
    # run_sim <tag> <sampler> <nsteps> <eta> <extra-args>
    local TAG="$1" SOLVER="$2" NSTEPS="$3" ETA="$4" EXTRA="$5"
    local LOG="$OUT_DIR/logs/sim_${TAG}.txt"
    local DONE_MARK="$OUT_DIR/logs/.done_${TAG}"
    if [[ -f "$DONE_MARK" ]]; then echo "  SKIP $TAG (done)"; return; fi
    echo "── $TAG"
    local SENTINEL; SENTINEL=$(mktemp); touch "$SENTINEL"
    local T0; T0=$(date +%s)
    local ARGS=(python ABIDES/abides.py -c world_agent_sim
                -t "$TICKER" -date "$DATE" -st "$START" -et "$END"
                -d True -type "$SOLVER" -nsteps "$NSTEPS" -eta "$ETA" -id "$CKPT")
    # shellcheck disable=SC2206
    [[ -n "$EXTRA" ]] && ARGS+=($EXTRA)
    echo "   ${ARGS[*]}"
    if ! "${ARGS[@]}" > "$LOG" 2>&1; then
        echo "  ERROR — see $LOG"; echo "## $TAG — SIM ERROR (see logs/sim_${TAG}.txt)" >> "$SUMMARY"
        rm -f "$SENTINEL"; return
    fi
    local SECS=$(( $(date +%s) - T0 ))
    local GEN_CSV
    GEN_CSV=$(find ABIDES/log -name "processed_orders.csv" -newer "$SENTINEL" ! -path "*/paper/*" 2>/dev/null | sort | tail -1)
    rm -f "$SENTINEL"
    if [[ -z "$GEN_CSV" ]]; then echo "  ERROR: no CSV"; echo "## $TAG — NO CSV" >> "$SUMMARY"; return; fi
    {
        echo "## $TAG  (${SECS}s)"
        echo '```'
        if [[ -n "$REAL_PATH" ]]; then
            python -m evaluation.quantitative_eval.flow_mix --real "$REAL_PATH" --gen "$GEN_CSV" 2>&1
        else
            python -m evaluation.quantitative_eval.flow_mix --gen "$GEN_CSV" 2>&1
        fi
        echo ""
        sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
        echo '```'; echo ""
    } >> "$SUMMARY"
    touch "$DONE_MARK"
    echo "  done in ${SECS}s"
}

# ── depth-temp sweep on DDIM_10 eta=0 ─────────────────────────────────────────
echo "# DDIM_10 eta=0 across depth-temp kappa" >> "$SUMMARY"
for K in $KAPPAS; do
    EXTRA=""
    [[ "$K" != "1.0" ]] && EXTRA="--depth-temp $K"
    run_sim "DDIM10_eta0_dtemp${K}" DDIM 10 0.0 "$EXTRA"
done

# ── best-kappa candidate combined with prior type-decode (controls the cancel/type mix) ──
echo "# DDIM_10 eta=0, depth-temp 2.0 + prior decode" >> "$SUMMARY"
run_sim "DDIM10_eta0_dtemp2.0_prior" DDIM 10 0.0 "--depth-temp 2.0 --type-decode prior"

# ── reference: DDPM_100 (the working stochastic baseline) ─────────────────────
echo "# DDPM_100 reference" >> "$SUMMARY"
run_sim "DDPM100_ref" DDPM 100 0.0 ""

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary: $SUMMARY"
echo "══════════════════════════════════════════"
echo "Look for the kappa where DDIM_10 crosses from frozen (unique mids ~6, depth neg=0)"
echo "to DDPM-like (mids in the dozens, depth neg>0, exec climbing toward real ~7%),"
echo "WITHOUT tipping into runaway drift (mid range blowing past ~0.4)."

if [[ "$DO_TRAIN" -eq 1 ]]; then
    echo ""; echo "Launching CONDITIONAL_DROPOUT=0.1 training in background..."
    nohup python main.py > /tmp/training_depthtemp.log 2>&1 &
    echo "  training PID: $!  | log: /tmp/training_depthtemp.log"
fi
