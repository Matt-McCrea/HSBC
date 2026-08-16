#!/bin/bash
# decode_eta_test.sh — Focused test of the two-axis diagnosis of the frozen/drifting mid-price,
# then launches overnight training.
#
# Findings that motivate this matrix (from hypothesis_results.md):
#   Axis 1 (freeze): DDIM-10 collapses order DEPTH to the best quote (68% at depth 0)
#     regardless of eta; DDPM-100 spreads it (14% at depth 0). Looks step-count driven.
#   Axis 2 (drift):  adding sampler variance (eta=1) blows MARKET-order decode from ~3%
#     to ~24% via the oversized MARKET anchor region, and that flood of executions is what
#     moves eta=1's price (drift, not discovery).
#
# So this matrix asks four concrete questions on the CURRENT checkpoint (val_ema=0.681):
#   Q1  Does prior-corrected decode fix the eta=1 market blowup / drift?   (eta1 default vs prior)
#   Q2  Does a MODERATE eta unfreeze without prior's help?                 (eta0.4 default)
#   Q3  Does moderate eta + prior decode give a clean moving market?       (eta0.4 prior)  <- the bet
#   Q4  Is the depth collapse purely step-count?  DDIM-100 deterministic   (the missing experiment,
#       now runnable after the nsteps off-by-one fix). If it spreads depth like DDPM-100,
#       Axis 1 is steps, not stochasticity.
# Plus controls: DDIM-10 eta0 (frozen baseline), DDPM-100 (working reference, default & prior).
#
# Default runs one checkpoint (~4-5h) so training gets the rest of the night. Add the
# more-trained epoch4 checkpoint with --checkpoints "0.681 0.719" if you'd rather spend
# the GPU on that than on training (it will roughly double the sweep time).
#
# Usage:
#   bash scripts/decode_eta_test.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv
#
# Results: decode_eta_results/<timestamp>/summary.md   Training log: /tmp/training_after_test.log

set -uo pipefail

TICKER="INTC"; DATE="20150130"; START="09:30:00"; END="10:00:00"
REAL_PATH=""
CHECKPOINTS="0.681"
OUT_DIR="decode_eta_results/$(date +%Y%m%d_%H%M%S)"
SKIP_TRAIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)        REAL_PATH="$2";    shift 2 ;;
        --checkpoints) CHECKPOINTS="$2";  shift 2 ;;   # space-separated val_ema ids
        --ticker)      TICKER="$2";       shift 2 ;;
        --date)        DATE="$2";         shift 2 ;;
        --start)       START="$2";        shift 2 ;;
        --end)         END="$2";          shift 2 ;;
        --out-dir)     OUT_DIR="$2";      shift 2 ;;
        --skip-train)  SKIP_TRAIN=1;      shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"
echo "# Decode x eta focused test — $(date '+%F %T')" > "$SUMMARY"
echo "Window: $DATE $START-$END | real: ${REAL_PATH:-none} | checkpoints: $CHECKPOINTS" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# Config matrix: "tag|sampler|nsteps|eta|extra-args"
CONFIGS=(
    "DDIM10_eta0|DDIM|10|0.0|"                                  # frozen control
    "DDIM10_eta1_default|DDIM|10|1.0|"                          # drift control (Q1 baseline)
    "DDIM10_eta1_prior|DDIM|10|1.0|--type-decode prior"         # Q1: prior fixes drift?
    "DDIM10_eta04_default|DDIM|10|0.4|"                         # Q2: moderate eta unfreezes?
    "DDIM10_eta04_prior|DDIM|10|0.4|--type-decode prior"        # Q3: the bet
    "DDIM100_eta0|DDIM|100|0.0|"                                # Q4: many-step deterministic
    "DDPM100_default|DDPM|100|0.0|"                             # working reference
    "DDPM100_prior|DDPM|100|0.0|--type-decode prior"           # prior on the good baseline
)

run_sim () {
    local TAG="$1" CKPT="$2" SOLVER="$3" NSTEPS="$4" ETA="$5" EXTRA="$6"
    local LOG="$OUT_DIR/logs/sim_${TAG}.txt"
    local DONE_MARK="$OUT_DIR/logs/.done_${TAG}"
    if [[ -f "$DONE_MARK" ]]; then echo "  SKIP $TAG (done)"; return; fi
    echo "── $TAG"

    local SENTINEL; SENTINEL=$(mktemp); touch "$SENTINEL"
    local T0; T0=$(date +%s)
    local ARGS=(python ABIDES/abides.py -c world_agent_sim
                -t "$TICKER" -date "$DATE" -st "$START" -et "$END"
                -d True -type "$SOLVER" -nsteps "$NSTEPS" -eta "$ETA" -id "$CKPT")
    [[ "$SOLVER" == HYBRID_* ]] && ARGS+=(--tail-steps 2)
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
        echo '```'
        echo ""
    } >> "$SUMMARY"
    touch "$DONE_MARK"
    echo "  done in ${SECS}s"
}

for CKPT in $CHECKPOINTS; do
    echo ""; echo "════════ checkpoint val_ema=$CKPT ════════"
    echo "# checkpoint val_ema=$CKPT" >> "$SUMMARY"
    for row in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG SOLVER NSTEPS ETA EXTRA <<< "$row"
        run_sim "ck${CKPT}_${TAG}" "$CKPT" "$SOLVER" "$NSTEPS" "$ETA" "$EXTRA"
    done
done

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary: $SUMMARY"
echo "══════════════════════════════════════════"
echo "Read as: Q1 eta1_prior should cut market-decode & stop the +11% drift;"
echo "         Q3 eta04_prior is the target (moves, lean-ish book, market~3%);"
echo "         Q4 DDIM100_eta0 depth spread vs DDPM100 answers 'is freeze just step count?'."

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo ""; echo "Launching overnight training (CONDITIONAL_DROPOUT=0.1 is the default)..."
    echo "  NB: configuration.py has IS_DATA_PREPROCESSED=False -> reruns preprocessing each"
    echo "  launch. If data/INTC/train.npy exists, set it True locally first to skip that."
    nohup python main.py > /tmp/training_after_test.log 2>&1 &
    echo "  training PID: $!  | log: /tmp/training_after_test.log"
else
    echo ""; echo "--skip-train set. Launch training yourself: nohup python main.py > /tmp/training.log 2>&1 &"
fi
