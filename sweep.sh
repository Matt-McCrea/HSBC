#!/bin/bash
# sweep.sh — Run all solver × NFE combinations and evaluate each automatically.
#
# Start it and leave it running. For each config it:
#   1. Runs the ABIDES/TRADES simulation (generates orders)
#   2. Runs sweep_eval.py (KL/JS + all 4 plot types)
#   3. Appends a row to sweep_results/<run>/summary.csv
#
# Usage
# -----
#   source /scratch0/mmccrea/env.sh          # activate venv, cd into repo
#   bash sweep.sh \
#       --real ABIDES/log/paper/market_replay_INTC_2015-01-30_16-00-00/processed_orders.csv \
#       [--ticker INTC] [--date 2015-01-30] [--start 09:30:00] [--end 16:00:00] \
#       [--id <checkpoint_val_loss>] [--out-dir sweep_results/run1]
#
# If --real is omitted, simulations still run and plots are still saved but
# KL/JS divergence will be skipped and comparisons show only the generated data.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TICKER="INTC"
DATE="2015-01-30"
START="09:30:00"
END="16:00:00"
REAL_PATH=""
CHECKPOINT_ID=""
OUT_DIR="sweep_results/$(date +%Y%m%d_%H%M%S)"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)      REAL_PATH="$2";      shift 2 ;;
        --ticker)    TICKER="$2";         shift 2 ;;
        --date)      DATE="$2";           shift 2 ;;
        --start)     START="$2";          shift 2 ;;
        --end)       END="$2";            shift 2 ;;
        --id)        CHECKPOINT_ID="$2";  shift 2 ;;
        --out-dir)   OUT_DIR="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs"

# ── Solver × NFE grid ─────────────────────────────────────────────────────────
# Format: "SOLVER:NSTEPS"
# NFE=4 configs removed (degenerate frozen-price behaviour across all families).
# UNIPC removed (degenerate at all step counts tested).
# HYBRID_PP_DDIM: total NSTEPS = (NSTEPS-2) DPM-Solver++ steps + 2 deterministic DDIM tail steps.
CONFIGS=(
    # ── DPM-Solver++ (primary fast ODE contribution) ─────────────────────
    "DPM_SOLVER_PP:8"
    "DPM_SOLVER_PP:10"
    "DPM_SOLVER_PP:20"
    # ── Hybrid DPM-Solver++ + DDPM tail (2 stochastic steps appended) ───
    "HYBRID_PP_DDPM:10"    # 8 PP + 2 DDPM
    "HYBRID_PP_DDPM:12"    # 10 PP + 2 DDPM
    "HYBRID_PP_DDPM:22"    # 20 PP + 2 DDPM
    # ── DPM-Solver (ε-prediction baseline) ───────────────────────────────
    "DPM_SOLVER:8"
    "DPM_SOLVER:10"
    "DPM_SOLVER:20"
    # ── Deterministic baselines ───────────────────────────────────────────
    "DDIM:10"
    "DDIM:20"
    "DDPM:100"   # nsteps ignored by DDPM — always runs all 100 diffusion steps
)

# ── Summary CSV ───────────────────────────────────────────────────────────────
SUMMARY="$OUT_DIR/summary.csv"
echo "tag,js_size,js_price,js_time,kl_size,kl_price,kl_time,sim_seconds" > "$SUMMARY"

echo ""
echo "=== TRADES Solver Sweep ==="
echo "Ticker   : $TICKER"
echo "Date     : $DATE $START → $END"
echo "Real CSV : ${REAL_PATH:-'(none — KL/JS will be skipped)'}"
echo "Checkpoint: ${CHECKPOINT_ID:-'(best val loss in checkpoints dir)'}"
echo "Output   : $OUT_DIR"
echo "Configs  : ${#CONFIGS[@]} runs"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
COMPLETED=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    SOLVER="${CONFIG%%:*}"
    NSTEPS="${CONFIG##*:}"
    TAG="${SOLVER}_${NSTEPS}"
    PLOT_DIR="$OUT_DIR/$TAG"
    LOG="$OUT_DIR/logs/${TAG}.txt"

    echo "──────────────────────────────────────────────────"
    echo "  Running: $TAG"
    echo "  → log: $LOG"

    # Skip if already done (allows resume after crash)
    if [[ -f "$PLOT_DIR/${TAG}_summary.json" ]]; then
        echo "  SKIP: already completed (delete $PLOT_DIR to re-run)"
        continue
    fi
    mkdir -p "$PLOT_DIR"

    # -- Build simulation command --
    SIM_ARGS=(
        python ABIDES/abides.py
        -c world_agent_sim
        -t "$TICKER"
        -date "$DATE"
        -st "$START"
        -et "$END"
        -d True
        -type "$SOLVER"
        -nsteps "$NSTEPS"
    )
    [[ -n "$REAL_PATH"       ]] && SIM_ARGS+=(--real-data-path "$REAL_PATH")
    [[ -n "$CHECKPOINT_ID"   ]] && SIM_ARGS+=(-id "$CHECKPOINT_ID")
    [[ "$SOLVER" == "HYBRID_PP_DDIM" || "$SOLVER" == "HYBRID_PP_DDPM" ]] && SIM_ARGS+=(--tail-steps 2)

    # -- Sentinel: find generated CSV by mtime after sim finishes --
    SENTINEL=$(mktemp)
    touch "$SENTINEL"

    SIM_START=$(date +%s)
    echo "  [SIM] ${SIM_ARGS[*]}"
    if "${SIM_ARGS[@]}" 2>&1 | tee "$LOG"; then
        SIM_OK=true
    else
        SIM_OK=false
    fi
    SIM_END=$(date +%s)
    SIM_SECS=$((SIM_END - SIM_START))

    if [[ "$SIM_OK" == false ]]; then
        echo "  ERROR: simulation failed — see $LOG"
        echo "${TAG},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,$SIM_SECS" >> "$SUMMARY"
        FAILED=$((FAILED + 1))
        rm -f "$SENTINEL"
        continue
    fi

    # -- Find the freshest processed_orders.csv written since sentinel --
    GEN_PATH=$(find ABIDES/log -name "processed_orders.csv" -newer "$SENTINEL" \
                   ! -path "*/paper/*" 2>/dev/null | sort | tail -1)
    rm -f "$SENTINEL"

    if [[ -z "$GEN_PATH" ]]; then
        echo "  ERROR: no generated processed_orders.csv found after simulation"
        echo "${TAG},NO_CSV,NO_CSV,NO_CSV,NO_CSV,NO_CSV,NO_CSV,$SIM_SECS" >> "$SUMMARY"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  Generated CSV: $GEN_PATH"
    cp "$GEN_PATH" "$PLOT_DIR/${TAG}_generated_orders.csv"

    # -- Run evaluation --
    EVAL_ARGS=(
        python evaluation/sweep_eval.py
        --gen     "$GEN_PATH"
        --tag     "$TAG"
        --out-dir "$PLOT_DIR"
    )
    [[ -n "$REAL_PATH" ]] && EVAL_ARGS+=(--real "$REAL_PATH")

    echo "  [EVAL] ${EVAL_ARGS[*]}"
    EVAL_OUT=$("${EVAL_ARGS[@]}" 2>&1 | tee -a "$LOG")

    # Extract the SWEEP_ROW line written by sweep_eval.py
    ROW=$(echo "$EVAL_OUT" | grep "^SWEEP_ROW," | tail -1 | sed 's/^SWEEP_ROW,//')
    if [[ -n "$ROW" ]]; then
        # Reformat: "DPM_SOLVER_PP_10,js_size=0.12,..." → "DPM_SOLVER_PP_10,0.12,..."
        CLEAN=$(echo "$ROW" | sed 's/[a-z_]*=//g')
        echo "${CLEAN},$SIM_SECS" >> "$SUMMARY"
    else
        echo "${TAG},EVAL_ERR,EVAL_ERR,EVAL_ERR,EVAL_ERR,EVAL_ERR,EVAL_ERR,$SIM_SECS" >> "$SUMMARY"
    fi

    COMPLETED=$((COMPLETED + 1))
    echo "  Done in ${SIM_SECS}s"
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Sweep complete: $COMPLETED succeeded, $FAILED failed"
echo "  Results: $OUT_DIR"
echo "=================================================="
echo ""
echo "Summary table:"
column -t -s',' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo ""
echo "Plots are in: $OUT_DIR/<TAG>/"
echo "  *_stylized_facts.pdf  — 6-panel stylized facts overlay"
echo "  *_order_type.pdf      — order type distribution"
echo "  *_spread.pdf          — bid-ask spread distribution"
echo "  *_kl.json             — KL / JS divergence numbers"
echo "  *_summary.json        — all metrics for this run"
