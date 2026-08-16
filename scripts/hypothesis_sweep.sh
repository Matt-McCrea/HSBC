#!/bin/bash
# hypothesis_sweep.sh — staged hypothesis testing for the frozen-mid-price problem.
#
# Runs stages in order of information-per-minute; each stage's results are conclusive
# on their own, so the sweep can be stopped after any stage:
#
#   Stage 0 — open-loop attribution (minutes each): sample the model on REAL test-set
#             conditioning with different samplers. If DDIM_10 eta=0 collapses HERE,
#             the problem is model x sampler, not the simulator.
#   Stage A — closed-loop sampler tests (H0: few-step deterministic sampling collapses
#             to the conditional mean -> depth-0 stacking -> walls -> frozen mid).
#             KEY TEST: DDIM_10 with eta=1.0 (stochastic) vs eta=0.0 (deterministic).
#   Stage B — code-fix isolation on the frozen config (DDIM_10 eta=0): each simulator
#             fix flag alone, then all together.
#   Stage C — best-of confirmation: DDPM_100 + all fixes (fidelity), Stage A winner + fixes.
#
# Simulation window is 09:30-10:00 (15 min replay seed + 15 min generation) for speed.
#
# Usage (from repo root on the GPU machine):
#   bash scripts/hypothesis_sweep.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-30-00_30/processed_orders.csv
#
# Results in hypothesis_results/<timestamp>/: per-run logs, flow-mix reports, summary.md.

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TICKER="INTC"
DATE="20150130"
START="09:30:00"
END="10:00:00"
REAL_PATH=""
CHECKPOINT_ID=""
OUT_DIR="hypothesis_results/$(date +%Y%m%d_%H%M%S)"
STAGES="0ABC"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)    REAL_PATH="$2";      shift 2 ;;
        --ticker)  TICKER="$2";         shift 2 ;;
        --date)    DATE="$2";           shift 2 ;;
        --start)   START="$2";          shift 2 ;;
        --end)     END="$2";            shift 2 ;;
        --id)      CHECKPOINT_ID="$2";  shift 2 ;;
        --out-dir) OUT_DIR="$2";        shift 2 ;;
        --stages)  STAGES="$2";         shift 2 ;;   # e.g. --stages 0A to run only stage 0 and A
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/open_loop"
SUMMARY="$OUT_DIR/summary.md"

echo "# Hypothesis sweep — $(date '+%F %T')" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Window: $DATE $START-$END | real: ${REAL_PATH:-none}" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# Record the normalization stats actually in use (verifies remote-vs-constants question)
echo "## normalization_stats.json" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
cat "data/$TICKER/normalization_stats.json" >> "$SUMMARY" 2>&1 || echo "(not found — slow-path recompute in use)" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
echo "" >> "$SUMMARY"

# ── Helpers ───────────────────────────────────────────────────────────────────

run_open_loop () {
    # run_open_loop <tag> <sampler> <nsteps> <eta>
    local TAG="$1" SOLVER="$2" NSTEPS="$3" ETA="$4"
    local LOG="$OUT_DIR/logs/openloop_${TAG}.txt"
    local JSON="$OUT_DIR/open_loop/${TAG}.json"
    if [[ -f "$JSON" ]]; then echo "  SKIP open-loop $TAG (exists)"; return; fi
    echo "── open-loop: $TAG"
    local ARGS=(python evaluation/diagnostics/open_loop_eval.py
                --type "$SOLVER" --nsteps "$NSTEPS" --eta "$ETA" --out "$JSON")
    [[ -n "$CHECKPOINT_ID" ]] && ARGS+=(--id "$CHECKPOINT_ID")
    if "${ARGS[@]}" > "$LOG" 2>&1; then
        {
            echo "### open-loop $TAG"
            echo '```'
            sed -n '/=== OPEN-LOOP RESULTS/,$p' "$LOG"
            echo '```'
            echo ""
        } >> "$SUMMARY"
    else
        echo "  ERROR — see $LOG"
        echo "### open-loop $TAG — ERROR (see logs/openloop_${TAG}.txt)" >> "$SUMMARY"
    fi
}

run_sim () {
    # run_sim <tag> <sampler> <nsteps> <eta> [extra sim args...]
    local TAG="$1" SOLVER="$2" NSTEPS="$3" ETA="$4"; shift 4
    local LOG="$OUT_DIR/logs/sim_${TAG}.txt"
    local DONE_MARK="$OUT_DIR/logs/.done_${TAG}"
    if [[ -f "$DONE_MARK" ]]; then echo "  SKIP sim $TAG (done)"; return; fi
    echo "── sim: $TAG"

    local SENTINEL; SENTINEL=$(mktemp); touch "$SENTINEL"
    local T0; T0=$(date +%s)
    local ARGS=(python ABIDES/abides.py -c world_agent_sim
                -t "$TICKER" -date "$DATE" -st "$START" -et "$END"
                -d True -type "$SOLVER" -nsteps "$NSTEPS" -eta "$ETA")
    [[ "$SOLVER" == HYBRID_* ]] && ARGS+=(--tail-steps 2)
    [[ -n "$CHECKPOINT_ID" ]] && ARGS+=(-id "$CHECKPOINT_ID")
    ARGS+=("$@")

    echo "   ${ARGS[*]}"
    if ! "${ARGS[@]}" > "$LOG" 2>&1; then
        echo "  ERROR — see $LOG"
        echo "## $TAG — SIM ERROR (see logs/sim_${TAG}.txt)" >> "$SUMMARY"
        rm -f "$SENTINEL"
        return
    fi
    local SECS=$(( $(date +%s) - T0 ))

    local GEN_CSV
    GEN_CSV=$(find ABIDES/log -name "processed_orders.csv" -newer "$SENTINEL" \
                  ! -path "*/paper/*" 2>/dev/null | sort | tail -1)
    rm -f "$SENTINEL"
    if [[ -z "$GEN_CSV" ]]; then
        echo "  ERROR: no generated CSV found"
        echo "## $TAG — NO CSV" >> "$SUMMARY"
        return
    fi

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

# ── Stage 0: open-loop attribution ───────────────────────────────────────────
if [[ "$STAGES" == *0* ]]; then
    echo ""
    echo "════════ STAGE 0: open-loop attribution ════════"
    echo "# Stage 0 — open-loop attribution" >> "$SUMMARY"
    run_open_loop "DDPM_100"       DDPM 100 0.0
    run_open_loop "DDIM_10_eta0"   DDIM 10  0.0
    run_open_loop "DDIM_10_eta1"   DDIM 10  1.0
    run_open_loop "DDIM_100_eta0"  DDIM 100 0.0
fi

# ── Stage A: closed-loop sampler tests (H0) ──────────────────────────────────
if [[ "$STAGES" == *A* ]]; then
    echo ""
    echo "════════ STAGE A: H0 sampler-collapse tests (no fixes) ════════"
    echo "# Stage A — sampler tests, no fixes" >> "$SUMMARY"
    run_sim "A_DDIM_10_eta0"      DDIM 10  0.0     # negative control (should freeze)
    run_sim "A_DDIM_10_eta1"      DDIM 10  1.0     # KEY TEST: stochasticity at low NFE
    run_sim "A_DDIM_20_eta1"      DDIM 20  1.0
    run_sim "A_DDIM_100_eta0"     DDIM 100 0.0     # full-step deterministic
    run_sim "A_HYBRID_PP_DDPM_10" HYBRID_PP_DDPM 10 0.0   # acceleration candidate
    run_sim "A_DDPM_100"          DDPM 100 0.0     # positive control (should move)
fi

# ── Stage B: code-fix isolation on the frozen config ─────────────────────────
if [[ "$STAGES" == *B* ]]; then
    echo ""
    echo "════════ STAGE B: fix isolation on DDIM_10 eta=0 ════════"
    echo "# Stage B — fix isolation (DDIM_10 eta=0)" >> "$SUMMARY"
    run_sim "B_fix_time"        DDIM 10 0.0 --fix-time
    run_sim "B_fix_cancel_bind" DDIM 10 0.0 --fix-cancel-bind
    run_sim "B_type_decode_l2"  DDIM 10 0.0 --type-decode l2
    run_sim "B_fix_lob_pad"     DDIM 10 0.0 --fix-lob-pad
    run_sim "B_drop_type2"      DDIM 10 0.0 --drop-type2-cond
    run_sim "B_ALL_fixes"       DDIM 10 0.0 --fix-time --fix-cancel-bind --type-decode l2 --fix-lob-pad --drop-type2-cond
fi

# ── Stage C: best-of confirmation ─────────────────────────────────────────────
if [[ "$STAGES" == *C* ]]; then
    echo ""
    echo "════════ STAGE C: confirmation ════════"
    echo "# Stage C — confirmation" >> "$SUMMARY"
    run_sim "C_DDPM_100_ALL_fixes"  DDPM 100 0.0 --fix-time --fix-cancel-bind --fix-lob-pad --drop-type2-cond
    run_sim "C_DDIM_10_eta1_ALL"    DDIM 10  1.0 --fix-time --fix-cancel-bind --fix-lob-pad --drop-type2-cond
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  Sweep complete. Summary: $SUMMARY"
echo "══════════════════════════════════════════════════"
echo ""
echo "What to look for:"
echo "  - Stage 0: does DDIM_10 eta=0 open-loop show depth-0 collapse + limit bias vs REAL?"
echo "    If yes -> sampler collapse confirmed independent of the simulator."
echo "  - Stage A: does eta=1.0 unfreeze DDIM_10 (unique mids >> 6, lean book)?"
echo "  - Stage B: does any single simulator fix unfreeze DDIM_10 eta=0?"
echo "  - DIAG decoded_pre_drop vs placed: attributes type imbalance to model vs drops."
