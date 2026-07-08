#!/bin/bash
# night_run.sh — three phases, then overnight training:
#   Phase 1: finish the decode x eta matrix on checkpoint 0.719 (the one that hung; the
#            resample-loop cap fix now prevents that). Captures the new depth histogram.
#   Phase 2: early-stochasticity test on checkpoint 0.681 — the new HYBRID_DDPM_PP sampler
#            (stochastic DDPM head, deterministic PP tail) at a few splits, vs the frozen
#            HYBRID_PP_DDPM control. Question: is the marketable (negative-depth) order tail
#            that drives executions/price movement set in the EARLY high-noise steps? If a
#            stochastic head + cheap deterministic tail moves the market at low total NFE,
#            that's a viable fast sampler; if it still freezes like the PP-head/DDPM-tail
#            inverse, the diversity needs many stochastic steps (DDPM is the only answer).
#   Phase 3: launch CONDITIONAL_DROPOUT=0.1 training in the background.
#
# Each sim's flow_mix + diagnostics (incl. new depth_pre_drop and resample exhausted lines)
# append to summary.md. All runs are resumable (.done_ markers). Phases are independently
# skippable: --skip-719 / --skip-step2 / --skip-train.
#
# Usage:
#   bash scripts/night_run.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv
#
# Results: night_run_results/<timestamp>/summary.md   Training log: /tmp/training_night.log

set -uo pipefail

TICKER="INTC"; DATE="20150130"; START="09:30:00"; END="10:00:00"
REAL_PATH=""
OUT_DIR="night_run_results/$(date +%Y%m%d_%H%M%S)"
SKIP_719=0; SKIP_STEP2=0; SKIP_TRAIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real)       REAL_PATH="$2";  shift 2 ;;
        --ticker)     TICKER="$2";     shift 2 ;;
        --date)       DATE="$2";       shift 2 ;;
        --start)      START="$2";      shift 2 ;;
        --end)        END="$2";        shift 2 ;;
        --out-dir)    OUT_DIR="$2";    shift 2 ;;
        --skip-719)   SKIP_719=1;      shift 1 ;;
        --skip-step2) SKIP_STEP2=1;    shift 1 ;;
        --skip-train) SKIP_TRAIN=1;    shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"
echo "# Night run — $(date '+%F %T')" > "$SUMMARY"
echo "Window: $DATE $START-$END | real: ${REAL_PATH:-none}" >> "$SUMMARY"
echo "" >> "$SUMMARY"

run_sim () {
    # run_sim <tag> <checkpoint_val> <sampler> <nsteps> <eta> <extra-args>
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

# ── Phase 1: checkpoint 0.719, full decode x eta matrix ───────────────────────
if [[ "$SKIP_719" -eq 0 ]]; then
    echo ""; echo "════════ PHASE 1: checkpoint 0.719 ════════"
    echo "# Phase 1 — checkpoint val_ema=0.719" >> "$SUMMARY"
    CK=0.719
    run_sim "ck${CK}_DDIM10_eta0"          "$CK" DDIM 10  0.0 ""
    run_sim "ck${CK}_DDIM10_eta1_default"  "$CK" DDIM 10  1.0 ""
    run_sim "ck${CK}_DDIM10_eta1_prior"    "$CK" DDIM 10  1.0 "--type-decode prior"
    run_sim "ck${CK}_DDIM10_eta04_default" "$CK" DDIM 10  0.4 ""
    run_sim "ck${CK}_DDIM10_eta04_prior"   "$CK" DDIM 10  0.4 "--type-decode prior"
    run_sim "ck${CK}_DDIM100_eta0"         "$CK" DDIM 100 0.0 ""
    run_sim "ck${CK}_DDPM100_default"      "$CK" DDPM 100 0.0 ""
    run_sim "ck${CK}_DDPM100_prior"        "$CK" DDPM 100 0.0 "--type-decode prior"
fi

# ── Phase 2: early-stochasticity (HYBRID_DDPM_PP) on checkpoint 0.681 ─────────
if [[ "$SKIP_STEP2" -eq 0 ]]; then
    echo ""; echo "════════ PHASE 2: early-stochasticity test (ck 0.681) ════════"
    echo "# Phase 2 — early stochasticity (HYBRID_DDPM_PP), ck 0.681" >> "$SUMMARY"
    CK=0.681
    # stochastic head + deterministic PP tail. --tail-steps = PP tail size; DDPM head = nsteps-tail.
    run_sim "s2_DDPMhead8_PPtail2"   "$CK" HYBRID_DDPM_PP 10 0.0 "--tail-steps 2"    # 8 DDPM + 2 PP
    run_sim "s2_DDPMhead5_PPtail5"   "$CK" HYBRID_DDPM_PP 10 0.0 "--tail-steps 5"    # 5 DDPM + 5 PP
    run_sim "s2_DDPMhead18_PPtail2"  "$CK" HYBRID_DDPM_PP 20 0.0 "--tail-steps 2"    # 18 DDPM + 2 PP
    # control: the inverse ordering that froze (fast head, stochastic tail)
    run_sim "s2_ctrl_PPhead8_DDPMtail2" "$CK" HYBRID_PP_DDPM 10 0.0 "--tail-steps 2"
fi

echo ""; echo "══════════════════════════════════════════"
echo "  Sims done. Summary: $SUMMARY"
echo "══════════════════════════════════════════"
echo "Read Phase 2 as: does any DDPM-head config move the market (unique mids >> 6,"
echo "  depth_pre_drop neg-fraction like DDPM ~20%) at low NFE? If yes -> fast sampler win."

# ── Phase 3: overnight training ───────────────────────────────────────────────
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo ""; echo "Launching CONDITIONAL_DROPOUT=0.1 training in background..."
    echo "  NB: configuration.py has IS_DATA_PREPROCESSED=False -> reruns preprocessing each"
    echo "  launch. If data/INTC/train.npy exists, set it True locally first to skip that."
    nohup python main.py > /tmp/training_night.log 2>&1 &
    echo "  training PID: $!  | log: /tmp/training_night.log"
else
    echo ""; echo "--skip-train set. Launch yourself: nohup python main.py > /tmp/training.log 2>&1 &"
fi
