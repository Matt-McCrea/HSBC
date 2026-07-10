#!/bin/bash
# eval_new_checkpoint.sh — decisive tests for the dropout-retrained checkpoint.
#
# THE QUESTION: did better calibration (dropout, val 0.681→0.656) un-collapse the depth
# distribution and unfreeze the fast/deterministic samplers? Old checkpoint 0.681 gave
# DDIM-10 η=0: 72% of orders at depth 0, 0% marketable, 6 unique mids (FROZEN). We want to
# see the depth histogram spread out and the mid-price move.
#
# Runs in information-per-minute order (headline tests first; slow DDPM reference last), so a
# short session still answers the big question. Resumable; each run appends flow_mix + the
# depth histogram to summary.md.
#
# Usage (auto-selects the best/lowest-val checkpoint in data/checkpoints/TRADES/):
#   bash scripts/eval_new_checkpoint.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv
# Pin a specific checkpoint:  --id 0.656
#
# WATCH in summary.md, per run:  DIAG depth_pre_drop (neg / 0 fractions) and unique mid-prices.
#   OLD 0.681 DDIM-10 η=0:  neg~0%, 0=72%, 6 mids (frozen)
#   TARGET (DDPM / real):   neg~3-24%, depth spread across levels, dozens of mids

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"
REAL=""; ID=""; OUT_DIR="eval_new_ckpt/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --real) REAL="$2"; shift 2;; --id) ID="$2"; shift 2;;
  --start) ST="$2"; shift 2;; --end) ET="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
echo "# New-checkpoint eval — $(date '+%F %T')  (id=${ID:-auto/best})" > "$SUM"; echo "" >> "$SUM"

run () { # run <tag> <type> <nsteps> <eta> <extra>
  local TAG="$1" TYPE="$2" NS="$3" ETA="$4" EXTRA="$5"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET"
           -d True -m TRADES -type "$TYPE" -nsteps "$NS" -eta "$ETA")
  [[ -n "$ID" ]] && A+=(-id "$ID")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA" ]] && A+=($EXTRA)
  echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1
    if [[ -n "$CSV" && -n "$REAL" ]]; then python -m evaluation.quantitative_eval.flow_mix --real "$REAL" --gen "$CSV" 2>&1
    elif [[ -n "$CSV" ]]; then python -m evaluation.quantitative_eval.flow_mix --gen "$CSV" 2>&1; fi
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

echo "════ STAGE 1: the headline — did the freeze get fixed? ════"
echo "# Stage 1 — freeze test (fast)" >> "$SUM"
run "DDIM10_eta0"        DDIM 10 0.0 ""                          # SAME config that froze on 0.681
run "HYBRID_DDPM_PP_8+2" HYBRID_DDPM_PP 10 0.0 "--tail-steps 2" # stochastic head (drifted on 0.681)

echo "════ STAGE 2: classifier-free guidance (unlocked by the dropout retrain) ════"
echo "# Stage 2 — CFG (note: guidance != 1.0 doubles NFE)" >> "$SUM"
run "DDIM10_eta0_g1.5"  DDIM 10 0.0 "--guidance-scale 1.5"      # sharpen conditioning
run "DDIM10_eta0_g0.7"  DDIM 10 0.0 "--guidance-scale 0.7"      # toward the marginal (more diversity)

echo "════ STAGE 3: fast ODE + DDPM reference ════"
echo "# Stage 3 — fast ODE + reference" >> "$SUM"
run "DPM_SOLVER_PP_10"  DPM_SOLVER_PP 10 0.0 ""
run "DDPM100_reference" DDPM 100 0.0 ""                          # this ckpt's 'gold' depth profile (slow)

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary: $SUM"
echo "══════════════════════════════════════════"
echo "READ: compare each run's  DIAG depth_pre_drop  and unique mid count against"
echo "  OLD 0.681 DDIM-10 η=0 (neg~0%, 72% at depth 0, 6 mids). If the new checkpoint's"
echo "  DDIM-10 spreads depth and moves the mid, better calibration fixed it."
