#!/bin/bash
# eval_new_checkpoint.sh — closed-loop battery, now loopable over MULTIPLE checkpoints.
#
# THE QUESTION (unchanged): does the sampler-collapse freeze depend on checkpoint quality?
# Old 0.681 DDIM-10 η=0: 72% depth-0, 6 mids (frozen). New 0.656 (dropout retrain) is NO better
# (see new_ckpt.md: 9-17 mids, 78-84% depth-0). So now we sweep checkpoints to map sensitivity.
#
# Per checkpoint it runs a LEAN, information-ordered battery:
#   DDPM_100          — positive control (stochastic, should move) — added back this round
#   DDIM10_eta0       — negative control (deterministic, should freeze)
#   HYBRID_DDPM_PP_8+2— the early-stochasticity lever (unfroze the OLD ckpt; did it regress?)
# then an OPTIONAL fuller set (CFG g1.5/g0.7, DPMpp) only if --full is passed.
#
# Resumable (per-checkpoint .done sentinels). Ctrl-C any time; rerun resumes.
#
# Usage — sweep a curated bracket of checkpoints (best → mid → old → exploder → untrained):
#   bash scripts/eval_new_checkpoint.sh \
#       --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv \
#       --ids "0.656 0.671 0.681 0.719 2.869"
# Single checkpoint, full battery (adds CFG + DPMpp):
#   bash scripts/eval_new_checkpoint.sh --real <csv> --ids 0.656 --full
# Auto/best checkpoint only:  (omit --ids)
#
# WATCH per run in summary.md:  DIAG depth_pre_drop (neg / 0 fractions) + unique mid count.
#   FROZEN  ~72-84% at depth 0, <20 mids  |  MOVING (DDPM/real) depth spread, dozens of mids

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"
REAL=""; IDS=""; FULL=0; OUT_ROOT="eval_new_ckpt/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --real) REAL="$2"; shift 2;;
  --ids)  IDS="$2";  shift 2;;      # space-separated list of val-loss ids; empty = auto/best
  --id)   IDS="$2";  shift 2;;      # alias, single id
  --full) FULL=1; shift;;           # add CFG (g1.5/g0.7) + DPMpp per checkpoint
  --start) ST="$2"; shift 2;; --end) ET="$2"; shift 2;;
  --out-dir) OUT_ROOT="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
[[ -z "$IDS" ]] && IDS="__auto__"   # sentinel = let world_agent_sim pick best/lowest val

run () { # run <out_dir> <id> <tag> <type> <nsteps> <eta> <extra>
  local OUT_DIR="$1" ID="$2" TAG="$3" TYPE="$4" NS="$5" ETA="$6" EXTRA="$7"
  local SUM="$OUT_DIR/summary.md"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET"
           -d True -m TRADES -type "$TYPE" -nsteps "$NS" -eta "$ETA")
  [[ "$ID" != "__auto__" ]] && A+=(-id "$ID")
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

for ID in $IDS; do
  TAGID=$([[ "$ID" == "__auto__" ]] && echo "auto" || echo "$ID")
  OUT_DIR="$OUT_ROOT/ckpt_${TAGID}"; mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
  echo "# Checkpoint $TAGID — $(date '+%F %T')" > "$SUM"; echo "" >> "$SUM"
  echo ""; echo "████ CHECKPOINT ${TAGID} ████"

  echo "════ STAGE 1: controls — DDPM (should move) vs DDIM (should freeze) ════"
  run "$OUT_DIR" "$ID" "DDPM_100"          DDPM 100 0.0 ""                          # positive control
  run "$OUT_DIR" "$ID" "DDIM10_eta0"       DDIM 10  0.0 ""                          # negative control
  run "$OUT_DIR" "$ID" "HYBRID_DDPM_PP_8+2" HYBRID_DDPM_PP 10 0.0 "--tail-steps 2" # early-stochastic lever
  run "$OUT_DIR" "$ID" "CHURN_10_s3_k0.3"  CHURN 10 0.0 "--churn-steps 3 --churn-strength 0.3"  # NEW: tunable early churn
  run "$OUT_DIR" "$ID" "CHURN_10_s4_k0.5"  CHURN 10 0.0 "--churn-steps 4 --churn-strength 0.5"  # NEW: stronger churn

  if [[ "$FULL" == "1" ]]; then
    echo "════ STAGE 2: classifier-free guidance ════"
    run "$OUT_DIR" "$ID" "DDIM10_eta0_g1.5" DDIM 10 0.0 "--guidance-scale 1.5"
    run "$OUT_DIR" "$ID" "DDIM10_eta0_g0.7" DDIM 10 0.0 "--guidance-scale 0.7"
    echo "════ STAGE 3: fast ODE ════"
    run "$OUT_DIR" "$ID" "DPM_SOLVER_PP_10" DPM_SOLVER_PP 10 0.0 ""
  fi
done

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summaries under: $OUT_ROOT/ckpt_*/summary.md"
echo "══════════════════════════════════════════"
echo "READ: per checkpoint, does DDPM_100 stay MOVING while DDIM10 FREEZES? If the freeze"
echo "  persists across ALL checkpoints (incl. best 0.656 and untrained 2.869), it's a"
echo "  sampler-intrinsic collapse, not a training-maturity artifact."
