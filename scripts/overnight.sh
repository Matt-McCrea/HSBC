#!/bin/bash
# overnight.sh — everything for tonight in ONE command. Launch and sleep.
#
#   bash scripts/overnight.sh
#   bash scripts/overnight.sh /path/to/real/processed_orders.csv    # override the real-data path
#
# Runs, in cheap-first order, both unattended and resumable:
#   1. open_loop_sweep.sh   — 7 checkpoints × 6 samplers (incl. CHURN), minutes/cell (~1-2 h).
#                             Sensitivity map; done early so there's signal if you wake up.
#   2. eval_new_checkpoint  — closed-loop battery (DDPM, DDIM, HYBRID_DDPM_PP, 2×CHURN) on the
#                             bracket best→mid→old→exploder→untrained (~90 min/ckpt).
#
# Neither uses -e, so a single failed run never kills the night; both skip already-done work on
# rerun. All output tees to overnight/<timestamp>/master.log plus each stage's own dirs.
#
# NOTE: this is the CLAMPED-baseline night. The unclamp retrain is a separate daytime job
# (scripts/unclamp_retrain.sh) — do NOT export UNCLAMP_DEPTH for this run.

set -uo pipefail
REAL="${1:-ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv}"
IDS="0.656 0.671 0.681 0.719 2.869"      # closed-loop bracket
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="overnight/${STAMP}"; mkdir -p "$DIR"
LOG="$DIR/master.log"

# unset in case a previous shell left it set — tonight is the clamped baseline
unset UNCLAMP_DEPTH

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== OVERNIGHT START ==="
say "real data : $REAL"
say "ids       : $IDS"
[[ -f "$REAL" ]] || say "WARNING: real data path not found — flow_mix will run gen-only"

say "──── STAGE 1/2: open-loop sensitivity matrix (cheap) ────"
bash scripts/open_loop_sweep.sh 2>&1 | tee -a "$LOG"
say "stage 1 done"

say "──── STAGE 2/2: closed-loop battery on bracket (slow) ────"
bash scripts/eval_new_checkpoint.sh --real "$REAL" --ids "$IDS" 2>&1 | tee -a "$LOG"
say "stage 2 done"

say "=== OVERNIGHT COMPLETE ==="
say "Read next: open_loop_sweep/*/  (summary table)  and  eval_new_ckpt/*/ckpt_*/summary.md"
say "Look for: a depth_pre_drop NEGATIVE bucket + rising unique-mid count on CHURN, and whether"
say "DDPM moves while DDIM freezes across ALL checkpoints (sampler-intrinsic) or only some."
