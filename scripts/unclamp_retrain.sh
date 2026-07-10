#!/bin/bash
# unclamp_retrain.sh — retrain TRADES with SIGNED depth (marketable orders kept as depth<0).
#
# THE ROOT CAUSE we're attacking: training clamps depth≥0 (utils_data.py:324), so a marketable,
# spread-crossing order (depth<0) is NEVER a training target. The model can only emit aggression as
# sampling *noise* — which is exactly why few-step deterministic samplers (which kill that noise)
# freeze the mid-price. Unclamping lets the model LEARN to cross the spread, so aggression becomes a
# signal it can reproduce deterministically.
#
# Mechanism: constants.UNCLAMP_DEPTH reads env var UNCLAMP_DEPTH. This shell exports it, so BOTH the
# preprocessing clamp AND the simulation-time conditioning clamp keep signed depth, consistently.
# IMPORTANT: you must export UNCLAMP_DEPTH=1 at SIMULATION time too (the eval shells inherit it if you
# export it in the same shell) — otherwise conditioning won't match training.
#
# Requires reprocessing (unclamped depth shifts μ_depth/σ_depth in normalization_stats), so the old
# clamped .npy + stats are backed up and IS_DATA_PREPROCESSED must be False.
#
# Usage:
#   bash scripts/unclamp_retrain.sh            # backup clamped data, reprocess+train unclamped
#   bash scripts/unclamp_retrain.sh --restore  # restore the clamped baseline data and exit

set -uo pipefail
STOCK="INTC"; DATA_DIR="data/${STOCK}"; BK="${DATA_DIR}/_clamped_backup"

if [[ "${1:-}" == "--restore" ]]; then
  [[ -d "$BK" ]] || { echo "no backup at $BK"; exit 1; }
  cp -v "$BK"/*.npy "$BK"/*.json "$DATA_DIR"/ 2>/dev/null
  echo "restored clamped baseline. Unset UNCLAMP_DEPTH for clamped runs."; exit 0
fi

export UNCLAMP_DEPTH=1
echo "UNCLAMP_DEPTH=$UNCLAMP_DEPTH  (signed depth active in preprocessing + conditioning)"

# 1. Sanity: preprocessing must run so the unclamped depth actually reaches the .npy + stats.
if grep -qE '^\s*self\.IS_DATA_PREPROCESSED\s*=\s*True' configuration.py; then
  echo "!! configuration.py has IS_DATA_PREPROCESSED = True — set it False so preprocessing re-runs"
  echo "   with unclamped depth, then re-run this script."; exit 1
fi

# 2. Back up the clamped baseline (.npy + normalization stats) once, so --restore works.
if [[ ! -d "$BK" ]]; then
  mkdir -p "$BK"
  cp -v "$DATA_DIR"/*.npy "$BK"/ 2>/dev/null || true
  cp -v "$DATA_DIR"/*normalization*.json "$BK"/ 2>/dev/null || true
  echo "backed up clamped baseline → $BK"
else
  echo "backup already exists at $BK (not overwriting)"
fi

# 3. Reprocess + train. main.py preprocesses first (IS_DATA_PREPROCESSED=False) then trains;
#    the new checkpoints land in data/checkpoints/TRADES/ alongside the clamped ones.
#    Tip: check the printed depth stats — mean_depth should now be < the clamped ~1.38, std wider.
echo "── launching preprocess + train (unclamped) ──"
python main.py

echo ""; echo "══════════════════════════════════════════"
echo "  Trained. New checkpoints in data/checkpoints/TRADES/ (unclamped-depth model)."
echo "  Evaluate with the SAME env var exported so conditioning matches training:"
echo ""
echo "    export UNCLAMP_DEPTH=1"
echo "    bash scripts/open_loop_sweep.sh --ids \"<new-val-loss-id>\""
echo "    bash scripts/eval_new_checkpoint.sh --real <replay csv> --ids \"<new-val-loss-id>\""
echo ""
echo "  WATCH: depth_pre_drop should now show a real NEGATIVE bucket (marketable orders the model"
echo "  learned), and DDIM/DPM++ should move the mid instead of freezing. Compare against the"
echo "  clamped baseline in new_ckpt.md."
echo "══════════════════════════════════════════"
