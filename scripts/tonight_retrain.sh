#!/bin/bash
# tonight_retrain.sh — ONE-PASTE launcher for the evening GPU session.
#   1. archives any existing checkpoints (so the fresh retrain is the unambiguous newest, and the
#      later sweep's val_ema -id lookup can't collide) — reversible, nothing is deleted;
#   2. launches the full reanchor + unclamp retrain from scratch (reanchor_retrain.sh does its own
#      preflight, flag-set, data backup, and background launch);
#   3. waits, then auto-runs the --check so you see BOTH corrections took before walking away.
#
# The exploratory sweep is NOT bundled here: it needs the CONVERGED checkpoint (~22h away), so it's
# a separate paste in the NEXT session. This shell prints that command at the end.
#
# Usage:  bash scripts/tonight_retrain.sh

set -uo pipefail
CKPT_DIR="data/checkpoints/TRADES"

echo "════════════════════════════════════════════════════════════════"
echo "  TONIGHT: archive old ckpts → launch reanchor retrain → --check"
echo "════════════════════════════════════════════════════════════════"

# 0. Refuse if a training run is already going (GPU contention).
if pgrep -f "main.py" > /dev/null; then
  echo "!! main.py is already running — kill it first. Refusing."; exit 1
fi

# 1. Archive existing checkpoints (reversible: moved, not deleted). Keeps the -id float-lookup
#    unambiguous for sweep_reanchored.sh, which auto-discovers the newest .ckpt.
shopt -s nullglob
existing=("$CKPT_DIR"/*.ckpt)
if (( ${#existing[@]} )); then
  ARCH="$CKPT_DIR/_archive_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ARCH"
  mv -v "${existing[@]}" "$ARCH"/
  echo "→ archived ${#existing[@]} old checkpoint(s) into $ARCH (restore with: mv \"$ARCH\"/*.ckpt \"$CKPT_DIR\"/)"
else
  echo "→ no existing checkpoints to archive"
fi
echo ""

# 2. Launch the retrain. reanchor_retrain.sh sets BOTH flags, pre-flights them (<2s), backs up the
#    pre-anchor data, verifies IS_DATA_PREPROCESSED=False, and launches python main.py in background.
#    If it refuses (e.g. IS_DATA_PREPROCESSED=True), we stop here — nothing was lost.
if ! bash scripts/reanchor_retrain.sh; then
  echo ""
  echo "!! retrain launcher refused/failed (see message above). Old checkpoints are safe in the"
  echo "   _archive_* dir — restore them if you're abandoning the retrain. Stopping."
  exit 1
fi

# 3. Give preprocessing + the first epoch's logging time to appear, then auto-check.
echo ""
echo "── waiting 150s for preprocessing to rewrite normalization_stats.json before --check ──"
echo "   (preprocessing can take a few minutes; if the numbers below look stale, just re-run:"
echo "    bash scripts/reanchor_retrain.sh --check)"
sleep 150
echo ""
bash scripts/reanchor_retrain.sh --check

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  If --check shows |mean_price| < ~100 and UNCLAMP=ON REANCHOR=ON,"
echo "  training is underway correctly. Let it converge (~22h)."
echo ""
echo "  NEXT SESSION, once converged, run the exploratory sweep:"
echo "      bash scripts/sweep_reanchored.sh"
echo "  (auto-discovers the new checkpoint; re-tunes sigma/target-exec and"
echo "   runs the decisive 75-min stability test.)"
echo "════════════════════════════════════════════════════════════════"
