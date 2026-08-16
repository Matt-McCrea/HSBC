#!/bin/bash
# ab_profile_ddpm_fix.sh — clean, apples-to-apples before/after timing for the DDPM
# dead-loss-computation fix (commit 89cb792).
#
# Contention makes any single timing number worthless for a real comparison — GPU sharing with
# another job inflates or distorts absolute times unpredictably, in either direction. This script
# gives a genuine A/B instead: temporarily reverts JUST models/diffusers/gaussian_diffusion.py to
# its state immediately before the fix, profiles it, restores it to HEAD, profiles again — same
# GPU, same conditions, same checkpoint, only that one file's contents different between the two
# runs. The file is GUARANTEED restored to HEAD on exit (trap), even on error or Ctrl-C.
#
# Run this ALONE on a free GPU — refuses to start if a training process is still active, since an
# A/B run under contention is exactly the unfair comparison this script exists to avoid.
#
# Usage (after `git pull`, so you actually have this script + the fix commit):
#   bash scripts/ab_profile_ddpm_fix.sh                       # defaults: --id 0.656 --n-calls 5
#   bash scripts/ab_profile_ddpm_fix.sh --id 0.656 --n-calls 10

set -euo pipefail
FILE="models/diffusers/gaussian_diffusion.py"
FIX_COMMIT="89cb792"          # the commit that added compute_loss=False + profiler markers
PRE_COMMIT="${FIX_COMMIT}~1"  # its parent -- last commit before the fix touched this file
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(--id 0.656 --n-calls 5)

OUT_DIR="ab_profile/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT_DIR"

# Safety: refuse if anything else is on the GPU.
if pgrep -f "main.py" > /dev/null; then
  echo "!! a main.py (training) process is still running. Kill it first -- an A/B under"
  echo "   contention is exactly the unfair comparison this script exists to avoid."
  exit 1
fi

# Safety: refuse if there are uncommitted local changes to the file -- this script overwrites it
# on disk and restores via 'git checkout HEAD', which would silently discard anything uncommitted.
if ! git diff --quiet -- "$FILE" || ! git diff --cached --quiet -- "$FILE"; then
  echo "!! $FILE has uncommitted changes. Commit, stash, or discard them first."
  exit 1
fi

# GUARANTEE restoration to HEAD no matter how this script exits.
cleanup() { git checkout HEAD -- "$FILE" 2>/dev/null; echo "[cleanup] restored $FILE to HEAD"; }
trap cleanup EXIT

echo "══════════════════════════════════════════"
echo "BEFORE — pre-fix (commit ${PRE_COMMIT}, dead loss computation still runs every step)"
echo "══════════════════════════════════════════"
git checkout "$PRE_COMMIT" -- "$FILE"
python scripts/profile_ddpm.py "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/before.txt"

echo ""
echo "══════════════════════════════════════════"
echo "AFTER — current HEAD (loss computation skipped during sampling)"
echo "══════════════════════════════════════════"
git checkout HEAD -- "$FILE"
python scripts/profile_ddpm.py "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/after.txt"

echo ""
echo "══════════════════════════════════════════"
echo "  Saved: $OUT_DIR/before.txt and $OUT_DIR/after.txt"
echo "  Compare the 'Self CPU time total' / 'Self CUDA time total' line at the bottom of each"
echo "  table -- total profiled time across the same n-calls sample() calls, same GPU, same"
echo "  checkpoint, only the dead-loss-computation removal different. That delta is your real,"
echo "  defensible speedup number."
echo "  (before.txt won't show ddpm_augment/NN_forward/loss_computation phase labels -- those"
echo "   markers were added in the same commit as the fix, so the raw op-level table is what to"
echo "   read there; after.txt has the clean phase breakdown.)"
echo "══════════════════════════════════════════"
