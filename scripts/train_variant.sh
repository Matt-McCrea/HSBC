#!/bin/bash
# train_variant.sh — train ONE clean, freshly-provenanced checkpoint for the instability paper's
# "model under test" (see analysis/model_under_test.md). Three variants share identical seed and
# hyperparameters, so they'd collide on checkpoint FILENAME if left in the shared
# data/checkpoints/TRADES/ directory (same rounded val_ema at the same epoch is plausible across
# variants) -- every checkpoint this run produces gets moved into its own
# data/checkpoints/TRADES_<variant>/ the moment training stops, diffed against what existed before
# launch, so nothing gets silently overwritten by the next variant's run.
#
# Stops after 2 epochs (Lightning's own max_epochs via MAX_EPOCHS_OVERRIDE, not a wall-clock
# guess) -- matches "epoch 1, at most epoch 2 is enough, ~1.5h each" and the original TRADES
# paper's own light training regime. KEEP_EPOCH_CHECKPOINTS_FLAG is set so BOTH epoch=0 and
# epoch=1 checkpoints survive (not just whichever has the better val loss), since either may be
# the one worth testing. A wall-clock --hours cap (default 5h) is a SAFETY NET only, in case
# something hangs -- it should never actually bind if 2 epochs really take ~3h.
#
# THREE COMMANDS, ONE PER VARIANT:
#   bash scripts/train_variant.sh baseline
#   bash scripts/train_variant.sh reanchor
#   bash scripts/train_variant.sh ss
# (or just: bash scripts/train_three_variants.sh   -- runs all three back to back, unattended)
set -uo pipefail
VARIANT="${1:?usage: train_variant.sh <baseline|reanchor|ss> [hours]}"
HOURS="${2:-5}"
CKPT_DIR="data/checkpoints/TRADES"
DEST_DIR="data/checkpoints/TRADES_${VARIANT}"

if pgrep -f "main.py" > /dev/null; then echo "!! training already running -- kill it first (single GPU)."; exit 1; fi

# clean, deterministic flag state for THIS run -- never resume across variants (each is an
# independent training run), never leave a stale flag from a previous invocation.
rm -f RESUME_TRAINING_FLAG
touch UNCLAMP_DEPTH_FLAG KEEP_EPOCH_CHECKPOINTS_FLAG
echo "2" > MAX_EPOCHS_OVERRIDE
case "$VARIANT" in
  baseline) rm -f PRICE_REANCHOR_FLAG SCHEDULED_SAMPLING_FLAG ;;
  reanchor) touch PRICE_REANCHOR_FLAG; rm -f SCHEDULED_SAMPLING_FLAG ;;
  ss)       rm -f PRICE_REANCHOR_FLAG; touch SCHEDULED_SAMPLING_FLAG ;;
  *) echo "!! unknown variant: $VARIANT (want baseline|reanchor|ss)"; exit 1 ;;
esac

FLAGS=$(python3 -c "import constants as cst; print('UNCLAMP_DEPTH=%s PRICE_REANCHOR=%s SCHEDULED_SAMPLING=%s' % (cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR, cst.SCHEDULED_SAMPLING))")
echo "== training variant: $VARIANT   $FLAGS   epochs=2 (safety cap ${HOURS}h)   $(date '+%F %T') =="
mkdir -p "$DEST_DIR"
BEFORE=$(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | sort)

LOG="train_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"
timeout "${HOURS}h" python -u main.py > "$LOG" 2>&1
RC=$?
rm -f MAX_EPOCHS_OVERRIDE
if [[ "$RC" -eq 124 ]]; then
  echo "  !! hit the ${HOURS}h SAFETY cap before finishing 2 epochs -- unexpected, check $LOG"
else
  echo "  training process exited rc=$RC (see $LOG for the full trace either way)"
fi

# move every NEW checkpoint (not present before this launch) into this variant's own directory --
# whatever epoch(s) got saved, so nothing from a collision-prone shared filename gets lost.
AFTER=$(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | sort)
MOVED=0
comm -13 <(echo "$BEFORE") <(echo "$AFTER") | while read -r f; do
  [[ -n "$f" ]] || continue
  mv "$f" "$DEST_DIR/"
  echo "  moved: $(basename "$f") -> $DEST_DIR/"
done
echo ""
echo "checkpoints for $VARIANT:"
ls -la "$DEST_DIR"/*.ckpt 2>/dev/null || echo "  (none produced -- check $LOG)"
echo ""
echo "Next: bash scripts/pin_trained_variant.sh $VARIANT"
