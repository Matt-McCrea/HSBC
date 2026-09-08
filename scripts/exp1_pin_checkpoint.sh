#!/bin/bash
# exp1_pin_checkpoint.sh — Experiment 1 (instability paper). Does NOT auto-resolve which
# checkpoint is "the model under test" -- that can't be done from a checkpoint file alone
# (configuration.py never persists UNCLAMP_DEPTH/PRICE_REANCHOR/SCHEDULED_SAMPLING into the saved
# hyper_parameters, and the file-flag gating means it's unrecoverable after the fact regardless).
# See analysis/model_under_test.md for the full reasoning -- this script is the checklist that
# turns that document's "Confirmed checkpoint" section from empty to filled in.
#
# Usage:  bash scripts/exp1_pin_checkpoint.sh
set -uo pipefail
CKPT_DIR="data/checkpoints/TRADES"

echo "== current sentinel-flag state (must be UNCLAMP_DEPTH=True, SCHEDULED_SAMPLING=False for a"
echo "   fresh training launch here) =="
python3 -c "import constants as cst; print('UNCLAMP_DEPTH   =', cst.UNCLAMP_DEPTH); \
print('SCHEDULED_SAMPLING =', cst.SCHEDULED_SAMPLING); \
print('PRICE_REANCHOR  =', cst.PRICE_REANCHOR, '(open question -- train/eval BOTH ways)')"
if python3 -c "import constants as cst; cst.DEPTH_INDEX_FIX" 2>/dev/null; then
  python3 -c "import constants as cst; print('DEPTH_INDEX_FIX =', cst.DEPTH_INDEX_FIX)"
else
  echo "DEPTH_INDEX_FIX = (no such flag on this branch -- the self-referential depth-indexing"
  echo "  fix is UNCONDITIONAL here, i.e. always on; that toggle exists only on the branch that"
  echo "  added f9edaa3, for an intentional no-fixes-baseline comparison. Confirm unconditional:"
  grep -q "index = j - 1" utils/utils_data.py \
    && echo "  confirmed: utils/utils_data.py uses index = j - 1 unconditionally." \
    || echo "  !! could not confirm — grep for the fix in utils/utils_data.py by hand."
fi

echo ""
echo "== every checkpoint present in $CKPT_DIR =="
if ls "$CKPT_DIR"/*.ckpt >/dev/null 2>&1; then
  ls -la "$CKPT_DIR"/*.ckpt
else
  echo "  (none)"
  echo ""; echo "READ: nothing to select. Place a checkpoint first."; exit 1
fi

# KNOWN DECOYS by val_ema -- do not select these for the model-under-test:
#   0.627 / 0.681 / 0.719 -- "Phase 1 diagnosis" checkpoints, explicitly PRE-FIX (predate
#     UNCLAMP_DEPTH being trained in at all) per analysis/MASTER_RESULTS.md's own phase table.
#   0.7   -- the documented FALLBACK recovery checkpoint (commit b0f449c: "Not the 0.627 winner
#     ... but a usable checkpoint"), not the preferred one.
# SS-epoch checkpoints (scheduled sampling, resumed from 0.724) are NOT distinguishable from the
# baseline by val_ema alone -- their filenames don't say "SS". If more than one checkpoint's
# val_ema starts with "0.724", or if you're not sure whether a file is the plain baseline or an
# SS-resumed one, do NOT guess: check whichever training log / launch history exists for it, or
# retrain cleanly per analysis/model_under_test.md's "What Track B must do" section.
echo ""
echo "== flagging known decoys (pre-fix / documented-fallback val_ema values) =="
DECOY_HIT=0
for v in "0.627" "0.681" "0.719" "0.7_epoch"; do
  if ls "$CKPT_DIR"/val_ema=${v}* >/dev/null 2>&1; then
    echo "  AVOID: $(ls "$CKPT_DIR"/val_ema=${v}*)  <- pre-fix or documented fallback, do not select"
    DECOY_HIT=1
  fi
done
[[ "$DECOY_HIT" == "0" ]] && echo "  (none of the known decoy val_ema values are present)"

echo ""
echo "== looking for the documented candidate (val_ema=0.724*) =="
MATCHES=$(ls "$CKPT_DIR"/val_ema=0.724* 2>/dev/null)
N_MATCHES=$(echo "$MATCHES" | grep -c . || true)
if [[ "$N_MATCHES" -eq 1 ]]; then
  echo "  FOUND exactly one: $MATCHES"
  echo "  This is the Phase-2 baseline per analysis/model_under_test.md."
  echo ""
  echo "  ACTION: copy this EXACT path into analysis/model_under_test.md's 'Confirmed checkpoint'"
  echo "  section now, then use it with every Track B script's --ckpt-path (never --id/-id, which"
  echo "  matches by rounded val_ema and can silently pick the wrong file if two checkpoints share"
  echo "  one — see analysis/MASTER_RESULTS.md 1.4)."
elif [[ "$N_MATCHES" -gt 1 ]]; then
  echo "  AMBIGUOUS — $N_MATCHES files all match val_ema=0.724*:"
  echo "$MATCHES" | sed 's/^/    /'
  echo "  Do NOT pick one arbitrarily — these could be the plain baseline and an SS-resumed"
  echo "  checkpoint that happen to share a rounded val_ema (the exact landmine"
  echo "  analysis/MASTER_RESULTS.md 1.4 warns about). Check mtimes / whatever training log exists"
  echo "  for each, or retrain cleanly, before filling in analysis/model_under_test.md."
else
  echo "  NOT FOUND. Per analysis/model_under_test.md this checkpoint is known to live only on"
  echo "  whichever machine trained it (never committed to git) and may have been lost the same"
  echo "  way 0.627 was (see commit b0f449c). If it's genuinely not on this box, retrain per"
  echo "  analysis/model_under_test.md's 'What Track B must do' section — do NOT guess and do NOT"
  echo "  proceed to Exp 2/3/4 until that file's 'Confirmed checkpoint' section is filled in."
fi

echo ""
echo "READ: fill in analysis/model_under_test.md's 'Confirmed checkpoint' section with the EXACT"
echo "file path this determines, before any Track B script runs."
