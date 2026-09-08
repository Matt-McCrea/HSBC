#!/bin/bash
# train_three_variants.sh — trains baseline, reanchor, and ss (scheduled-sampling/teacher-forcing)
# checkpoints back to back, unattended (~5h safety cap each, ~15h total). Pins each one into
# analysis/model_under_test.md automatically as it finishes -- safe to walk away from / get
# disconnected mid-run; resume by rerunning this same command (already-completed variants are
# skipped once analysis/model_under_test.md has their CKPT_PATH_<VARIANT> line).
#
# ONE COMMAND (run with nohup so a dropped connection doesn't kill it):
#   nohup bash scripts/train_three_variants.sh > train_three_variants.log 2>&1 &
#   disown
# Then check progress any time with:  tail -f train_three_variants.log
set -uo pipefail
HOURS="${1:-5}"

for VARIANT in baseline reanchor ss; do
  VAR_UPPER=$(echo "$VARIANT" | tr '[:lower:]' '[:upper:]')
  if grep -q "^CKPT_PATH_${VAR_UPPER}=" analysis/model_under_test.md 2>/dev/null; then
    echo "SKIP $VARIANT -- already pinned in analysis/model_under_test.md"
    continue
  fi
  echo "==================== $VARIANT ===================="
  bash scripts/train_variant.sh "$VARIANT" "$HOURS"
  bash scripts/pin_trained_variant.sh "$VARIANT"
done

echo ""
echo "All variants attempted. Check analysis/model_under_test.md for CKPT_PATH_BASELINE/"
echo "CKPT_PATH_REANCHOR/CKPT_PATH_SS -- any missing means that variant's training or pinning"
echo "failed; check its train_<variant>_*.log."
