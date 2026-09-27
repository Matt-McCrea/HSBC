#!/bin/bash
# backup_before_gap.sh — back up everything not yet in git before a month-long absence: every
# trained checkpoint directory, and every experiment's result data (exp0-4_results/). Reports the
# size of the raw per-run generated CSVs separately (NOT staged automatically -- those can be
# large; the .score.json/summary.md files already staged contain every number logged so far, so
# the raw CSVs are only needed later for detailed per-run charts, your call whether to include).
#
# `data/` is gitignored wholesale (it also covers the licensed LOBSTER market data, which must
# never be committed -- see READMEFORMEHMET.md) -- checkpoints live under data/checkpoints/ so a
# plain `git add` silently skips them with no error. -f is required, every time, or this fails
# exactly the way it did on 2026-09-26/27: a "successful" backup commit containing zero
# checkpoints, discovered only after the remote session was already gone.
#
# ONE COMMAND:
#   bash scripts/backup_before_gap.sh
set -uo pipefail

echo "== staging every checkpoint directory (data/ is gitignored -- -f is required, not optional) =="
git add -f data/checkpoints/TRADES_* 2>&1

echo ""
echo "== staging all experiment result data =="
git add exp0_results exp2_results exp3_results exp4_results 2>&1

echo ""
echo "== staged files (first 40) =="
git status --short | head -40
N=$(git status --short | wc -l | tr -d ' ')
echo "... ($N files staged total)"

N_CKPT_ON_DISK=$(find data/checkpoints/TRADES_* -name "*.ckpt" 2>/dev/null | wc -l | tr -d ' ')
N_CKPT_STAGED=$(git status --short -- data/checkpoints 2>/dev/null | grep -c "\.ckpt$")
echo ""
echo "checkpoints on disk: $N_CKPT_ON_DISK   checkpoints staged: $N_CKPT_STAGED"
if [[ "$N_CKPT_ON_DISK" -gt 0 && "$N_CKPT_STAGED" -eq 0 ]]; then
  echo "!! checkpoints exist on disk but NONE are staged -- the exact silent failure this script"
  echo "!! was written to catch. Do not proceed. Check .gitignore and rerun with git add -f by hand."
  exit 1
fi

echo ""
echo "== raw per-run generated CSVs -- NOT staged, size for you to judge =="
du -ch ABIDES/log/world_agent_* 2>/dev/null | tail -1

if [[ "$N" -eq 0 ]]; then
  echo ""
  echo "Nothing to commit -- everything already backed up."
  exit 0
fi

echo ""
echo "Committing and pushing the staged files..."
git commit -m "Back up all checkpoints and experiment result data before a month-long gap

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
git push origin rl-execution

echo ""
echo "Done. If the raw-CSV size above is manageable and you want those backed up too, run:"
echo "  git add ABIDES/log/world_agent_*"
echo "  git commit -m 'Back up raw per-run generated CSVs'"
echo "  git push origin rl-execution"
