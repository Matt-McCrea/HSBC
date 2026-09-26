#!/bin/bash
# backup_before_gap.sh — back up everything not yet in git before a month-long absence: the
# never-committed INTC checkpoints, and every experiment's result data (exp0-4_results/). Reports
# the size of the raw per-run generated CSVs separately (NOT staged automatically -- those can be
# large; the .score.json/summary.md files already staged contain every number logged so far, so
# the raw CSVs are only needed later for detailed per-run charts, your call whether to include).
#
# ONE COMMAND:
#   bash scripts/backup_before_gap.sh
set -uo pipefail

echo "== staging INTC checkpoints (never committed until now) =="
git add data/checkpoints/TRADES_baseline data/checkpoints/TRADES_reanchor data/checkpoints/TRADES_ss 2>&1

echo ""
echo "== staging all experiment result data =="
git add exp0_results exp2_results exp3_results exp4_results 2>&1

echo ""
echo "== staged files (first 40) =="
git status --short | head -40
N=$(git status --short | wc -l | tr -d ' ')
echo "... ($N files staged total)"

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
git commit -m "Back up INTC checkpoints and all experiment result data before a month-long gap

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
git push origin rl-execution

echo ""
echo "Done. If the raw-CSV size above is manageable and you want those backed up too, run:"
echo "  git add ABIDES/log/world_agent_*"
echo "  git commit -m 'Back up raw per-run generated CSVs'"
echo "  git push origin rl-execution"
