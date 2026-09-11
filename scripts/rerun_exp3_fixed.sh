#!/bin/bash
# rerun_exp3_fixed.sh — reruns Exp 3 (teacher-forced control) for all three variants with the
# bucket-by-time fix, then prints all three results. See open_loop_eval.py's own comment for the
# fix; ask Claude "what was the problem" for the plain explanation.
#
# ONE COMMAND:
#   bash scripts/rerun_exp3_fixed.sh
set -uo pipefail

for VARIANT in baseline reanchor ss; do
  echo "==== $(date '+%F %T') exp3 $VARIANT (fixed bucketing) ===="
  bash scripts/exp3_teacher_forced.sh --variant "$VARIANT" --out-dir "exp3_results/fixed_${VARIANT}"
done

echo ""
echo "==== all three done ===="
cat exp3_results/fixed_*/open_loop_*.json
