#!/bin/bash
# overnight_batch.sh — three more Exp 2 spot-checks (reanchor + ss on 2015-01-15, baseline on
# 2015-01-22) then Exp 3 (teacher-forced control) for all three variants. Sequential (single GPU),
# unattended. Resumable: each Exp 2 step uses a fixed --out-dir so a rerun after an interruption
# hits exp2_survival_sweep.sh's own .done sentinels and skips finished work; each Exp 3 step is
# skipped if its output file already exists.
#
# ONE COMMAND (run with nohup so a dropped connection doesn't kill it):
#   nohup bash scripts/overnight_batch.sh > overnight_batch.log 2>&1 &
#   disown
# Check progress any time with:  tail -f overnight_batch.log
set -uo pipefail

echo "==== $(date '+%F %T') exp2 reanchor 2015-01-15 ===="
bash scripts/exp2_survival_sweep.sh --days 20150115 --seeds 30 --et 11:00:00 --variant reanchor \
  --out-dir exp2_results/overnight_reanchor_20150115

echo "==== $(date '+%F %T') exp2 ss 2015-01-15 ===="
bash scripts/exp2_survival_sweep.sh --days 20150115 --seeds 30 --et 11:00:00 --variant ss \
  --out-dir exp2_results/overnight_ss_20150115

echo "==== $(date '+%F %T') exp2 baseline 2015-01-22 ===="
bash scripts/exp2_survival_sweep.sh --days 20150122 --seeds 30 --et 11:00:00 --variant baseline \
  --out-dir exp2_results/overnight_baseline_20150122

for VARIANT in baseline reanchor ss; do
  OUT="exp3_results/overnight_${VARIANT}"
  if ls "$OUT"/open_loop_*.json >/dev/null 2>&1; then
    echo "SKIP exp3 $VARIANT -- already done"
    continue
  fi
  echo "==== $(date '+%F %T') exp3 $VARIANT ===="
  bash scripts/exp3_teacher_forced.sh --variant "$VARIANT" --out-dir "$OUT"
done

echo ""
echo "==== $(date '+%F %T') ALL DONE ===="
echo "Read: exp2_results/overnight_*/summary.md and exp3_results/overnight_*/open_loop_*.txt"
