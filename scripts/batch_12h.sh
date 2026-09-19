#!/bin/bash
# batch_12h.sh — four more 90-min Exp 2 spot-checks (~3h each, ~12h total), evening out coverage
# across the three variants (baseline had 3 days already, reanchor/ss had 2 each): two new
# baseline days, one new reanchor day, one new ss day. Sequential (single GPU), unattended.
# Resumable: fixed --out-dir per step hits exp2_survival_sweep.sh's own .done sentinels.
#
# ONE COMMAND (run with nohup so a dropped connection doesn't kill it):
#   nohup bash scripts/batch_12h.sh > batch_12h.log 2>&1 &
#   disown
# Check progress any time with:  tail -f batch_12h.log
set -uo pipefail

echo "==== $(date '+%F %T') exp2 baseline 2015-01-02 ===="
bash scripts/exp2_survival_sweep.sh --days 20150102 --seeds 30 --et 11:00:00 --variant baseline \
  --out-dir exp2_results/batch12_baseline_20150102

echo "==== $(date '+%F %T') exp2 baseline 2015-01-09 ===="
bash scripts/exp2_survival_sweep.sh --days 20150109 --seeds 30 --et 11:00:00 --variant baseline \
  --out-dir exp2_results/batch12_baseline_20150109

echo "==== $(date '+%F %T') exp2 reanchor 2015-01-16 ===="
bash scripts/exp2_survival_sweep.sh --days 20150116 --seeds 30 --et 11:00:00 --variant reanchor \
  --out-dir exp2_results/batch12_reanchor_20150116

echo "==== $(date '+%F %T') exp2 ss 2015-01-23 ===="
bash scripts/exp2_survival_sweep.sh --days 20150123 --seeds 30 --et 11:00:00 --variant ss \
  --out-dir exp2_results/batch12_ss_20150123

echo ""
echo "==== $(date '+%F %T') ALL DONE ===="
cat exp2_results/batch12_*/summary.md
