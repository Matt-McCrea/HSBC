#!/bin/bash
# session3_batch.sh — the whole 3-day session plan, chained and unattended: finish INTC Exp 4,
# then train + survival-check + teacher-force TSLA baseline and reanchor (tier 1+2), then TSLA ss
# as a lower-priority tier 3 if there's still time. Sequential (single GPU). Fully resumable: every
# training step is skipped if already pinned in analysis/model_under_test.md, every survival check
# uses a fixed --out-dir so exp2_survival_sweep.sh's own .done sentinels handle the rest, every
# Exp 3 step is skipped if its output file already exists (same idioms as
# train_three_variants.sh/overnight_batch.sh/batch_12h.sh).
#
# Exp 0 for TSLA is NOT in here -- it's CPU-only and should run in its own parallel window (given
# separately), not serialized behind GPU work it doesn't compete with.
#
# ONE COMMAND (nohup so a dropped connection doesn't kill it):
#   nohup bash scripts/session3_batch.sh > session3_batch.log 2>&1 &
#   disown
# Check progress any time with:  tail -f session3_batch.log
set -uo pipefail

train_and_pin () {  # train_and_pin <variant>
  local V="$1" VU
  VU=$(echo "$V" | tr '[:lower:]' '[:upper:]')
  if grep -q "^CKPT_PATH_${VU}=" analysis/model_under_test.md 2>/dev/null; then
    echo "SKIP train $V -- already pinned"; return
  fi
  echo "==== $(date '+%F %T') train $V ===="
  bash scripts/train_variant.sh "$V"
  bash scripts/pin_trained_variant.sh "$V"
}

survival_check () {  # survival_check <variant> <day> <et>
  local V="$1" D="$2" ET="$3"
  local OUT="exp2_results/session3_${V}_${D}"
  echo "==== $(date '+%F %T') exp2 $V $D ($ET) ===="
  bash scripts/exp2_survival_sweep.sh --days "$D" --seeds 30 --et "$ET" --variant "$V" --out-dir "$OUT"
}

teacher_forced_check () {  # teacher_forced_check <variant>
  local V="$1"
  local OUT="exp3_results/session3_${V}"
  if ls "$OUT"/open_loop_*.json >/dev/null 2>&1; then
    echo "SKIP exp3 $V -- already done"; return
  fi
  echo "==== $(date '+%F %T') exp3 $V ===="
  bash scripts/exp3_teacher_forced.sh --variant "$V" --out-dir "$OUT"
}

# ---- TIER 0: finish INTC Exp 4 (one more day, new checkpoint pair) ----
echo "==== $(date '+%F %T') exp4 baseline INTC 2015-01-16 ===="
bash scripts/exp4_checkpoint_sweep.sh --ckpt-dir data/checkpoints/TRADES_baseline --day 20150116 --seeds 30 --et 11:00:00 \
  --out-dir exp4_results/session3_baseline_20150116

# NOTE: the quick 30-min sanity check below uses day 2015-01-08 for every variant -- deliberately
# NOT reused in any of the 90-min loops (out-dir/.done naming doesn't encode horizon, so the same
# day at two different horizons would collide and the second run would silently skip).

# ---- TIER 1: tsla_baseline ----
train_and_pin tsla_baseline
survival_check tsla_baseline 20150108 10:00:00   # smoke-equivalent, 30min, cheap sanity check first
for D in 20150102 20150107 20150115 20150122 20150130; do
  survival_check tsla_baseline "$D" 11:00:00       # 90-min horizon, matches INTC's own numbers
done
teacher_forced_check tsla_baseline

# ---- TIER 2: tsla_reanchor -- does "reanchor fails earlier" generalize to TSLA? ----
train_and_pin tsla_reanchor
survival_check tsla_reanchor 20150108 10:00:00
for D in 20150105 20150109 20150116 20150123; do
  survival_check tsla_reanchor "$D" 11:00:00
done
teacher_forced_check tsla_reanchor

# ---- TIER 3 (lower priority -- only reached if tiers 0-2 leave enough budget): tsla_ss ----
train_and_pin tsla_ss
survival_check tsla_ss 20150108 10:00:00
for D in 20150106 20150113; do
  survival_check tsla_ss "$D" 11:00:00
done
teacher_forced_check tsla_ss

echo ""
echo "==== $(date '+%F %T') ALL TIERS ATTEMPTED ===="
echo "Read: exp2_results/session3_*/summary.md and exp3_results/session3_*/open_loop_*.json"
echo ""
echo "!! BEFORE ending this session: back up trained checkpoints to git (*.ckpt is LFS-tracked)."
echo "!!   git add data/checkpoints/TRADES_tsla_baseline data/checkpoints/TRADES_tsla_reanchor data/checkpoints/TRADES_tsla_ss"
echo "!!   git commit -m 'Back up TSLA checkpoints before a month-long gap'"
echo "!!   git push origin rl-execution"
