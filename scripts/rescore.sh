#!/bin/bash
# rescore.sh — re-run rl_execution.survival_metrics against every generated run already on disk
# that hasn't been scored yet. No GPU cost, pure pandas over CSVs that already exist -- use this
# instead of re-typing long paths, or instead of re-running the (expensive) generation whenever
# scoring failed/crashed/was skipped for a run that itself succeeded.
#
# ONE COMMAND, ZERO ARGUMENTS:
#   bash scripts/rescore.sh
set -uo pipefail
REAL_SEED="30"   # every real-replay comparison in this pipeline uses seed 30 (see exp2_survival_sweep.sh)
COUNT=0; SCORED=0; FAILED=0

for GEN_DIR in ABIDES/log/world_agent_*_DDPM_*; do
  [[ -d "$GEN_DIR" ]] || continue
  GEN_CSV="$GEN_DIR/processed_orders.csv"
  [[ -f "$GEN_CSV" ]] || continue
  SCORE_JSON="$GEN_DIR/score.json"
  [[ -f "$SCORE_JSON" ]] && continue   # already scored -- rescore.sh is idempotent, safe to rerun anytime

  BASE=$(basename "$GEN_DIR")
  if [[ "$BASE" =~ world_agent_([A-Za-z]+)_([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}-[0-9]{2}-[0-9]{2})_([0-9]+)_ ]]; then
    SYM="${BASH_REMATCH[1]}"; DATE="${BASH_REMATCH[2]}"; ET="${BASH_REMATCH[3]}"; SEED="${BASH_REMATCH[4]}"
  else
    echo "SKIP (dir name doesn't match world_agent_<SYM>_<DATE>_<ET>_<SEED>_...): $BASE"
    continue
  fi
  REAL_CSV="ABIDES/log/market_replay_${SYM}_${DATE}_${ET}_${REAL_SEED}/processed_orders.csv"
  COUNT=$((COUNT + 1))
  if [[ ! -f "$REAL_CSV" ]]; then
    echo "MISSING real replay for $BASE -- expected $REAL_CSV"
    FAILED=$((FAILED + 1)); continue
  fi

  echo "-- $BASE  (seed=$SEED)"
  if python -m rl_execution.survival_metrics --gen "$GEN_CSV" --real "$REAL_CSV" --out "$SCORE_JSON"; then
    SCORED=$((SCORED + 1))
  else
    echo "   FAILED -- see error above"
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

echo "checked: $COUNT  scored: $SCORED  failed: $FAILED"
[[ "$COUNT" -eq 0 ]] && echo "(nothing found to score -- either nothing's unscored, or no ABIDES/log/world_agent_*_DDPM_* dirs exist yet)"
