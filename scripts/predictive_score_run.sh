#!/bin/bash
# predictive_score_run.sh — run the TRADES predictive score (MAE) over every generated config
# present for a given day, against that day's real replay. CPU-friendly but much faster on GPU;
# no training of the diffusion model, only the small evaluation LSTMs. Resolves the real
# reference and enumerates the generated dirs automatically, then calls predictive_batch.
#
# Usage:  bash scripts/predictive_score_run.sh                        # Jan-30, 30-min, auto
#         bash scripts/predictive_score_run.sh --date 20150122 --et 10:00:00 --seeds 5
#         bash scripts/predictive_score_run.sh --real path/to/real/processed_orders.csv
set -uo pipefail
TICKER="INTC"; DATE="20150130"; ET="10:00:00"; ST="09:30:00"; SEEDS="3"; LOOKBACK="100"
REAL=""; EXTRA=""
while [[ $# -gt 0 ]]; do case "$1" in
  --date) DATE="$2"; shift 2;; --et) ET="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;;
  --lookback) LOOKBACK="$2"; shift 2;; --real) REAL="$2"; shift 2;;
  *) EXTRA="$EXTRA $1"; shift;; esac; done

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
DD="$(ymd_dash "$DATE")"; ED="$(etdash "$ET")"
OUT="predictive_scores/${DATE}_${ED}"

# resolve the real reference (market replay processed_orders for the day)
if [[ -z "$REAL" ]]; then
  REAL="ABIDES/log/market_replay_${TICKER}_${DD}_${ED}_30/processed_orders.csv"
fi
if [[ ! -f "$REAL" ]]; then
  echo "!! real reference not found: $REAL"
  echo "   generate it first, e.g.:  python ABIDES/abides.py -c world_agent_sim -t $TICKER -date $DATE -st $ST -et $ET"
  exit 1
fi

# enumerate generated configs for this day (any world_agent dir with a processed_orders.csv)
mapfile -t DIRS < <(ls -d ABIDES/log/world_agent_${TICKER}_${DD}_${ED}_*/ 2>/dev/null)
GENARGS=()
for d in "${DIRS[@]}"; do
  csv="${d}processed_orders.csv"
  [[ -f "$csv" ]] || continue
  # label = sampler+steps+config suffix, i.e. strip the known "world_agent_..._30_" prefix and
  # the "val_ema=X.XXX_" checkpoint id from the middle (keeps DDIM/DDPM + steps distinct).
  label="$(basename "$d")"
  label="${label#world_agent_${TICKER}_${DD}_${ED}_30_}"
  label="$(echo "$label" | sed -E 's/val_ema=[0-9.]+_?//; s/_$//')"
  [[ -z "$label" ]] && label="base"
  GENARGS+=(--gen "${label}=${csv}")
done
if [[ ${#GENARGS[@]} -eq 0 ]]; then
  echo "!! no generated processed_orders.csv found under ABIDES/log/world_agent_${TICKER}_${DD}_${ED}_*/"; exit 1
fi

echo "real: $REAL"
echo "configs: ${#GENARGS[@]}   out: $OUT   seeds=$SEEDS lookback=$LOOKBACK"
# shellcheck disable=SC2086
python -m evaluation.quantitative_eval.predictive_batch \
  --real "$REAL" "${GENARGS[@]}" \
  --out-dir "$OUT" --seeds "$SEEDS" --lookback "$LOOKBACK" $EXTRA
