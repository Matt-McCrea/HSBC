#!/bin/bash
# headline_vs_trades.sh --- OUR ten steps against THEIR hundred steps, like for like.
#
# THE CLAIM THIS SUPPORTS: our accelerated configuration beats the PUBLISHED TRADES model on
# LOB-Bench, at ~12x lower per-order cost. That is the dissertation's thesis, and the baseline
# it needs is TRADES's own released output --- not our own DDPM-100, which answers the much
# less interesting question of whether ten of our steps beat a hundred of our steps.
#
# WHAT ALREADY EXISTS: lob_bench_reanchored_75min/ has exactly this comparison on Intel
# 2015-01-30, 09:45-11:00, against their released file --- our DDIM-10 + fixes scores 0.276
# (mean Wasserstein excluding inter-arrival) against their released 0.798. But it is on
# checkpoint 0.627, the OLD frozen checkpoint, not the model we ship. This regenerates it on
# the adopted checkpoints so the headline describes the actual final model.
#
# WINDOW: 09:45-11:00 is not arbitrary --- it is the window their released CSVs cover, so gen,
# real and their output are all scored over the same 75 minutes. Any other window makes the
# comparison approximate.
#
# INTER-ARRIVAL IS EXCLUDED when scoring against their file: their released format carries only
# 0.1s timestamp resolution, which inflates that one metric. File artefact, not model behaviour.
# State that in the write-up rather than appearing to drop an inconvenient metric.
#
#   nohup bash scripts/headline_vs_trades.sh > headline.log 2>&1 &
set -uo pipefail

TICKER="INTC"; SEED="30"; ST="09:45:00"; ET="11:00:00"
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
DEADLINE="19:00"; CAP_SECS=5400
OUT_DIR="headline/$(date +%Y%m%d_%H%M%S)"

# ordered by value: SS epoch 4 is the leading candidate and 0130 is the day whose released
# comparison already sits in the replication table, so the first cell alone completes the claim.
PLAN=(
  "sse4_0130|0.69_epoch=4|20150130"
  "sse4_0129|0.69_epoch=4|20150129"
  "0724_0130|0.724_epoch=0|20150130"
  "0724_0129|0.724_epoch=0|20150129"
)

while [[ $# -gt 0 ]]; do case "$1" in
  --deadline) DEADLINE="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

DEADLINE_EPOCH=$(date -d "today $DEADLINE" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $DEADLINE" +%s)

find_ckpt () { local h; for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
    data/checkpoints/TRADES_other_recovered data/checkpoints; do
    h=$(ls "$d"/*"$1"*.ckpt 2>/dev/null | head -1); [[ -n "$h" ]] && { echo "$h"; return 0; }; done; return 1; }
ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }

mkdir -p "$OUT_DIR/logs" headline
ln -sfn "$(basename "$OUT_DIR")" headline/latest
PROG="$OUT_DIR/progress.txt"; : > "$PROG"

echo "=== preflight ==="
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py > "$OUT_DIR/logs/qt.txt" 2>&1
echo "window $ST-$ET  (matches TRADES's released coverage)" | tee -a "$PROG"

for c in "${PLAN[@]}"; do
  IFS='|' read -r TAG FRAG D <<< "$c"
  REMAIN=$(( DEADLINE_EPOCH - $(date +%s) ))
  [[ "$REMAIN" -lt "$CAP_SECS" ]] && { echo "  -- deadline, stopping before $TAG" | tee -a "$PROG"; break; }
  CK=$(find_ckpt "$FRAG") || { echo "!! ckpt '$FRAG' not found, skipping $TAG" | tee -a "$PROG"; continue; }

  # real replay over the SAME window, so LOB-Bench scores gen and real like for like
  REALP="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$D")_${ET//:/-}_${SEED}/processed_orders.csv"
  [[ -f "$REALP" ]] || { echo "   generating real replay $D"; \
    python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
      > "$OUT_DIR/logs/real_${D}.txt" 2>&1; }

  echo "[$(date +%T)] -- $TAG  ($(basename "$CK") $D $ST-$ET)" | tee -a "$PROG"
  T0=$(date +%s)
  if timeout -k 15 "$CAP_SECS" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
        -st "$ST" -et "$ET" -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" $BASE > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    S=$(( $(date +%s) - T0 )); CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)
    echo "   OK $((S/60))m  $CSV" | tee -a "$PROG"; echo "$CSV" >> "$OUT_DIR/csv_list.txt"
  else
    RC=$?; S=$(( $(date +%s) - T0 ))
    [[ $RC -eq 124 || $RC -eq 137 ]] && M=TIMEOUT || M="ERROR rc=$RC"
    echo "   $M after $((S/60))m --- continuing" | tee -a "$PROG"
  fi
done

echo ""; echo "=== done ==="; cat "$PROG"
echo ""
echo "NEXT (locally, LOB-Bench does not run on the remote):"
echo "  score each CSV against its matched real replay AND against the released TRADES file"
echo "  for the same day, excluding inter-arrival. Target to beat: their 0.798 on 2015-01-30,"
echo "  0.855 on 2015-01-29. Reference point on the old 0.627 checkpoint: 0.276."
