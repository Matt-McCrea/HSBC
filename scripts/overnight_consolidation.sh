#!/bin/bash
# overnight_consolidation.sh --- close the remaining evidence gaps for the write-up.
#
# THE GAP THIS MAINLY FILLS: the long-horizon result ("both candidates survive two hours where
# single-step collapses") currently rests on ONE day, 2015-01-29. It is the newest and most
# load-bearing claim in the paper and a single day is thin support for it. A second day roughly
# doubles its weight for ~5h of GPU.
#
# COST MODEL (measured, not guessed --- per-cell time scales with the SIMULATED window):
#     DDIM-10 :  30min window ~25min | 60min ~78min | 2h ~156min
#     DDPM-100:  roughly 3.5x the DDIM-10 figure at the same window
# Always --dry-run first; it prints the real total.
#
# Usage:
#   bash scripts/overnight_consolidation.sh --dry-run
#   bash scripts/overnight_consolidation.sh                      # ~6.8h
#   bash scripts/overnight_consolidation.sh --confirm-arm "--depth-drift 0.25 --depth-drift-phi 0.9998 --book-target-thick 2.0 --book-cancel-rate 0.5"
#       ^ adds a 2h confirmation of the winning sweep arm (+2.6h). Use the arm's exact flags.
set -uo pipefail

TICKER="INTC"; SEED="30"
WIN="--depth-noise 0.3 --size-reshape --type-decode prior"
HEAD="0.69_epoch=4"; OTHER="0.724_epoch=0"; DDPM_CKPT="0.681_epoch=3"
CONFIRM_ARM=""; DRY=0
OUT_DIR="overnight/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --dry-run) DRY=1; shift;;
  --confirm-arm) CONFIRM_ARM="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# tag | ckpt | date | st | et | sampler | nsteps | seed | est_min | extra flags
PLAN=(
  # --- P1: second day for the long-horizon claim. The single highest-value pair. ---
  "lh2_ss_e4_0130|$HEAD|20150130|10:00:00|12:00:00|DDIM|10|30|156|$WIN"
  "lh2_0724_0130|$OTHER|20150130|10:00:00|12:00:00|DDIM|10|30|156|$WIN"
  # --- P2: completes the seed table (the one cell that missed the last window) ---
  "seed32_0724|$OTHER|20150130|09:30:00|10:00:00|DDIM|10|32|25|$WIN"
  # --- P3: our DDPM-100 on 0129, giving DDPM-vs-DDPM on BOTH days in the replication table
  #         rather than only 2015-01-30. Pre-fix checkpoint, as the replication requires. ---
  "repl_ddpm100_0129|$DDPM_CKPT|20150129|09:30:00|10:00:00|DDPM|100|30|88|"
)
[[ -n "$CONFIRM_ARM" ]] && PLAN+=("confirm_arm_2h|$HEAD|20150129|10:00:00|12:00:00|DDIM|10|30|156|$WIN $CONFIRM_ARM")

if [[ "$DRY" == "1" ]]; then
  TOT=0
  printf '%-22s %-16s %-10s %-14s %-10s %s\n' TAG CKPT DATE WINDOW SAMPLER EST
  for c in "${PLAN[@]}"; do IFS='|' read -r t ck d st et sm ns sd est ex <<< "$c"
    printf '%-22s %-16s %-10s %-14s %-10s ~%smin\n' "$t" "$ck" "$d" "$st-$et" "$sm-$ns" "$est"
    TOT=$((TOT+est)); done
  echo ""; echo "cells: ${#PLAN[@]}   TOTAL: ~$((TOT/60))h $((TOT%60))m"
  exit 0
fi

mkdir -p "$OUT_DIR/logs"; mkdir -p overnight
ln -sfn "$(basename "$OUT_DIR")" overnight/latest
SUM="$OUT_DIR/summary.md"; : > "$SUM"
echo "| cell | sampler | window | seed | wall-clock | csv |" >> "$SUM"
echo "|---|---|---|---|---|---|" >> "$SUM"

find_ckpt () {
  local hit
  for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
           data/checkpoints/TRADES_other_recovered data/checkpoints; do
    hit=$(ls "$d"/*"$1"*.ckpt 2>/dev/null | head -1); [[ -n "$hit" ]] && { echo "$hit"; return 0; }
  done; return 1
}
ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }

echo "=== preflight ==="
pgrep -f "main.py" > /dev/null && { echo "!! training running --- kill it first"; exit 1; }
pgrep -f "drift_persistence_sweep" > /dev/null && { echo "!! drift sweep still running --- wait or kill it"; exit 1; }
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py > "$OUT_DIR/logs/qt.txt" 2>&1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null || true

for c in "${PLAN[@]}"; do
  IFS='|' read -r TAG FRAG D ST ET SAMP NST SD EST EXTRA <<< "$c"
  DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "   SKIP $TAG"; continue; }
  CK=$(find_ckpt "$FRAG") || { echo "!! ckpt '$FRAG' not found -- skipping $TAG"; continue; }
  CAP=$(( EST * 60 * 2 ))          # 100% headroom: never kill a cell just short of finishing
  REALP="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$D")_${ET//:/-}_${SEED}/processed_orders.csv"
  [[ -f "$REALP" ]] || python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
      -st "$ST" -et "$ET" > "$OUT_DIR/logs/real_${D}_${ET//:/-}.txt" 2>&1

  echo "-- $TAG   [$SAMP-$NST $D $ST-$ET seed=$SD, est ~${EST}min, cap $((CAP/60))min]"
  T0=$(date +%s)
  if ! timeout -k 30 "$CAP" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
        -st "$ST" -et "$ET" -d True -m TRADES -type "$SAMP" -nsteps "$NST" -eta 0.0 \
        --ckpt-path "$CK" -seed "$SD" $EXTRA > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    RC=$?; SECS=$(( $(date +%s) - T0 ))
    W="ERROR rc=$RC"; [[ $RC -eq 124 || $RC -eq 137 ]] && W="TIMEOUT ${CAP}s"
    echo "   $W after $((SECS/60))m"
    echo "| $TAG | $SAMP-$NST | $D $ST-$ET | $SD | **$W** | |" >> "$SUM"; continue
  fi
  SECS=$(( $(date +%s) - T0 ))
  CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)
  echo "| $TAG | $SAMP-$NST | $D $ST-$ET | $SD | $((SECS/60))m $((SECS%60))s | ${CSV:-none} |" >> "$SUM"
  touch "$DONE"; echo "   done $((SECS/60))m $((SECS%60))s"
done

echo ""; echo "=== COMPLETE. Summary: $SUM ==="
cat "$SUM"
