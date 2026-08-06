#!/bin/bash
# paper_figure_runs.sh --- overnight run plan producing every CSV the write-up still needs.
#
# Ordered by value, highest first, so an interrupted night loses the least. Resumable via .done
# sentinels: re-running skips completed cells. Each cell records its own wall-clock.
#
# WHAT IS ALREADY DONE AND NOT REPEATED HERE:
#   - winner 0.724 and SS epochs 2/3/4 across all 20 days at 30 min (cross-day stats: keep as is)
#   - the DDIM-1 / DDPM-100 replication CSVs and TRADES's released outputs
#
# THE GAPS THIS FILLS:
#   A. no long-horizon run of the winning model exists at all --- everything stops at 30 min, but
#      the single-step failure only appears around minute 73, so the fix is currently untested in
#      the regime where the thing it fixes actually happens
#   B. no DDPM-100 on the CURRENT checkpoint --- the "beats DDPM" claim rests on ckpt 0.627
#   C. no sampler ablation on 0.724 --- is depth-noise still needed post data-pipeline fixes?
#   D. no seed robustness on the final model
#
# Budget: ~8h of the ~9-10h window, leaving headroom. Estimates come from measured 30-min runs
# (mean 21 min at DDIM-10 on this checkpoint); DDPM-100 is ~3.5x DDIM-10 in wall-clock (not 10x ---
# the ABIDES matching engine cost is fixed per order and does not scale with denoising steps).
#
# Usage:
#   bash scripts/paper_figure_runs.sh                  # everything, in priority order
#   bash scripts/paper_figure_runs.sh --phases 1,2     # selected phases only
#   bash scripts/paper_figure_runs.sh --dry-run        # print the plan and exit
set -uo pipefail

TICKER="INTC"; SEED="30"
WIN="--depth-noise 0.3 --size-reshape --type-decode prior"      # the winning decode config
LEVER="--book-target-thick 2.0 --book-cancel-rate 0.5"          # long-horizon book-balancing cancel
CAP=10800                                                        # 3h per-cell cap
PHASES="1,2,3,4,5"; DRY=0
OUT_DIR="paper_runs/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --phases) PHASES="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  --cap) CAP="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"; mkdir -p paper_runs
ln -sfn "$(basename "$OUT_DIR")" paper_runs/latest
SUM="$OUT_DIR/summary.md"; : > "$SUM"
STATUS="$OUT_DIR/STATUS.txt"

have_phase () { [[ ",$PHASES," == *",$1,"* ]]; }

find_ckpt () {
  local frag="$1" hit
  for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
           data/checkpoints/TRADES_other_recovered data/checkpoints; do
    hit=$(ls "$d"/*"$frag"*.ckpt 2>/dev/null | head -1)
    [[ -n "$hit" ]] && { echo "$hit"; return 0; }
  done
  return 1
}
ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }

ensure_real () {   # <yyyymmdd> <st> <et>
  local D="$1" ST="$2" ET="$3"
  local rp="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$D")_${ET//:/-}_${SEED}/processed_orders.csv"
  [[ -f "$rp" ]] || python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
      -st "$ST" -et "$ET" > "$OUT_DIR/logs/real_${D}_${ET//:/-}.txt" 2>&1
  echo "$rp"
}

status () { { echo "=== PAPER FIGURE RUNS --- $(date '+%F %T') ==="; echo "$1";
              echo ""; echo "--- completed ---"; grep '^| ' "$SUM" 2>/dev/null; } > "$STATUS"; }

run () {   # <tag> <ckptfrag> <date> <st> <et> <sampler> <nsteps> <seed> <decode args...>
  local TAG="$1" FRAG="$2" D="$3" ST="$4" ET="$5" SAMP="$6" NST="$7" SD="$8"; shift 8
  local DEC="$*"
  local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "   SKIP $TAG"; return; }
  local CK; CK=$(find_ckpt "$FRAG") || { echo "!! ckpt '$FRAG' not found -- skipping $TAG"; return; }

  status "running: $TAG"
  echo "-- $TAG   [$SAMP-$NST seed=$SD $D $ST-$ET]"
  local REALP; REALP=$(ensure_real "$D" "$ST" "$ET")
  local T0; T0=$(date +%s)
  if ! timeout -k 30 "$CAP" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
        -st "$ST" -et "$ET" -d True -m TRADES -type "$SAMP" -nsteps "$NST" -eta 0.0 \
        --ckpt-path "$CK" -seed "$SD" $DEC > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    local RC=$?; local SECS=$(( $(date +%s) - T0 ))
    local why="ERROR rc=$RC"; [[ $RC -eq 124 || $RC -eq 137 ]] && why="TIMEOUT at ${CAP}s"
    echo "   $why after $((SECS/60))m"
    echo "| $TAG | $SAMP-$NST | $D $ST-$ET | seed $SD | **$why** | |" >> "$SUM"
    return
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)
  echo "| $TAG | $SAMP-$NST | $D $ST-$ET | seed $SD | $((SECS/60))m $((SECS%60))s | ${CSV:-none} |" >> "$SUM"
  touch "$DONE"
  echo "   done in $((SECS/60))m $((SECS%60))s"
}

if [[ "$DRY" == "1" ]]; then
  cat <<'PLAN'
PLAN (est. ~8h of a 9-10h window)

 P1  ~11m   sampler ablation on the winner, 30min 0129
            - DDIM-1 vanilla, DDIM-10 vanilla   -> does depth-noise still earn its place?
 P2  ~2.8h  long-horizon, 2h 0129 10:00-12:00   *** the critical gap ***
            - winner 0.724 base config
            - SS epoch 4 base config            -> both final-model candidates past minute 73
 P3  ~2.5h  DDPM-100 on the CURRENT checkpoint, 30min
            - 0.724 on 0130 and 0129            -> same-checkpoint acceleration claim
 P4  ~1.4h  long-horizon WITH book-balancing lever, 2h 0129
            - winner 0.724 + bt2.0/r0.5         -> does the winner still need the lever?
 P5  ~1.4h  seed robustness, 30min 0130, seeds 31/32
            - winner and SS epoch 4
PLAN
  exit 0
fi

# ---------------------------------------------------------------- preflight
echo "=== preflight ==="
if pgrep -f "main.py" > /dev/null; then
  echo "!! training is running --- kill it first, GPU contention invalidates every timing below."; exit 1
fi
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG    # every checkpoint below is post-fix
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py \
    > "$OUT_DIR/logs/build_targets.txt" 2>&1
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null || true
echo "| cell | sampler | window | seed | wall-clock | csv |" >> "$SUM"
echo "|---|---|---|---|---|---|" >> "$SUM"


# ================================================================ PHASE 1
if have_phase 1; then
  echo ""; echo "########## P1: sampler ablation on the winner (~11m) ##########"
  echo "# vanilla = no depth-noise/size-reshape/type-decode, isolating the sampler"
  run p1_ddim1_vanilla   0.724_epoch=0 20150129 09:30:00 10:00:00 DDIM 1  30
  run p1_ddim10_vanilla  0.724_epoch=0 20150129 09:30:00 10:00:00 DDIM 10 30
fi

# ================================================================ PHASE 2
if have_phase 2; then
  echo ""; echo "########## P2: LONG HORIZON --- the critical gap (~2.8h) ##########"
  echo "# 10:00-12:00 matches the window where single-step DDIM collapses (~min 73)"
  run p2_winner_2h    0.724_epoch=0 20150129 10:00:00 12:00:00 DDIM 10 30 $WIN
  run p2_ss_e4_2h     0.69_epoch=4  20150129 10:00:00 12:00:00 DDIM 10 30 $WIN
fi

# ================================================================ PHASE 3
if have_phase 3; then
  echo ""; echo "########## P3: DDPM-100 on the current checkpoint (~2.5h) ##########"
  echo "# gives a same-checkpoint acceleration comparison; the existing one is on ckpt 0.627"
  run p3_ddpm100_0130 0.724_epoch=0 20150130 09:30:00 10:00:00 DDPM 100 30 $WIN
  run p3_ddpm100_0129 0.724_epoch=0 20150129 09:30:00 10:00:00 DDPM 100 30 $WIN
fi

# ================================================================ PHASE 4
if have_phase 4; then
  echo ""; echo "########## P4: long horizon WITH the book-balancing lever (~1.4h) ##########"
  echo "# the lever was validated at 90min on ckpt 0.627 --- untested on the winner"
  run p4_winner_2h_lever 0.724_epoch=0 20150129 10:00:00 12:00:00 DDIM 10 30 $WIN $LEVER
fi

# ================================================================ PHASE 5
if have_phase 5; then
  echo ""; echo "########## P5: seed robustness on the final models (~1.4h) ##########"
  for SD in 31 32; do
    run "p5_winner_s${SD}" 0.724_epoch=0 20150130 09:30:00 10:00:00 DDIM 10 "$SD" $WIN
    run "p5_ss_e4_s${SD}"  0.69_epoch=4  20150130 09:30:00 10:00:00 DDIM 10 "$SD" $WIN
  done
fi

status "COMPLETE"
echo ""; echo "=========================================================="
echo " DONE. Summary: $SUM"
echo " Live status at any point:  cat paper_runs/latest/STATUS.txt"
echo "=========================================================="
cat "$SUM"
