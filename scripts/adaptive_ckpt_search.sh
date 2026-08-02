#!/bin/bash
# adaptive_ckpt_search.sh — find a checkpoint that's stable across every trading day.
#
# Kills any other GPU-using process first (no contention, no confusion about what's running).
# Tests checkpoints in priority order (known-good baselines first). For each checkpoint,
# sweeps days with the two hardest known-drift days (0107, 0129) FIRST, then the rest of the
# month, each capped at 40 min. The moment a checkpoint times out on ANY day, it's marked
# FAILED and abandoned immediately — no point confirming a checkpoint we already know fails,
# better to spend that time on the next candidate. The first checkpoint to clear every day
# is the winner and the search stops there.
#
# This is a fast triage pass (pass/fail + basic uniq_mid/ret1s_std), not the full flow_mix/
# LOB-Bench comparison — run those afterward on the winning checkpoint's saved CSVs.
#
# Usage (LONG — full month, every checkpoint currently in the dir, newest-first, ~up to 2 days
# worst case):
#   bash scripts/adaptive_ckpt_search.sh --out-tag long
#
# Usage (SHORT — one hard day, a chosen subset by substring match on the real filenames present,
# fits a ~2h window):
#   bash scripts/adaptive_ckpt_search.sh --out-tag short --days "20150107" \
#     --ckpts "epoch=2 epoch=3" --cap-secs 1800
#
# Run both at once on two different machines (they share the project filestore) — --out-tag
# keeps their output dirs and "latest" symlinks distinct so they don't clobber each other.
set -uo pipefail

CKPT_DIR="data/checkpoints/TRADES"
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"; SEED="30"
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
CAP_SECS=2400   # 40 min per-day cap
CKPTS_ARG=""    # space-separated substrings to match against ckpt filenames, in priority order
DAYS_ARG=""     # space-separated days; empty = full month
OUT_TAG="run"

while [[ $# -gt 0 ]]; do case "$1" in
  --cap-secs) CAP_SECS="$2"; shift 2;;
  --ckpts) CKPTS_ARG="$2"; shift 2;;
  --days) DAYS_ARG="$2"; shift 2;;
  --out-tag) OUT_TAG="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# auto-discover whatever's actually in the checkpoint dir right now — newest-first by mtime, so
# the most recently trained epochs (most likely to reflect the retrain's current behavior) go
# first. --ckpts filters this real list by substring; it no longer refers to a fixed hardcoded set.
mapfile -t ALL_CKPTS < <(cd "$CKPT_DIR" 2>/dev/null && ls -t -- *.ckpt 2>/dev/null)
[[ ${#ALL_CKPTS[@]} -gt 0 ]] || { echo "!! no .ckpt files in $CKPT_DIR"; exit 1; }

if [[ -n "$CKPTS_ARG" ]]; then
  PRIORITY_CKPTS=()
  for SUB in $CKPTS_ARG; do
    for F in "${ALL_CKPTS[@]}"; do
      [[ "$F" == *"$SUB"* ]] && PRIORITY_CKPTS+=("$F")
    done
  done
else
  PRIORITY_CKPTS=("${ALL_CKPTS[@]}")
fi

# hardest/known-drift days first, so a bad checkpoint fails fast rather than late
if [[ -n "$DAYS_ARG" ]]; then DAYS=($DAYS_ARG)
else DAYS=(20150107 20150129 20150102 20150105 20150106 20150108 20150109 20150112 20150113 20150114 20150115 20150116 20150120 20150121 20150122 20150123 20150126 20150127 20150128 20150130); fi

OUT_DIR="ckpt_search/${OUT_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/logs"
mkdir -p ckpt_search
ln -sfn "$(basename "$OUT_DIR")" "ckpt_search/latest-${OUT_TAG}"
STATUS="$OUT_DIR/STATUS.txt"
PROGRESS="$OUT_DIR/progress.txt"
: > "$PROGRESS"

echo "=== killing any other GPU-using processes ==="
pkill -f abides.py 2>/dev/null
pkill -f checkpoint_stability.sh 2>/dev/null
pkill -f single_ckpt 2>/dev/null
pkill -f main.py 2>/dev/null
# kill stale prior invocations of THIS script only — exclude our own PID ($$) and our parent
# ($PPID), since "pkill -f adaptive_ckpt_search.sh" would otherwise match and kill ourselves
# (our own command line contains that string) before doing anything else.
pgrep -f adaptive_ckpt_search.sh | grep -vE "^($$|$PPID)\$" | xargs -r kill 2>/dev/null
sleep 2
echo "GPU state now:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  (nvidia-smi unavailable)"

touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRECHECK" == "True True" ]] || { echo "!! flags not True True — refusing. Got: $PRECHECK"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }

movemetric () {
  python3 - "$1" <<'PY' 2>/dev/null
import sys, numpy as np, pandas as pd, datetime as dt
try:
    df = pd.read_csv(sys.argv[1]); df["dt"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    for c in ("ask_price_1","bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.ask_price_1.abs()<9e9)&(df.bid_price_1.abs()<9e9)&(df.ask_price_1>0)&(df.bid_price_1>0)]
    df["mid"] = (df.ask_price_1+df.bid_price_1)/2/10000.0
    df = df.dropna(subset=["dt","mid"]); df = df[df.dt.dt.time >= dt.time(9,45,0)]
    s = df.set_index("dt")["mid"].resample("1s").last().ffill().dropna(); r = np.log(s).diff().dropna()
    print("uniq_mid={}  ret1s_std={:.2f}bp  mid_range_tk={:.0f}".format(
        df.mid.round(3).nunique(), r.std()*1e4, (df.mid.max()-df.mid.min())*100))
except Exception as e:
    print("movemetric failed:", e)
PY
}

write_status () {
  { echo "=== ADAPTIVE CHECKPOINT SEARCH — $(date '+%F %T') ==="
    echo "$1"
    echo "cap: ${CAP_SECS}s per day  |  out: $OUT_DIR"
    echo ""
    echo "--- progress so far ---"
    cat "$PROGRESS" 2>/dev/null
  } > "$STATUS"
}

echo "checkpoints to try, in order: ${PRIORITY_CKPTS[*]}"
echo "days per checkpoint (${#DAYS[@]}), hardest first: ${DAYS[*]}"
write_status "starting"

WINNER=""
for CKNAME in "${PRIORITY_CKPTS[@]}"; do
  CK="$CKPT_DIR/$CKNAME"
  TAG=$(echo "$CKNAME" | sed -E 's/^val_ema=//; s/_INTC.*//')
  if [[ ! -f "$CK" ]]; then
    echo "!! missing $CK, skipping" | tee -a "$PROGRESS"; continue
  fi
  {
    echo ""
    echo "########## testing checkpoint $TAG ##########"
  } | tee -a "$PROGRESS"
  write_status "testing checkpoint: $TAG"
  FAILED=0
  for D in "${DAYS[@]}"; do
    LOGF="$OUT_DIR/logs/${TAG}__${D}.txt"
    write_status "checkpoint: $TAG   day: $D   (cap ${CAP_SECS}s)"
    T0=$(date +%s)
    if timeout -k 15 "$CAP_SECS" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
          -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK" -seed "$SEED" $BASE \
          > "$LOGF" 2>&1; then
      SECS=$(( $(date +%s) - T0 ))
      CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$LOGF" | tail -1)
      RES="  $D  OK  (${SECS}s)"
      [[ -n "$CSV" && -f "$CSV" ]] && RES="$RES   $(movemetric "$CSV")"
      echo "$RES" | tee -a "$PROGRESS"
      write_status "checkpoint: $TAG   day: $D   done (${SECS}s)"
    else
      RC=$?
      SECS=$(( $(date +%s) - T0 ))
      if [[ $RC -eq 124 || $RC -eq 137 ]]; then
        echo "  $D  TIMEOUT after ${SECS}s — likely unstable. ABANDONING $TAG, moving to next checkpoint." | tee -a "$PROGRESS"
      else
        echo "  $D  ERROR (rc=$RC) after ${SECS}s — see logs/${TAG}__${D}.txt. ABANDONING $TAG." | tee -a "$PROGRESS"
      fi
      FAILED=1
      write_status "checkpoint: $TAG   ABANDONED at day $D"
      break
    fi
  done
  if [[ "$FAILED" == "0" ]]; then
    {
      echo ""
      echo "*** WINNER: $TAG cleared all ${#DAYS[@]} days within ${CAP_SECS}s each! ***"
    } | tee -a "$PROGRESS"
    WINNER="$TAG"
    write_status "WINNER FOUND: $TAG — cleared all ${#DAYS[@]} days"
    break
  fi
done

if [[ -z "$WINNER" ]]; then
  echo "" | tee -a "$PROGRESS"
  echo "!! no checkpoint cleared every day — see $PROGRESS for how far each got." | tee -a "$PROGRESS"
  write_status "SEARCH COMPLETE — no full winner. Check progress above for the best partial performer."
else
  write_status "SEARCH COMPLETE — WINNER: $WINNER"
fi

echo ""
echo "=== SEARCH COMPLETE — live view any time: cat ckpt_search/latest-${OUT_TAG}/STATUS.txt ==="
cat "$STATUS"
