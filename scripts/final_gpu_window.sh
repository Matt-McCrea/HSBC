#!/bin/bash
# final_gpu_window.sh --- unattended plan for the LAST GPU window (hard end 20:00).
#
# WHAT IS ACTUALLY OPEN. Both final-model candidates are already fully evidenced on stability:
# 0.724_epoch=0, SS epoch 3 and SS epoch 4 each cleared all 20 trading days
# (analysis/appendix_checkpoint_evidence.md). Full-month LOB-Bench exists for DDIM-10 on both
# lineages (0.724: 0.468; SS e4: 0.346). Seeds are done. Long-horizon now has two days.
#
# The weak link is the CORE ACCELERATION CLAIM. "10 steps beat 100" currently rests on a SINGLE
# day: INTC 2015-01-30, grand-mean Wasserstein 0.447 (DDIM-10) against 0.575 (DDPM-100). Every
# other headline number in the write-up is a 20-day figure. A reviewer will go straight at the
# one-day comparison, and it is the claim the whole dissertation is built on.
#
# So this window buys DDPM-100 days. Each day added moves the central result from anecdote toward
# a distribution.
#
# DESIGN NOTES
#   * DDPM-100 runs on 0.724_epoch=0, NOT on an SS checkpoint. Scheduled sampling was trained
#     against its own DDIM-10 rollouts, so running DDPM-100 on it is not a clean step-count
#     ablation. 0.724 is the vanilla-trained lineage and already has the matching DDIM-10 month.
#   * Same window (09:30-10:00), same seed, same decode flags as the DDIM-10 month, so the only
#     difference between the two rows is sampler and step count.
#   * The amplitude sweep is allowed three cells first (0.05/0.08/0.12). With the existing dd=0.25
#     point that is four points on the response curve --- enough to present persistence as a
#     quantified diagnostic. Precise calibration is only needed if the knob ships, and it should
#     not: fitted on 60 min of one day, to the metric it is scored on, and adopting it would
#     invalidate the 20-day stability record with no time left to re-earn it.
#
# Run NOW, alongside the sweep. It sleeps until handover.
#   nohup bash scripts/final_gpu_window.sh > final_window.log 2>&1 &
#   disown
#
# Watch:  cat ddpm_month/latest/STATUS.txt
set -uo pipefail

CUT_AFTER="amp0.12"        # last sweep cell to allow before handover
DEADLINE="19:30"           # never START a day that cannot finish by here; hard GPU end is 20:00
CAP_SECS=6000              # 100 min/day. DDPM-100 is ~3.5x DDIM-10 at the same window.
CKPT_FRAG="0.724_epoch=0"
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"; SEED="30"
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
POLL=120
SKIP_WAIT=0

# The five days still outstanding after the 2026-08-08 run. The other fifteen completed and are
# under ddpm_month/<earlier timestamp>/ --- do not re-run them, tar both directories when pulling.
DAYS=(20150123 20150126 20150127 20150128 20150106)

# Full month, for when this is next run from scratch. 0129/0130 lead because both have an
# existing DDIM-10 comparison AND TRADES's own released output, so an early cutoff still leaves
# the most useful days in hand:
#   20150129 20150130 20150107 20150102 20150105 20150108 20150109 20150112 20150113 20150114
#   20150115 20150116 20150120 20150121 20150122 20150123 20150126 20150127 20150128 20150106

while [[ $# -gt 0 ]]; do case "$1" in
  --cut-after) CUT_AFTER="$2"; shift 2;;
  --deadline) DEADLINE="$2"; shift 2;;
  --ckpt) CKPT_FRAG="$2"; shift 2;;
  --days) DAYS=($2); shift 2;;
  --now) SKIP_WAIT=1; shift;;           # skip the sweep entirely, start DDPM immediately
  --cap-secs) CAP_SECS="$2"; shift 2;;  # hard kill per day; also the INITIAL headroom estimate
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# --deadline none: run until the system takes the GPU away. Safe here because every day writes
# its CSV and progress row on completion, so being killed mid-day costs only the day in flight.
# --cap-secs 0: no per-day timeout either. The one thing this gives up is protection against a
# hung day, which would then block every remaining day rather than being cut loose.
if [[ "$DEADLINE" == "none" ]]; then
  DEADLINE_EPOCH=0
else
  DEADLINE_EPOCH=$(date -d "today $DEADLINE" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $DEADLINE" +%s)
fi

echo "=== final GPU window ==="
echo "handover after sweep cell : $CUT_AFTER   (--now to skip)"
echo "then                      : DDPM-100 month on $CKPT_FRAG, ${CAP_SECS}s/day cap"
echo "stop starting days at     : $DEADLINE"
echo ""

# ---- phase 1: let the sweep clear three amplitude cells ----
if [[ "$SKIP_WAIT" == "0" ]]; then
  while true; do
    [[ -f drift_sweep/latest/logs/.done_${CUT_AFTER} ]] && { echo "[$(date +%T)] $CUT_AFTER done --- handing over"; break; }
    if ! pgrep -f drift_persistence_sweep > /dev/null; then
      echo "[$(date +%T)] sweep not running and $CUT_AFTER not done --- proceeding"; break; fi
    if [[ $(date +%s) -ge $((DEADLINE_EPOCH - CAP_SECS)) ]]; then
      echo "[$(date +%T)] deadline pressure --- cutting the sweep now"; break; fi
    echo "[$(date +%T)] sweep running, cells done: $(ls drift_sweep/latest/logs/.done_* 2>/dev/null | wc -l | tr -d ' ')"
    sleep "$POLL"
  done
  echo "[$(date +%T)] stopping the amplitude sweep (completed cells are kept)"
  pkill -f drift_persistence_sweep 2>/dev/null; sleep 2
  pkill -f abides.py 2>/dev/null; sleep 5
  ls drift_sweep/latest/logs/.done_* 2>/dev/null | sed 's/.*\.done_/  kept: /'
fi

# ---- phase 2: DDPM-100 across the month ----
find_ckpt () {
  local hit
  for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
           data/checkpoints/TRADES_other_recovered data/checkpoints; do
    hit=$(ls "$d"/*"$1"*.ckpt 2>/dev/null | head -1); [[ -n "$hit" ]] && { echo "$hit"; return 0; }
  done; return 1
}

OUT_DIR="ddpm_month/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/logs" ddpm_month
ln -sfn "$(basename "$OUT_DIR")" ddpm_month/latest
STATUS="$OUT_DIR/STATUS.txt"; PROG="$OUT_DIR/progress.txt"; : > "$PROG"

touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
CK=$(find_ckpt "$CKPT_FRAG") || { echo "!! ckpt '$CKPT_FRAG' not found"; exit 1; }
echo "ckpt: $(basename "$CK")" | tee -a "$PROG"

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
except Exception as e: print("movemetric failed:", e)
PY
}
write_status () { { echo "=== DDPM-100 MONTH --- $(date '+%F %T') ==="; echo "$1"
    echo "ckpt $(basename "$CK")  cap ${CAP_SECS}s  deadline $DEADLINE"; echo ""; cat "$PROG"; } > "$STATUS"; }

DONE_N=0; MAXSECS=0
for D in "${DAYS[@]}"; do
  REMAIN=$(( DEADLINE_EPOCH - $(date +%s) ))
  # Headroom must come from MEASURED day times, not the static cap. The cap is sized for the
  # worst case and is a kill-switch; using it as the scheduling threshold reserves far more
  # time than a day actually needs and silently drops days that would have fitted. (This cost
  # 5 days of a 20-day month on 2026-08-08: cap 6000s against a real 1350s/day, so it refused
  # to start a 22-minute job with 91 minutes left.) Once two days have completed, require
  # 1.25x the slowest observed instead.
  NEED=$CAP_SECS
  [[ "$DONE_N" -ge 2 && "$MAXSECS" -gt 0 ]] && NEED=$(( MAXSECS * 5 / 4 ))
  if [[ "$DEADLINE_EPOCH" -gt 0 && "$REMAIN" -lt "$NEED" ]]; then
    echo "  -- deadline: ${REMAIN}s left < ${NEED}s needed (slowest day ${MAXSECS}s), stopping before $D" | tee -a "$PROG"
    write_status "STOPPED at deadline before $D"; break
  fi
  write_status "running $D  (${DONE_N} days done, ${REMAIN}s left)"
  echo "[$(date +%T)] -- $D" ; T0=$(date +%s)
  LOGF="$OUT_DIR/logs/${D}.txt"
  # ${TO[@]+"${TO[@]}"} not "${TO[@]}": under `set -u`, bash <4.4 treats an EMPTY array
  # expansion as an unbound variable and aborts. This form expands to nothing when unset.
  TO=(); [[ "$CAP_SECS" -gt 0 ]] && TO=(timeout -k 15 "$CAP_SECS")
  if ${TO[@]+"${TO[@]}"} python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
        -st "$ST" -et "$ET" -d True -m TRADES -type DDPM -nsteps 100 -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" $BASE > "$LOGF" 2>&1; then
    SECS=$(( $(date +%s) - T0 )); CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$LOGF" | tail -1)
    R="  $D  OK  (${SECS}s)"; [[ -n "$CSV" && -f "$CSV" ]] && R="$R   $(movemetric "$CSV")"
    echo "$R" | tee -a "$PROG"; echo "$CSV" >> "$OUT_DIR/csv_list.txt"; DONE_N=$((DONE_N+1))
    [[ "$SECS" -gt "$MAXSECS" ]] && MAXSECS=$SECS   # feeds the adaptive deadline headroom
  else
    RC=$?; SECS=$(( $(date +%s) - T0 ))
    [[ $RC -eq 124 || $RC -eq 137 ]] && M="TIMEOUT" || M="ERROR rc=$RC"
    echo "  $D  $M after ${SECS}s --- continuing" | tee -a "$PROG"
  fi
done

write_status "COMPLETE --- $DONE_N days"
echo ""; echo "=== $DONE_N DDPM-100 days complete ==="; cat "$PROG"
echo ""; echo "CSVs listed in $OUT_DIR/csv_list.txt --- LOB-Bench these locally against the"
echo "matching DDIM-10 days to turn the one-day 0.447-vs-0.575 result into a distribution."
