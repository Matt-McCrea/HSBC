#!/bin/bash
# tsla_overnight_finish.sh — hand the GPU from the SS retrain to the deliverable runs, unattended.
#
# Launch this NOW, alongside the running retrain. It sleeps until the handover time (or until enough
# checkpoints exist), stops training, then runs the two-hour cells at the CORRECTED config and stops
# cleanly before the GPU window closes.
#
# WHY A FIXED HANDOVER TIME. Training has no natural end a script can detect — EarlyStopping may
# never fire inside the window, and "enough epochs" is a judgement. So the retrain gets a fixed
# share of the window and the deliverables get the rest, which is the allocation that matters when
# the window is hard-capped.
#
# CONFIG, all of it corrected since the earlier sweeps:
#   sigma        1.0        chosen on activity (mids 110->185 over sigma 0.15->1.0)
#   type prior   window-matched AND training-derived. Market share is 3.4x higher in the opening
#                half hour (0.128) than at 10:00-12:00 (0.038), so the 2h cells use the 2h prior.
#                Deriving from a test day would leak; deriving from the wrong window is worse.
#
#   nohup bash scripts/tsla_overnight_finish.sh > tsla_finish.log 2>&1 & disown
set -uo pipefail

TICKER="TSLA"; START="2015-01-02"; END="2015-01-30"
HANDOVER="05:30"              # stop training at this time (or earlier if MIN_CKPTS reached)
DEADLINE="11:30"              # never start a cell that cannot finish by here
MIN_CKPTS=8                   # hand over early once the SS lineage has this many checkpoints
ST="10:00:00"; ET="12:00:00"  # the two-hour window
SIGMA="1.0"
PRIOR="0.4918,0.4703,0.0380"  # 10:00-12:00, training days, pooled
BASE_CKPT=""                  # the pre-SS baseline; default = epoch 11
CAP_SECS=7200                 # 2h/cell. TSLA 30-min cells ran 89-587s, so 2h should be well inside.
SEED="30"; DRY=0
PY="${PY:-python}"
OUT_DIR="paper_runs/TSLA_finish_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --handover) HANDOVER="$2"; shift 2;; --deadline) DEADLINE="$2"; shift 2;;
  --min-ckpts) MIN_CKPTS="$2"; shift 2;; --sigma) SIGMA="$2"; shift 2;;
  --type-prior) PRIOR="$2"; shift 2;; --base-ckpt) BASE_CKPT="$2"; shift 2;;
  --cap-secs) CAP_SECS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

export TICKER TRADING_START="$START" TRADING_END="$END"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/real"
say () { echo "[$(date +%T)] $*"; }
epoch_of () { date -d "today $1" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $1" +%s; }

HANDOVER_E=$(epoch_of "$HANDOVER"); DEADLINE_E=$(epoch_of "$DEADLINE")
# both times are "tomorrow" when launched near midnight
NOW=$(date +%s)
[[ "$HANDOVER_E" -lt "$NOW" ]] && HANDOVER_E=$((HANDOVER_E + 86400))
[[ "$DEADLINE_E" -lt "$NOW" ]] && DEADLINE_E=$((DEADLINE_E + 86400))

DATA_DIR="data/${TICKER}/${TICKER}_${START}_${END}"
[[ -d "$DATA_DIR" ]] || DATA_DIR=$(ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1)
DAYS=$(ls "$DATA_DIR" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u | tail -2 | tr -d '-' | tr '\n' ' ')
[[ -n "$BASE_CKPT" ]] || BASE_CKPT=$(ls data/checkpoints/TRADES/*epoch=11*.ckpt 2>/dev/null | head -1)

if [[ "$DRY" == "1" ]]; then
  cat <<EOF
=== TSLA overnight finish ===
now       : $(date +%H:%M)
handover  : $HANDOVER  (or once $MIN_CKPTS checkpoints exist)
deadline  : $DEADLINE
test days : ${DAYS:-<none found>}
window    : $ST-$ET
sigma     : $SIGMA
prior     : $PRIOR
baseline  : ${BASE_CKPT:-<epoch 11 not found>}
out       : $OUT_DIR

cells, in priority order:
  1  SS-best   DDIM-10  $(awk '{print $1}' <<< "$DAYS")   2h
  2  SS-best   DDIM-10  $(awk '{print $2}' <<< "$DAYS")   2h
  3  baseline  DDIM-10  $(awk '{print $1}' <<< "$DAYS")   2h
  4  baseline  DDIM-10  $(awk '{print $2}' <<< "$DAYS")   2h
  5  baseline  DDPM-100 $(awk '{print $1}' <<< "$DAYS")   2h  (no decode flags)

SS cells run first: without them there is no scheduled-sampling result at all, whereas the baseline
already has 6 days of triage evidence.
EOF
  exit 0
fi

# ---- phase 1: wait, then stop training ------------------------------------
say "waiting for handover at $HANDOVER (or $MIN_CKPTS checkpoints)"
while true; do
  N=$(ls data/checkpoints/TRADES/*.ckpt 2>/dev/null | wc -l | tr -d ' ')
  NOW=$(date +%s)
  [[ "$NOW" -ge "$HANDOVER_E" ]] && { say "handover time reached ($N checkpoints)"; break; }
  [[ "$N" -ge "$MIN_CKPTS" ]] && { say "$N checkpoints reached — handing over early"; break; }
  pgrep -f "main.py" >/dev/null || { say "training stopped on its own ($N checkpoints)"; break; }
  say "  training: $N checkpoints, $(( (HANDOVER_E - NOW) / 60 ))min to handover"
  sleep 600
done
pkill -f main.py 2>/dev/null; sleep 10
rm -f SCHEDULED_SAMPLING_FLAG RESUME_TRAINING_FLAG    # decode-time runs must not resume training
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
say "training stopped; $(ls data/checkpoints/TRADES/*.ckpt 2>/dev/null | wc -l | tr -d ' ') checkpoints present"

# Newest checkpoint = the furthest-trained SS epoch.
SS_CKPT=$(ls -t data/checkpoints/TRADES/*.ckpt 2>/dev/null | head -1)
say "SS checkpoint : $(basename "${SS_CKPT:-none}")"
say "baseline ckpt : $(basename "${BASE_CKPT:-none}")"

# ---- phase 2: the deliverable cells ---------------------------------------
FIX="--depth-noise $SIGMA --size-reshape --type-decode prior --type-prior $PRIOR"
D1=$(awk '{print $1}' <<< "$DAYS"); D2=$(awk '{print $2}' <<< "$DAYS")

run () { # run <tag> <ckpt> <date> <sampler> <nsteps> <extra>
  local TAG="$1" CK="$2" D="$3" SAMP="$4" NST="$5" EXTRA="$6"
  [[ -n "$CK" && -f "$CK" ]] || { say "SKIP $TAG (no checkpoint)"; return; }
  [[ -f "$OUT_DIR/logs/.done_${TAG}" ]] && { say "SKIP $TAG (done)"; return; }
  local REMAIN=$(( DEADLINE_E - $(date +%s) ))
  [[ "$REMAIN" -lt "$CAP_SECS" ]] && { say "deadline: ${REMAIN}s left, stopping before $TAG"; return 1; }
  say "-- $TAG  [$SAMP-$NST $D $ST-$ET]"
  local T0; T0=$(date +%s)
  local A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type "$SAMP" -nsteps "$NST" -eta 0.0 --ckpt-path "$CK" -seed "$SEED")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA" ]] && A+=($EXTRA)
  if timeout -k 30 "$CAP_SECS" "${A[@]}" > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    local S=$(( $(date +%s) - T0 )); touch "$OUT_DIR/logs/.done_${TAG}"
    grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1 > "$OUT_DIR/logs/.csv_${TAG}"
    say "   done $((S/60))m $((S%60))s"
  else
    local RC=$?; local S=$(( $(date +%s) - T0 ))
    say "   $([[ $RC -eq 124 || $RC -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$RC") after $((S/60))m"
  fi
}

# SS first: without these there is no scheduled-sampling result at all.
run "ss_${D1}"       "$SS_CKPT"   "$D1" DDIM 10  "$FIX" || true
run "ss_${D2}"       "$SS_CKPT"   "$D2" DDIM 10  "$FIX" || true
run "base_${D1}"     "$BASE_CKPT" "$D1" DDIM 10  "$FIX" || true
run "base_${D2}"     "$BASE_CKPT" "$D2" DDIM 10  "$FIX" || true
run "ddpm_van_${D1}" "$BASE_CKPT" "$D1" DDPM 100 ""     || true

# ---- real references + summary (CPU) --------------------------------------
for D in $DAYS; do
  R="$OUT_DIR/real/real_${TICKER}_${D}.csv"
  [[ -f "$R" ]] || "$PY" -m evaluation.stylized_custom.lobster_real_reference \
      --ticker "$TICKER" --date "$D" --st "$ST" --et "$ET" --out "$R" >/dev/null 2>&1
done

"$PY" - "$OUT_DIR" <<'PY'
import glob, os, sys, numpy as np, pandas as pd
out = sys.argv[1]
def stat(p, warm=15):
    df = pd.read_csv(p); df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df.ask_price_1 > 0) & (df.bid_price_1 > 0) & (df.ask_price_1.abs() < 9e9)
           & (df.bid_price_1.abs() < 9e9)].copy()
    d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 1e4
    d = d.dropna(subset=["dt", "mid"]); d = d[d.dt >= d.dt.min() + pd.Timedelta(minutes=warm)]
    s = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
    r = np.log(s).diff().dropna(); m = d.TYPE.value_counts(normalize=True).mul(100).round(1)
    return (int(s.round(3).nunique()), float((s.max()-s.min())*100), float(r.std()*1e4),
            m.get("LIMIT_ORDER", float('nan')), m.get("ORDER_CANCELLED", float('nan')),
            m.get("ORDER_EXECUTED", float('nan')))
rows = []
for f in sorted(glob.glob(os.path.join(out, "logs", ".csv_*"))):
    tag = os.path.basename(f).replace(".csv_", ""); c = open(f).read().strip()
    if c and os.path.exists(c):
        try: rows.append((tag,) + stat(c))
        except Exception as e: print(f"  {tag}: {e}")
for f in sorted(glob.glob(os.path.join(out, "real", "real_*.csv"))):
    try: rows.append(("REAL " + os.path.basename(f).split("_")[-1][:8],) + stat(f))
    except Exception: pass
lines = ["", "| cell | mids | range(tk) | 1s vol(bp) | limit% | cancel% | exec% |",
         "|---|---|---|---|---|---|---|"]
for t, m, rg, v, l, c, e in rows:
    lines.append(f"| {t} | {m} | {rg:.0f} | {v:.2f} | {l} | {c} | {e} |")
txt = "\n".join(lines)
print(txt); open(os.path.join(out, "summary.md"), "w").write(txt + "\n")
PY

say "COMPLETE — $OUT_DIR/summary.md"
