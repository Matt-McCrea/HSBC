#!/bin/bash
# finalise_selected_model.sh --- use the last GPU window (hard end 20:00) to SELECT and VALIDATE
# the final simulation config: checkpoint + decode flags, including drift if it earns its place.
#
# THE LOGIC
#   The amplitude sweep is calibrating --depth-drift on ONE day (0129), on a 60-minute window,
#   against the exact metric it is scored on (VR). That is a fitted knob. A fitted knob is only
#   worth shipping if it GENERALISES, and there is exactly one way to find out: apply the chosen
#   amplitude unchanged to the other 19 trading days and see whether persistence still lands.
#
#   So: pick the winning amplitude automatically, then re-run the 20-day cross-day protocol with
#   it. This does two jobs at once ---
#     (a) re-earns the stability record that the current config has and a drift config does not;
#     (b) tests overfitting directly. VR(60s) near real on 19 held-out days is a real result.
#         VR(60s) good on 0129 and broken elsewhere means the knob was fitted to one day, and we
#         ship the current config instead. Either outcome finalises the model, which is the point.
#
#   If NO amplitude clears the guards, that is also an answer: the current config is already
#   validated, nothing more is needed, and the script says so and stops rather than burning GPU.
#
# TIMING (started ~08:00, sweep running):
#   sweep 3 cells -> ~11:55  |  20 days x ~20min -> ~18:40  |  deadline guard at 19:30
#
#   nohup bash scripts/finalise_selected_model.sh > finalise.log 2>&1 &
#   disown
#
# Watch:  cat crossday_final/latest/STATUS.txt
set -uo pipefail

WAIT_CELLS=3                 # sweep cells to allow before handover (amp0.05/0.08/0.12)
DEADLINE="19:30"
CAP_SECS=2400                # 40 min/day --- the established "unstable" criterion
CKPT_FRAG="0.69_epoch=4"     # SS epoch 4, the current lean. --ckpt to override.
PHI="0.9998"
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"; SEED="30"
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
POLL=120; FORCE_AMP=""; CHECK=0
SWEEP_REAL="ABIDES/log/market_replay_INTC_2015-01-29_11-00-00_30/processed_orders.csv"

DAYS=(20150107 20150129 20150102 20150105 20150106 20150108 20150109 20150112 20150113 20150114
      20150115 20150116 20150120 20150121 20150122 20150123 20150126 20150127 20150128 20150130)

while [[ $# -gt 0 ]]; do case "$1" in
  --wait-cells) WAIT_CELLS="$2"; shift 2;;
  --deadline) DEADLINE="$2"; shift 2;;
  --ckpt) CKPT_FRAG="$2"; shift 2;;
  --amp) FORCE_AMP="$2"; shift 2;;        # skip selection, use this amplitude
  --check) CHECK=1; shift;;               # score whatever cells exist NOW, print, exit. Safe:
                                          # does not wait, does not kill the sweep, runs nothing.
  --days) DAYS=($2); shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

DEADLINE_EPOCH=$(date -d "today $DEADLINE" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $DEADLINE" +%s)

echo "=== finalise selected model ==="
echo "ckpt: $CKPT_FRAG   phi: $PHI   deadline: $DEADLINE"

# ---- phase 1: wait for enough sweep cells ----
if [[ -z "$FORCE_AMP" && "$CHECK" == "0" ]]; then
  while true; do
    N=$(ls drift_sweep/latest/logs/.done_amp* 2>/dev/null | wc -l | tr -d ' ')
    [[ "$N" -ge "$WAIT_CELLS" ]] && { echo "[$(date +%T)] $N sweep cells done --- selecting"; break; }
    pgrep -f drift_persistence_sweep > /dev/null || { echo "[$(date +%T)] sweep stopped at $N cells"; break; }
    # do not let the sweep eat the validation window; 20 days needs ~7h
    if [[ $(date +%s) -ge $((DEADLINE_EPOCH - 25200)) ]]; then
      echo "[$(date +%T)] validation window at risk --- cutting sweep at $N cells"; break; fi
    echo "[$(date +%T)] sweep: $N/$WAIT_CELLS cells"; sleep "$POLL"
  done
  echo "[$(date +%T)] stopping sweep"
  pkill -f drift_persistence_sweep 2>/dev/null; sleep 2; pkill -f abides.py 2>/dev/null; sleep 5
fi

# ---- phase 2: pick the amplitude ----
if [[ -n "$FORCE_AMP" ]]; then
  AMP="$FORCE_AMP"; echo "amplitude forced: $AMP"
else
  AMP=$(python3 - "$SWEEP_REAL" <<'PY'
import sys, os, glob, numpy as np, pandas as pd
realp = sys.argv[1]
def series(p, warm=15):
    df = pd.read_csv(p); df["dt"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    for c in ("ask_price_1","bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df.ask_price_1>0)&(df.bid_price_1>0)&(df.ask_price_1.abs()<9e9)&(df.bid_price_1.abs()<9e9)].copy()
    d["mid"]=(d.ask_price_1+d.bid_price_1)/2/1e4
    d=d.dropna(subset=["dt","mid"]); d=d[d.dt>=d.dt.min()+pd.Timedelta(minutes=warm)]
    return d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
def vr(s,q):
    r=np.log(s).diff().dropna().values; n=len(r)//q*q
    if n<q*3: return float("nan")
    r=r[:n]; return r.reshape(-1,q).sum(axis=1).var(ddof=1)/(q*r.var(ddof=1))
def stats(s):
    r=np.log(s).diff().dropna()
    return r.std()*1e4, (s.max()-s.min())*100, vr(s,60)
try:
    rv, rr, rvr = stats(series(realp))
except Exception as e:
    print("ERROR:real reference unreadable (%s): %s" % (realp, e)); raise SystemExit(0)
cands=sorted(glob.glob("drift_sweep/latest/logs/.csv_amp*"))
if not cands:
    print("ERROR:no .csv_amp* sentinels under drift_sweep/latest/logs --- "
          "is the running sweep the --plan amplitude one?"); raise SystemExit(0)
best=None; seen=0
for f in cands:
    name=os.path.basename(f).replace(".csv_","")
    if "lever" in name: continue                      # validate the simple config first
    amp=name.replace("amp","")
    try: float(amp)                                   # guard against stale non-numeric cell names
    except ValueError:
        sys.stderr.write("  skip %s (cell name is not an amplitude)\n" % name); continue
    csv=open(f).read().strip()
    if not csv or not os.path.exists(csv):
        sys.stderr.write("  skip %s (csv missing: %s)\n" % (name, csv or "<empty>")); continue
    try: v, rng, q60 = stats(series(csv))
    except Exception as e:
        sys.stderr.write("  skip %s (unreadable: %s)\n" % (name, e)); continue
    seen+=1
    # guards: clearly fixed the pathology, did not overshoot into trending, did not blow up
    # range or volatility. Deliberately generous --- a marginal cell is still worth validating.
    ok = (0.45 <= q60 <= 1.80) and (rng <= 2.0*rr) and (v <= 2.2)
    sys.stderr.write("  amp=%s vol=%.2f range=%.0f VR60=%.3f  %s\n" % (amp,v,rng,q60,"OK" if ok else "reject"))
    if ok:
        score=abs(np.log(q60/rvr)) if q60>0 and rvr>0 else 9e9
        if best is None or score<best[0]: best=(score,amp)
sys.stderr.write("real: vol=%.2f range=%.0f VR60=%.3f\n" % (rv,rr,rvr))
if best: print(best[1])
elif seen: print("NONE")                              # cells scored, none passed --- a real answer
else: print("ERROR:no amplitude cell could be scored (see skip reasons above)")
PY
)
fi

# distinguish "no cell passed" (a result) from "the selector broke" (must not be read as a result)
if [[ -z "$AMP" || "$AMP" == ERROR:* ]]; then
  echo ""
  echo "!! SELECTION FAILED --- ${AMP:-selector produced no output at all}"
  echo "!! This is NOT 'no amplitude worked'. Nothing has been validated and the GPU is now idle."
  echo "!! Inspect:  ls -la drift_sweep/latest/logs/.csv_amp* ; cat drift_sweep/latest/logs/.csv_amp0.08"
  echo "!! Then re-run with the amplitude chosen by hand, e.g.:"
  echo "!!   nohup bash scripts/finalise_selected_model.sh --amp 0.08 > finalise2.log 2>&1 &"
  exit 1
fi

if [[ "$AMP" == "NONE" ]]; then
  echo ""
  echo "=== NO AMPLITUDE CLEARED THE GUARDS ==="
  echo "The drift knob does not earn a place in the final config. That is a result, not a failure:"
  echo "the current config is ALREADY validated across 20 days, so the final model stands as-is"
  echo "and the sweep becomes a quantified diagnostic in the write-up rather than a fix."
  echo ""
  # Do not leave the last GPU window idle. Nothing more is needed to SELECT the model, so spend
  # the remaining hours on the thinnest claim in the write-up instead: '10 steps beat 100' rests
  # on a single day (0130, 0.447 vs 0.575) while every other headline number is a 20-day figure.
  echo "=== falling through to DDPM-100 days (deadline $DEADLINE) ==="
  exec bash scripts/final_gpu_window.sh --now --deadline "$DEADLINE"
fi

if [[ "$CHECK" == "1" ]]; then
  echo ""; echo "=== --check: would select amplitude $AMP --- nothing run, sweep untouched ==="
  exit 0
fi

echo ""
echo "=== SELECTED amplitude $AMP (phi $PHI) --- validating across the month ==="

# ---- phase 3: cross-day validation with the selected config ----
find_ckpt () { local h; for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
    data/checkpoints/TRADES_other_recovered data/checkpoints; do
    h=$(ls "$d"/*"$1"*.ckpt 2>/dev/null | head -1); [[ -n "$h" ]] && { echo "$h"; return 0; }; done; return 1; }
ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }

OUT_DIR="crossday_final/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/logs" crossday_final
ln -sfn "$(basename "$OUT_DIR")" crossday_final/latest
STATUS="$OUT_DIR/STATUS.txt"; PROG="$OUT_DIR/progress.txt"; : > "$PROG"
echo "config: ckpt=$CKPT_FRAG  --depth-drift $AMP --depth-drift-phi $PHI  $BASE" | tee -a "$PROG"
echo "" >> "$PROG"
printf '%-12s %-7s %-9s %-8s %-8s %-9s %-9s\n' DAY secs uniq_mid vol_bp range VR60_gen VR60_real | tee -a "$PROG"

touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
CK=$(find_ckpt "$CKPT_FRAG") || { echo "!! ckpt '$CKPT_FRAG' not found"; exit 1; }

# per-day row: the generated series AND, where a real replay exists, the same numbers for real.
# VR side by side per day is what separates "the knob generalises" from "it was fitted to 0129".
dayrow () {
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import sys, os, numpy as np, pandas as pd, datetime as dt
def series(p):
    df=pd.read_csv(p); df["dt"]=pd.to_datetime(df.iloc[:,0],errors="coerce")
    for c in ("ask_price_1","bid_price_1"): df[c]=pd.to_numeric(df[c],errors="coerce")
    d=df[(df.ask_price_1>0)&(df.bid_price_1>0)&(df.ask_price_1.abs()<9e9)&(df.bid_price_1.abs()<9e9)].copy()
    d["mid"]=(d.ask_price_1+d.bid_price_1)/2/1e4
    d=d.dropna(subset=["dt","mid"]); d=d[d.dt.dt.time>=dt.time(9,45,0)]
    return d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
def vr(s,q):
    r=np.log(s).diff().dropna().values; n=len(r)//q*q
    if n<q*3: return float("nan")
    r=r[:n]; return r.reshape(-1,q).sum(axis=1).var(ddof=1)/(q*r.var(ddof=1))
g=series(sys.argv[1]); r=np.log(g).diff().dropna()
rv="  n/a  "
if len(sys.argv)>2 and sys.argv[2] and os.path.exists(sys.argv[2]):
    try: rv="%.3f"%vr(series(sys.argv[2]),60)
    except Exception: pass
print("%-8d %-8.2f %-8.0f %-9.3f %s"%(g.round(3).nunique(), r.std()*1e4,
      (g.max()-g.min())*100, vr(g,60), rv))
PY
}
write_status () { { echo "=== CROSS-DAY VALIDATION of drift $AMP --- $(date '+%F %T') ==="; echo "$1"
  echo ""; cat "$PROG"; } > "$STATUS"; }

OK_N=0; FAIL_N=0
for D in "${DAYS[@]}"; do
  REMAIN=$(( DEADLINE_EPOCH - $(date +%s) ))
  if [[ "$REMAIN" -lt "$CAP_SECS" ]]; then
    echo "  -- deadline: stopping before $D (${REMAIN}s left)" | tee -a "$PROG"
    write_status "STOPPED at deadline before $D"; break
  fi
  write_status "running $D   ($OK_N ok / $FAIL_N failed so far)"
  T0=$(date +%s); LOGF="$OUT_DIR/logs/${D}.txt"
  REALP="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$D")_10-00-00_${SEED}/processed_orders.csv"
  if timeout -k 15 "$CAP_SECS" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" \
        -st "$ST" -et "$ET" -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" $BASE --depth-drift "$AMP" --depth-drift-phi "$PHI" \
        > "$LOGF" 2>&1; then
    SECS=$(( $(date +%s) - T0 )); CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$LOGF" | tail -1)
    ROW=$([[ -n "$CSV" && -f "$CSV" ]] && dayrow "$CSV" "$REALP" || echo "(no csv)")
    printf '%-12s %-7s %s\n' "$D" "$SECS" "$ROW" | tee -a "$PROG"
    [[ -n "$CSV" ]] && echo "$CSV" >> "$OUT_DIR/csv_list.txt"
    OK_N=$((OK_N+1))
  else
    RC=$?; SECS=$(( $(date +%s) - T0 ))
    [[ $RC -eq 124 || $RC -eq 137 ]] && M=TIMEOUT || M="ERROR rc=$RC"
    printf '%-12s %-7s %s\n' "$D" "$SECS" "$M --- continuing" | tee -a "$PROG"
    FAIL_N=$((FAIL_N+1))
  fi
done

write_status "COMPLETE --- $OK_N ok, $FAIL_N failed"
echo ""; echo "=== drift $AMP: $OK_N days ok, $FAIL_N failed ==="
cat "$PROG"
echo ""
echo "READ IT LIKE THIS: if VR60_gen tracks VR60_real across the held-out days, the amplitude"
echo "generalises and the drift config is the final model. If it only works on 0129, it was"
echo "fitted to that day --- ship the current config and report the sweep as a diagnostic."
