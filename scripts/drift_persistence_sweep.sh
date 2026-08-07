#!/bin/bash
# drift_persistence_sweep.sh --- can decode-time persistence fix the mean-reversion pathology?
#
# THE PROBLEM (measured on the 2h runs, INTC 2015-01-29):
#   variance ratio VR(q) = Var(q-period return) / (q * Var(1-period return));  1 = random walk
#
#                     1s vol   range    VR(10s)  VR(60s)  VR(300s)
#     real             1.18bp   56tk      0.897    0.931     1.019   <- random walk
#     SS epoch 4       1.35bp   28tk      0.205    0.047     0.016   <- violently mean-reverting
#     0.724            1.77bp   14tk      0.300    0.098     0.027
#
#   Our models already have MORE 1-second volatility than real. The deficiency is that the movement
#   cancels instead of accumulating, which is why the range is half real's despite more jitter.
#   So the fix is persistence, NOT more variance --- raising --depth-noise would make it worse.
#
# THE LEVER: --depth-drift is an AR(1) directional bias on the depth channel that ticks PER ORDER
#   (event time). At the observed ~45 events/s, the default phi=0.995 gives only ~4.4s of
#   persistence, which is why it was previously judged "modest" --- it was an order of magnitude
#   too short to touch the 60-300s timescale where the mean reversion actually bites.
#
#     phi=0.9995 -> ~44 s      phi=0.9998 -> ~111 s      phi=0.9999 -> ~221 s
#
# THE DESIGN: sweep persistence FIRST (it sets which timescale is fixed), amplitude second. Also
#   trade iid noise for persistent drift rather than stacking them, since 1s volatility is already
#   above real --- hence the reduced --depth-noise in the trade-off arm.
#
# SCORED ON: VR(60s) and VR(300s) -> 1.0, while ret1s_std stays ~1.2-1.5bp and range grows toward
#   56tk. NOT uniq_mid alone --- that is the proxy that misled us into reading this as an activity
#   deficit in the first place.
#
# Usage:
#   bash scripts/drift_persistence_sweep.sh                    # full sweep, ~3.5h
#   bash scripts/drift_persistence_sweep.sh --quick            # persistence only, ~1.5h
#   bash scripts/drift_persistence_sweep.sh --window 60        # 60-min cells instead of 120
#   bash scripts/drift_persistence_sweep.sh --dry-run
set -uo pipefail

TICKER="INTC"; DAY="20150129"; SEED="30"
CKPT_FRAG="0.69_epoch=4"          # the long-horizon winner
ST="10:00:00"; ET="12:00:00"      # 2h: the horizon where the pathology is measurable
BASE="--size-reshape --type-decode prior"
CAP=9000
QUICK=0; DRY=0
OUT_DIR="drift_sweep/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --quick) QUICK=1; shift;;
  --dry-run) DRY=1; shift;;
  --ckpt) CKPT_FRAG="$2"; shift 2;;
  --day) DAY="$2"; shift 2;;
  --window) [[ "$2" == "60" ]] && ET="11:00:00"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# cell name | depth-noise | depth-drift | phi        (empty drift = control)
CELLS=(
  "control_current|0.3|0.0|0.995"          # what we ship today --- the reference row
  "phi44s|0.3|0.25|0.9995"                 # persistence sweep, amplitude held
  "phi111s|0.3|0.25|0.9998"
  "phi221s|0.3|0.25|0.9999"
  "tradeoff_phi111s|0.2|0.25|0.9998"       # trade iid noise for drift (1s vol already too high)
)
EXTRA=(
  "amp_lo_phi111s|0.3|0.15|0.9998"         # amplitude sweep at the best-guess persistence
  "amp_hi_phi111s|0.3|0.40|0.9998"
  "tradeoff_hard_phi221s|0.15|0.35|0.9999" # strongest trade-off arm
)
[[ "$QUICK" == "0" ]] && CELLS+=("${EXTRA[@]}")

if [[ "$DRY" == "1" ]]; then
  echo "ckpt=$CKPT_FRAG  day=$DAY  window=$ST-$ET"
  printf '%-24s %-12s %-12s %-10s\n' CELL DEPTH-NOISE DEPTH-DRIFT PHI
  for c in "${CELLS[@]}"; do IFS='|' read -r n dn dd phi <<< "$c"
    printf '%-24s %-12s %-12s %-10s\n' "$n" "$dn" "$dd" "$phi"; done
  echo ""; echo "cells: ${#CELLS[@]}  est: ~$(( ${#CELLS[@]} * 26 ))min"
  exit 0
fi

mkdir -p "$OUT_DIR/logs"; mkdir -p drift_sweep
ln -sfn "$(basename "$OUT_DIR")" drift_sweep/latest
SUM="$OUT_DIR/summary.md"; : > "$SUM"

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
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
PRE=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py > "$OUT_DIR/logs/qt.txt" 2>&1
CK=$(find_ckpt "$CKPT_FRAG") || { echo "!! ckpt '$CKPT_FRAG' not found"; exit 1; }
echo "ckpt: $(basename "$CK")"

REALP="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$DAY")_${ET//:/-}_${SEED}/processed_orders.csv"
[[ -f "$REALP" ]] || { echo "-- generating real replay"; \
  python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DAY" -st "$ST" -et "$ET" \
    > "$OUT_DIR/logs/real.txt" 2>&1; }

for c in "${CELLS[@]}"; do
  IFS='|' read -r NAME DN DD PHI <<< "$c"
  DONE="$OUT_DIR/logs/.done_${NAME}"
  [[ -f "$DONE" ]] && { echo "   SKIP $NAME"; continue; }
  DRIFTARGS=""
  [[ "$DD" != "0.0" ]] && DRIFTARGS="--depth-drift $DD --depth-drift-phi $PHI"
  echo "-- $NAME   (dn=$DN drift=$DD phi=$PHI)"
  T0=$(date +%s)
  if ! timeout -k 30 "$CAP" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DAY" \
        -st "$ST" -et "$ET" -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" --depth-noise "$DN" $BASE $DRIFTARGS \
        > "$OUT_DIR/logs/${NAME}.txt" 2>&1; then
    echo "   FAILED/TIMEOUT"; echo "| $NAME | $DN | $DD | $PHI | FAILED | | | | |" >> "$SUM"; continue
  fi
  SECS=$(( $(date +%s) - T0 ))
  CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${NAME}.txt" | tail -1)
  echo "$CSV" > "$OUT_DIR/logs/.csv_${NAME}"
  touch "$DONE"; echo "   done $((SECS/60))m"
done

# ---- score every cell on the variance ratio ----
python3 - "$OUT_DIR" "$REALP" <<'PY'
import sys, os, glob, numpy as np, pandas as pd
out, realp = sys.argv[1], sys.argv[2]

def series(p, warm=15):
    df = pd.read_csv(p); df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df.ask_price_1 > 0) & (df.bid_price_1 > 0)
           & (df.ask_price_1.abs() < 9e9) & (df.bid_price_1.abs() < 9e9)].copy()
    d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 10000.0
    d = d.dropna(subset=["dt", "mid"]); t0 = d.dt.min()
    d = d[d.dt >= t0 + pd.Timedelta(minutes=warm)]
    return d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()

def vr(s, q):
    r = np.log(s).diff().dropna().values
    n = len(r)//q*q
    if n < q*3: return float("nan")
    r = r[:n]; return r.reshape(-1, q).sum(axis=1).var(ddof=1) / (q * r.var(ddof=1))

def row(lab, s):
    r = np.log(s).diff().dropna()
    return (lab, r.std()*1e4, (s.max()-s.min())*100, s.round(3).nunique(),
            vr(s,10), vr(s,60), vr(s,300))

rows = [row("REAL (target)", series(realp))]
for f in sorted(glob.glob(os.path.join(out, "logs", ".csv_*"))):
    name = os.path.basename(f)[5:]
    csv = open(f).read().strip()
    if csv and os.path.exists(csv):
        try: rows.append(row(name, series(csv)))
        except Exception as e: print("  !! %s: %s" % (name, e))

hdr = f"{'cell':<24}{'1s vol':>8}{'range':>7}{'uniq':>6}{'VR10s':>8}{'VR60s':>8}{'VR300s':>8}"
print("\n" + "="*len(hdr)); print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r[0]:<24}{r[1]:8.2f}{r[2]:7.0f}{r[3]:6d}{r[4]:8.3f}{r[5]:8.3f}{r[6]:8.3f}")
print("="*len(hdr))
print("GOAL: VR60s and VR300s -> 1.0 (real is ~0.93 / ~1.02) while 1s vol stays ~1.2-1.5bp")
print("      and range grows toward real's. A cell that raises 1s vol but leaves VR near 0 has")
print("      added jitter, not persistence, and is NOT an improvement.")

with open(os.path.join(out, "summary.md"), "w") as f:
    f.write("# Drift-persistence sweep\n\n")
    f.write("| cell | 1s vol (bp) | range (tk) | uniq mids | VR(10s) | VR(60s) | VR(300s) |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.0f} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} |\n")
    f.write("\nGoal: VR(60s)/VR(300s) approach 1.0 (real ~0.93/~1.02) while 1s volatility stays "
            "near 1.2-1.5bp and range grows toward real's. Rising 1s vol with VR still near zero "
            "means jitter was added, not persistence.\n")
PY
echo ""; echo "Summary: $OUT_DIR/summary.md"
