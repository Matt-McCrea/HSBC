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
#
#   THE SAME PATHOLOGY IS IN TRADES'S OWN RELEASED OUTPUT (VR60s 0.099 / 0.153 on 0130 / 0129), so
#   it is inherited from the architecture rather than introduced by our decode-time corrections.
#
#   ONE INTERVENTION IS ALREADY PROVEN: the book-balancing cancel (--book-target-thick 2.0
#   --book-cancel-rate 0.5) took VR60s from 0.047 to 0.112 and range from 28 to 32tk on the 2h run
#   (P4). That was NOT the predicted direction --- it trims whichever side has grown thick, so it
#   should act as a restoring force. It does not. Lever arms therefore run early, and the headline
#   question is whether it COMPOUNDS with drift.
#
#   TWO FURTHER HYPOTHESES, which this sweep is built to separate:
#     (a) persistence is the missing ingredient -> --depth-drift with long phi should fix it, and
#         iid noise should NOT, since independent draws cancel;
#     (b) volatility buys persistence -> across existing CSVs the high-volatility runs have far
#         better variance ratios (DDPM-100 on 0.681: 1svol 4.54, VR60s 0.835; DDIM-1 on 0.763:
#         4.93, 0.765) than the low-volatility post-fix ones (1.35-1.95, VR60s 0.05-0.10).
#   (b) is confounded by checkpoint, and contradicts the theory behind (a). Both get arms here.
#
# THE LEVER: --depth-drift is an AR(1) directional bias on the depth channel that ticks PER ORDER
#   (event time). At the observed ~45 events/s, the default phi=0.995 gives only ~4.4s of
#   persistence, which is why it was previously judged "modest" --- it was an order of magnitude
#   too short to touch the 60-300s timescale where the mean reversion actually bites.
#
#     phi=0.9995 -> ~44 s      phi=0.9998 -> ~111 s      phi=0.9999 -> ~221 s
#
# THE DESIGN: sweep persistence FIRST (it sets which timescale is fixed), amplitude second; add
#   trade-off arms that swap iid noise for drift, raw high-noise arms to test hypothesis (b), and a
#   no-type-prior arm testing whether pinning the type mix to a fixed prior restores direction.
#
# SCORED ON: VR(60s) and VR(300s) -> 1.0, while ret1s_std stays ~1.2-1.5bp and range grows toward
#   56tk. NOT uniq_mid alone --- that is the proxy that misled us into reading this as an activity
#   deficit in the first place.
#
# Usage:
# COST: per-cell wall-clock is driven by the SIMULATED window, not the cell count --- measured at
#   ~25min for a 30min window, ~78min for 60min, ~156min for 2h. A 14-cell sweep at the 2h window is
#   ~36h, NOT the ~6h originally estimated. Always run --dry-run first; it now prints the real total.
#
#   Recommended for a ~6h budget:
#     bash scripts/drift_persistence_sweep.sh --window 60 --cells 5     # ~6.5h, VR(60s) reliable
#
#   Other forms:
#     bash scripts/drift_persistence_sweep.sh --dry-run                 # print plan + honest cost
#     bash scripts/drift_persistence_sweep.sh --window 30 --cells 12    # ~5h, but VR(300s) unusable
#
# WINDOW vs MEASUREMENT QUALITY: VR(60s) needs ~25+ non-overlapping minute blocks to be stable, so a
#   60min window is adequate for it; VR(300s) needs a 2h window to be trustworthy (on a 30min window
#   real itself reads 1.62 against its true ~1.02). Prefer 60min and read VR(60s) as the headline.
#   bash scripts/drift_persistence_sweep.sh --window 60        # 60-min cells instead of 120
#   bash scripts/drift_persistence_sweep.sh --dry-run
set -uo pipefail

TICKER="INTC"; DAY="20150129"; SEED="30"
CKPT_FRAG="0.69_epoch=4"          # the long-horizon winner
ST="10:00:00"; ET="12:00:00"      # 2h: the horizon where the pathology is measurable
BASE="--size-reshape --type-decode prior"
# Per-cell wall-clock scales with the SIMULATED window, not the cell count: a 2h window costs
# ~2.6h, a 60min window ~1.3h, a 30min window ~25min. CAP is set from the window below with ~50%
# headroom so a cell is never killed just short of finishing (which is what happened on the first
# attempt: cells reached simulated 11:50 of a 12:00 window and were cut off at 98%).
CAP=""
QUICK=0; DRY=0; MAXCELLS=0
OUT_DIR="drift_sweep/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --quick) QUICK=1; shift;;
  --dry-run) DRY=1; shift;;
  --ckpt) CKPT_FRAG="$2"; shift 2;;
  --day) DAY="$2"; shift 2;;
  --window) case "$2" in 30) ET="10:30:00";; 60) ET="11:00:00";; 120) ET="12:00:00";; esac; shift 2;;
  --cells) MAXCELLS="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# cell name | depth-noise | depth-drift | phi | lever(0/1)
#
# Ordered by expected value: the lever arms run early because the book-balancing cancel is the ONLY
# intervention so far MEASURED to improve persistence (P4: VR60s 0.047 -> 0.112, range 28 -> 32tk),
# whereas every drift/noise arm is still speculative. If the sweep is cut short, the compounding
# question is the one worth having answered.
CELLS=(
  "control_current|0.3|0.0|0.995|0"        # what we ship today --- the reference row
  "lever_only|0.3|0.0|0.995|1"             # re-run of the P4 config, in-sweep so it is directly comparable
  "lever_drift_phi111s|0.3|0.25|0.9998|1"  # *** does the proven lever COMPOUND with drift? ***
  "phi111s|0.3|0.25|0.9998|0"              # drift alone, mid persistence
  "phi221s|0.3|0.25|0.9999|0"              # drift alone, long persistence
  "tradeoff_phi111s|0.2|0.25|0.9998|0"     # trade iid noise for drift (1s vol already too high)
)
EXTRA=(
  "lever_tradeoff|0.2|0.25|0.9999|1"       # lever + long drift + reduced noise: everything at once
  "amp_hi_phi111s|0.3|0.40|0.9998|0"       # amplitude sweep at the best-guess persistence
  "amp_lo_phi111s|0.3|0.15|0.9998|0"
  "phi44s|0.3|0.25|0.9995|0"               # short-persistence anchor: expected to do little
  "tradeoff_hard_phi221s|0.15|0.35|0.9999|0"
  # --- noise arms: test the OBSERVATIONAL hypothesis that volatility buys persistence ---
  # Across existing CSVs the high-volatility runs have far better variance ratios
  # (DDPM-100 on 0.681: 1svol 4.54, VR60s 0.835; DDIM-1 on 0.763: 4.93, 0.765) than the
  # low-volatility post-fix ones (1.35-1.95, VR60s 0.05-0.10). That is confounded by
  # checkpoint, so it needs a controlled test rather than an assumption. Theory says iid
  # noise should cancel and NOT buy persistence; the data hints otherwise. Settle it.
  "noise_hi|0.5|0.0|0.995|0"
  "noise_hi_drift|0.5|0.25|0.9998|0"       # noise + drift together
  "no_type_prior|0.3|0.0|0.995|0"          # drop --type-decode prior (see CELLBASE below)
)
[[ "$QUICK" == "0" ]] && CELLS+=("${EXTRA[@]}")
[[ "$MAXCELLS" -gt 0 ]] && CELLS=("${CELLS[@]:0:$MAXCELLS}")

# minutes of simulated time -> measured wall-clock per cell -> cap with headroom
case "$ET" in
  10:30:00) PERCELL=25;  CAP=${CAP:-3600};;
  11:00:00) PERCELL=78;  CAP=${CAP:-9000};;
  12:00:00) PERCELL=156; CAP=${CAP:-18000};;
  *)        PERCELL=156; CAP=${CAP:-18000};;
esac

if [[ "$DRY" == "1" ]]; then
  echo "ckpt=$CKPT_FRAG  day=$DAY  window=$ST-$ET"
  printf '%-24s %-12s %-12s %-10s %-7s\n' CELL DEPTH-NOISE DEPTH-DRIFT PHI LEVER
  for c in "${CELLS[@]}"; do IFS='|' read -r n dn dd phi lv <<< "$c"
    printf '%-24s %-12s %-12s %-10s %-7s\n' "$n" "$dn" "$dd" "$phi" \
      "$([[ "$lv" == "1" ]] && echo yes || echo -)"; done
  TOT=$(( ${#CELLS[@]} * PERCELL ))
  echo ""; echo "window $ST-$ET  ->  ~${PERCELL}min/cell  (measured, not guessed)"
  echo "cells: ${#CELLS[@]}   TOTAL: ~$(( TOT / 60 ))h $(( TOT % 60 ))m   per-cell cap: $(( CAP / 60 ))min"
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
  IFS='|' read -r NAME DN DD PHI LV <<< "$c"
  DONE="$OUT_DIR/logs/.done_${NAME}"
  [[ -f "$DONE" ]] && { echo "   SKIP $NAME"; continue; }
  DRIFTARGS=""
  [[ "$DD" != "0.0" ]] && DRIFTARGS="--depth-drift $DD --depth-drift-phi $PHI"
  [[ "$LV" == "1" ]] && DRIFTARGS="$DRIFTARGS --book-target-thick 2.0 --book-cancel-rate 0.5"
  # the no_type_prior arm drops --type-decode prior to test whether pinning the type mix to a
  # fixed prior (0.49/0.48/0.03) acts as a restoring force on direction
  CELLBASE="$BASE"
  [[ "$NAME" == no_type_prior* ]] && CELLBASE="--size-reshape"
  echo "-- $NAME   (dn=$DN drift=$DD phi=$PHI)"
  T0=$(date +%s)
  if ! timeout -k 30 "$CAP" python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DAY" \
        -st "$ST" -et "$ET" -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" --depth-noise "$DN" $CELLBASE $DRIFTARGS \
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
