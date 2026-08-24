#!/bin/bash
# sim_adopted_tsla.sh — simulate the ADOPTED TSLA SS checkpoint over the two test days at the aligned
# 2h window, and print a one-line stability/quality row per day so you can confirm it held.
#
# Default checkpoint is the 16th TSLA checkpoint (val_ema=0.821) — the one ss_epoch_stability.sh
# picked (cond_z[depth] single-digit, 1s-vol 1.35bp ~ real, exec 2.0% ~ real). Override with --ckpt
# or --nth. Resumable (.done sentinels); refuses to run while training is up.
#
#   bash scripts/sim_adopted_tsla.sh
#   bash scripts/sim_adopted_tsla.sh --ckpt data/checkpoints/TRADES/val_ema=0.821_..._TSLA_....ckpt
set -uo pipefail

TICKER="TSLA"; DAYS="20150129 20150130"; ST="10:00:00"; ET="12:00:00"
SIGMA="0.3"; TYPE_PRIOR="0.4912,0.4673,0.0416"
CKDIR="data/checkpoints/TRADES"; CKPT=""; NTH=16
CAP_SECS=2400; PY="${PY:-python}"
OUT_DIR="paper_runs/tsla_adopted_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --ckpt) CKPT="$2"; shift 2;; --nth) NTH="$2"; shift 2;; --days) DAYS="$2"; shift 2;;
  --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;; --sigma) SIGMA="$2"; shift 2;;
  --type-prior) TYPE_PRIOR="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"; say(){ echo "[$(date +%T)] $*"; }
pgrep -f "main.py" >/dev/null && { echo "!! main.py (training) running — kill it first (GPU contention)."; exit 1; }
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT

[[ -n "$CKPT" ]] || CKPT=$(ls -tr "$CKDIR"/*"$TICKER"*.ckpt 2>/dev/null | sed -n "${NTH}p")
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "!! no checkpoint (nth=$NTH) in $CKDIR — pass --ckpt"; exit 1; }
say "adopted checkpoint: $(basename "$CKPT")"
echo "# Adopted TSLA sim — $(date '+%F %T')  $(basename "$CKPT")  $ST-$ET  sigma=$SIGMA" > "$SUM"

run(){
  local D="$1"; local TAG="d${D}"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local ROW="$OUT_DIR/logs/row_${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" && -f "$ROW" ]] && { say "SKIP $D"; return; }
  say "-- $D  [$ST-$ET]"
  local T; T=$(mktemp); touch "$T"; local T0; T0=$(date +%s)
  local A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CKPT" -seed 30
           --depth-noise "$SIGMA" --size-reshape --type-decode prior --type-prior "$TYPE_PRIOR")
  if ! timeout -k 30 "$CAP_SECS" "${A[@]}" > "$LOG" 2>&1; then
    local rc=$?; say "   $([[ $rc -eq 124 || $rc -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$rc")"
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$T" ! -path "*market_replay*" ! -path "*/paper/*" 2>/dev/null | sort | tail -1); rm -f "$T"
  "$PY" - "$LOG" "${CSV:-none}" "$D" "$SECS" > "$ROW" <<'PY'
import sys, re, numpy as np, pandas as pd
log, csv, day, secs = sys.argv[1:5]
txt = open(log, errors="ignore").read()
m = re.search(r'cond_z\[depth\]: min=[-\d.eE+]+ mean=[-\d.eE+]+ max=([-\d.eE+]+)', txt)
dmax = float(m.group(1)) if m else float('nan')
mids = rng = vol = exe = '-'
if csv != "none":
    try:
        d = pd.read_csv(csv); d["dt"] = pd.to_datetime(d.iloc[:, 0], errors="coerce")
        for c in ("ask_price_1", "bid_price_1"): d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[(d.ask_price_1 > 0) & (d.bid_price_1 > 0) & (d.ask_price_1.abs() < 9e9) & (d.bid_price_1.abs() < 9e9)].dropna(subset=["dt"])
        d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 1e4
        d = d[d.dt >= d.dt.min() + pd.Timedelta(minutes=15)]
        s = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna(); r = np.log(s).diff().dropna()
        mids = int(s.round(3).nunique()); rng = f"{(s.max()-s.min())*100:.0f}"; vol = f"{r.std()*1e4:.2f}"
        exe = d.TYPE.value_counts(normalize=True).mul(100).round(1).get("ORDER_EXECUTED", "-")
    except Exception as e: mids = f"(err {type(e).__name__})"
dmax_s = f"{dmax:.1f}" if np.isfinite(dmax) else "nan"
try: bad = (not np.isfinite(dmax)) or dmax > 20 or (vol != '-' and float(vol) > 4)
except Exception: bad = True
print(f"| {day} | {dmax_s} | {mids} | {rng} | {vol} | {exe} | {csv} | {secs}s | {'**DIVERGED**' if bad else 'ok'} |")
PY
  touch "$DONE"; say "   $(cat "$ROW")"
}

for D in $DAYS; do run "$D"; done

{ echo ""; echo "| day | condZ_depth_max | mids | range(tk) | 1s_vol(bp) | exec% | csv | secs | verdict |"
  echo "|---|---|---|---|---|---|---|---|---|"
  for D in $DAYS; do cat "$OUT_DIR/logs/row_d${D}.txt" 2>/dev/null; done; } >> "$SUM"

echo ""; say "DONE — $SUM"; echo "(real TSLA ~1.3-1.5bp, exec ~2%; condZ_depth_max should stay single-digit)"; echo ""; cat "$SUM"
