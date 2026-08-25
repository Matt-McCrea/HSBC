#!/bin/bash
# tsla_depth_clip_test.sh — test the --depth-cond-clip guard on the adopted TSLA checkpoint over the
# two test days at the 2h horizon, where the base config diverges (cond_z[depth] -> ~-2000, mid runs
# away). The clamp bounds the standardised DEPTH conditioning before it enters the model, so a single
# order priced far through the book cannot cascade. This is otherwise the SAME base config, so it is a
# clean A/B against the earlier no-clamp 2h runs.
#
# READ: if the clamp arrests it, the RAW cond_z[depth] range stays bounded (the loop never diverges),
# depth_cond_clipped is small, and range/vol come back toward real. If it merely survives, cond_z is
# bounded but range/vol stay high -> the guard is necessary but not sufficient.
#
#   bash scripts/tsla_depth_clip_test.sh                 # 0.821 (16th ckpt), clip=8, both days
#   bash scripts/tsla_depth_clip_test.sh --clip 6 --nth 11
set -uo pipefail

TICKER="TSLA"; DAYS="20150129 20150130"; ST="10:00:00"; ET="12:00:00"
SIGMA="0.3"; TYPE_PRIOR="0.4912,0.4673,0.0416"; CLIP="8"
CKDIR="data/checkpoints/TRADES"; CKPT=""; NTH=16
CAP_SECS=2400; PY="${PY:-python}"
OUT_DIR="paper_runs/tsla_depthclip_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --ckpt) CKPT="$2"; shift 2;; --nth) NTH="$2"; shift 2;; --clip) CLIP="$2"; shift 2;;
  --days) DAYS="$2"; shift 2;; --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --sigma) SIGMA="$2"; shift 2;; --type-prior) TYPE_PRIOR="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"; say(){ echo "[$(date +%T)] $*"; }
pgrep -f "main.py" >/dev/null && { echo "!! main.py (training) running — kill it first (GPU contention)."; exit 1; }
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT

[[ -n "$CKPT" ]] || CKPT=$(ls -tr "$CKDIR"/*"$TICKER"*.ckpt 2>/dev/null | sed -n "${NTH}p")
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "!! no checkpoint (nth=$NTH) in $CKDIR — pass --ckpt"; exit 1; }
say "checkpoint: $(basename "$CKPT")   depth-cond-clip=$CLIP   window: $ST-$ET"
echo "# TSLA depth-cond-clip test — $(date '+%F %T')  $(basename "$CKPT")  clip=$CLIP" > "$SUM"

run(){
  local D="$1"; local TAG="d${D}"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local ROW="$OUT_DIR/logs/row_${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" && -f "$ROW" ]] && { say "SKIP $D"; return; }
  say "-- $D  [$ST-$ET  clip=$CLIP]"
  local T; T=$(mktemp); touch "$T"; local T0; T0=$(date +%s)
  local A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CKPT" -seed 30
           --depth-noise "$SIGMA" --size-reshape --type-decode prior --type-prior "$TYPE_PRIOR"
           --depth-cond-clip "$CLIP")
  if ! timeout -k 30 "$CAP_SECS" "${A[@]}" > "$LOG" 2>&1; then
    local rc=$?; say "   $([[ $rc -eq 124 || $rc -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$rc")"
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$T" ! -path "*market_replay*" ! -path "*/paper/*" 2>/dev/null | sort | tail -1); rm -f "$T"
  "$PY" - "$LOG" "${CSV:-none}" "$D" "$SECS" > "$ROW" <<'PY'
import sys, re, numpy as np, pandas as pd
log, csv, day, secs = sys.argv[1:5]
txt = open(log, errors="ignore").read()
m = re.search(r'cond_z\[depth\]: min=([-\d.eE+]+) mean=[-\d.eE+]+ max=([-\d.eE+]+)', txt)
dmin = float(m.group(1)) if m else float('nan'); dmax = float(m.group(2)) if m else float('nan')
clip = (re.search(r'depth_cond_clipped=(\d+)', txt) or [None, '-'])[1]
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
dr = f"{dmin:.0f}/{dmax:.0f}" if np.isfinite(dmin) else "nan"
try: bad = (not np.isfinite(dmin)) or abs(dmin) > 30 or abs(dmax) > 30 or (vol != '-' and float(vol) > 4)
except Exception: bad = True
print(f"| {day} | {dr} | {clip} | {mids} | {rng} | {vol} | {exe} | {secs}s | {'**DIVERGED**' if bad else 'ok'} |")
PY
  touch "$DONE"; say "   $(cat "$ROW")"
}

for D in $DAYS; do run "$D"; done

{ echo ""; echo "| day | condZ_depth min/max | depthClipd | mids | range(tk) | 1s_vol(bp) | exec% | secs | verdict |"
  echo "|---|---|---|---|---|---|---|---|---|"
  for D in $DAYS; do cat "$OUT_DIR/logs/row_d${D}.txt" 2>/dev/null; done; } >> "$SUM"

echo ""; say "DONE — $SUM"
echo "(want: condZ_depth min/max both single-digit, range/vol back toward real ~1.3-1.5bp; big depthClipd + still-huge condZ = clamp fired but did not arrest it)"
echo ""; cat "$SUM"
