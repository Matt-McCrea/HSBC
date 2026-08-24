#!/bin/bash
# ss_epoch_stability.sh — trial each SS-lineage checkpoint on a short sim and report a one-line
# stability row, so you SELECT the last stable epoch instead of the newest (which may have diverged,
# as val_ema=0.816 did: cond_z[depth] max ~2010, clock frozen at 1us steps, cancels unmatched).
#
# Selection is on SIMULATION stability, not val loss. The decisive column is cond_z[depth] max:
# a healthy run keeps it single-digit; a diverged run runs to tens/hundreds/thousands. Also watch
# 1s-vol (real TSLA ~1.3-1.5 bp) and mid range. This is the epoch-4 selection method from the thesis.
#
# Resumable (.done sentinels). CPU/GPU sim only; refuses to run while training is up (contention).
#
#   bash scripts/ss_epoch_stability.sh                              # auto-derive prior, all TSLA ckpts
#   bash scripts/ss_epoch_stability.sh --type-prior 0.4912,0.4673,0.0416
#   bash scripts/ss_epoch_stability.sh --ckpts 'data/checkpoints/TRADES/*val_ema=0.8*TSLA*.ckpt'
set -uo pipefail

TICKER="TSLA"; START="2015-01-02"; END="2015-01-30"
SIM_DAY="20150130"; ST="10:00:00"; ET="10:30:00"
SIGMA="0.3"; TYPE_PRIOR=""
CKDIR="data/checkpoints/TRADES"
CKPTS=""                          # glob of checkpoints to trial; default = all of the ticker's
CAP_SECS=1200                     # per-cell wall cap; a diverged 30-min run took ~8 min
PY="${PY:-python}"
OUT_DIR="paper_runs/ss_stability_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --ticker) TICKER="$2"; shift 2;; --ckpts) CKPTS="$2"; shift 2;;
  --type-prior) TYPE_PRIOR="$2"; shift 2;; --sigma) SIGMA="$2"; shift 2;;
  --sim-day) SIM_DAY="$2"; shift 2;; --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --cap-secs) CAP_SECS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/stability.md"
say(){ echo "[$(date +%T)] $*"; }
pgrep -f "main.py" >/dev/null && { echo "!! main.py (training) running — kill it first (GPU contention)."; exit 1; }
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT

# type prior (auto-derive from TRAINING days if not passed) — it is a TRAINING/decode input
if [[ -z "$TYPE_PRIOR" ]]; then
  say "deriving $TICKER type prior from TRAINING days…"; DLOG="$OUT_DIR/logs/derive_prior.log"
  bash scripts/derive_type_prior.sh --ticker "$TICKER" --start "$START" --end "$END" > "$DLOG" 2>&1 || true
  TYPE_PRIOR=$(grep -oE 'TYPE_PRIOR = [0-9.,]+' "$DLOG" | tail -1 | sed 's/.*= //')
  [[ -n "$TYPE_PRIOR" ]] || { echo "!! prior derivation failed — see $DLOG. Pass --type-prior."; exit 1; }
fi
say "type prior: $TYPE_PRIOR   sigma: $SIGMA   window: $SIM_DAY $ST-$ET"

[[ -n "$CKPTS" ]] || CKPTS="$CKDIR/*${TICKER}*.ckpt"
# oldest-first, so the table reads in training order
CK_LIST=$(ls -tr $CKPTS 2>/dev/null)
[[ -n "$CK_LIST" ]] || { echo "!! no checkpoints match: $CKPTS"; exit 1; }
say "trialing $(wc -l <<<"$CK_LIST" | tr -d ' ') checkpoints"
echo "# SS epoch stability — $(date '+%F %T')  $TICKER  $ST-$ET  sigma=$SIGMA  prior=$TYPE_PRIOR" > "$SUM"

run(){
  local CK="$1" IDX="$2"; local TAG="ck${IDX}"
  local VE; VE=$(basename "$CK" | sed -E 's/.*val_ema=([0-9.]+).*/\1/')
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local ROW="$OUT_DIR/logs/row_${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" && -f "$ROW" ]] && { say "SKIP $TAG (val_ema=$VE)"; return; }
  say "-- $TAG  val_ema=$VE"
  local T; T=$(mktemp); touch "$T"; local T0; T0=$(date +%s)
  local A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$SIM_DAY" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK" -seed 30
           --depth-noise "$SIGMA" --size-reshape --type-decode prior --type-prior "$TYPE_PRIOR")
  if ! timeout -k 30 "$CAP_SECS" "${A[@]}" > "$LOG" 2>&1; then
    local rc=$?; say "   $([[ $rc -eq 124 || $rc -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$rc") (still parsing what we have)"
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$T" ! -path "*market_replay*" ! -path "*/paper/*" 2>/dev/null | sort | tail -1); rm -f "$T"
  "$PY" - "$LOG" "${CSV:-none}" "$VE" "$SECS" > "$ROW" <<'PY'
import sys, re, numpy as np, pandas as pd
log, csv, ve, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
txt = open(log, errors="ignore").read()
def gmax(feat):
    m = re.search(rf'cond_z\[{feat}\]: min=[-\d.eE+]+ mean=[-\d.eE+]+ max=([-\d.eE+]+)', txt)
    return float(m.group(1)) if m else float('nan')
dmax, tmax = gmax('depth'), gmax('time')
cfb = (re.search(r'cancel_empty_depth_fallbacks=(\d+)', txt) or [None, '-'])[1]
mids = rng = vol = exe = '-'
if csv != "none":
    try:
        d = pd.read_csv(csv); d["dt"] = pd.to_datetime(d.iloc[:, 0], errors="coerce")
        for c in ("ask_price_1", "bid_price_1"): d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[(d.ask_price_1 > 0) & (d.bid_price_1 > 0) & (d.ask_price_1.abs() < 9e9)
              & (d.bid_price_1.abs() < 9e9)].dropna(subset=["dt"])
        d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 1e4
        d = d[d.dt >= d.dt.min() + pd.Timedelta(minutes=15)]
        s = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna(); r = np.log(s).diff().dropna()
        mids = int(s.round(3).nunique()); rng = f"{(s.max()-s.min())*100:.0f}"; vol = f"{r.std()*1e4:.2f}"
        exe = d.TYPE.value_counts(normalize=True).mul(100).round(1).get("ORDER_EXECUTED", "-")
    except Exception as e:
        mids = f"(err {type(e).__name__})"
# verdict: cond_z[depth] must stay single-digit; big value or huge vol = diverged
try:
    bad = (not np.isfinite(dmax)) or dmax > 20 or (vol != '-' and float(vol) > 4)
except Exception:
    bad = True
dmax_s = f"{dmax:.1f}" if np.isfinite(dmax) else "nan"
tmax_s = f"{tmax:.1f}" if np.isfinite(tmax) else "nan"
print(f"| {ve} | {dmax_s} | {tmax_s} | {mids} | {rng} | {vol} | {exe} | {cfb} | {secs}s | {'**DIVERGED**' if bad else 'ok'} |")
PY
  touch "$DONE"; say "   $(cat "$ROW")"
}

i=0; for CK in $CK_LIST; do i=$((i+1)); run "$CK" "$i"; done

# assemble the table in training order
{ echo ""
  echo "| ckpt(val_ema) | condZ_depth_max | condZ_time_max | mids | range(tk) | 1s_vol(bp) | exec% | cxlFallbk | secs | verdict |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
  i=0; for CK in $CK_LIST; do i=$((i+1)); cat "$OUT_DIR/logs/row_ck${i}.txt" 2>/dev/null; done
} >> "$SUM"

echo ""; say "DONE — $SUM"
echo "READ: pick the LAST checkpoint with condZ_depth_max in single digits AND 1s_vol near real (~1.3-1.5 bp)."
echo "      condZ_depth_max in the tens+ = diverged closed loop (val_ema=0.816 was ~2010)."
echo ""; cat "$SUM"
