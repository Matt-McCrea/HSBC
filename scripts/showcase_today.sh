#!/bin/bash
# showcase_today.sh — a short, present-ready run: the current good models + the stylized-facts
# battery, on one checkpoint, one day (INTC 2015-01-30, 09:30->10:00). ~35-40 min on a free GPU.
#
# WHAT IT PRODUCES (for tomorrow's slides):
#   1. A flow/movement table (cancel%, exec%, ret1s_std, lag1, uniq_mid, touch sizes) for:
#        FROZEN     DDIM10, size-reshape + prior, NO depth-noise   -> the failure (flat mid)
#        dn0.3      DDIM10 + --depth-noise 0.3                      -> the headline fix (alive)
#        dn0.3_dd0.2  headline + --depth-drift 0.2                  -> the volatility variant
#   2. analysis/plots/showcase_battery.png — Real vs those three, 6-panel stylized-facts battery.
#      The FROZEN vs dn0.3 mid-trace is the money shot; the return density/ACF/size/spread panels
#      show dn0.3 is realistic, not just moving.
#
# It is resumable (.done sentinels): if the GPU is reclaimed, rerun and it continues.
#
# Usage:  bash scripts/showcase_today.sh              # auto-discover the checkpoint in CKPT_DIR
#         bash scripts/showcase_today.sh --id 0.627

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior"
REAL="ABIDES/log/market_replay_${TICKER}_2015-01-30_10-00-00_30/processed_orders.csv"
LOB_DIR="data/INTC/INTC_2015-01-02_2015-01-30_10"
OUT_DIR="showcase/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

if pgrep -f "main.py" > /dev/null; then echo "!! training (main.py) running — kill it first."; exit 1; fi
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
echo "flags -> UNCLAMP_DEPTH PRICE_REANCHOR = $PRECHECK"
[[ "$PRECHECK" == "True True" ]] || { echo "!! flags not True True — refusing. Got: $PRECHECK"; exit 1; }

valof () { basename "$1" | sed -E 's/^[^=]*=([0-9.]+).*/\1/'; }
if [[ -z "$ID" ]]; then
  NEWEST=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -1)
  [[ -n "$NEWEST" ]] || { echo "!! no .ckpt in $CKPT_DIR"; exit 1; }
  ID=$(valof "$NEWEST"); echo "auto-discovered: $(basename "$NEWEST") -> -id $ID"
fi
echo "# Showcase — $(date '+%F %T')  ckpt val_ema=$ID" > "$SUM"

[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f "$REAL" ]] || { echo "── regenerating real replay CSV ──"; \
  python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET" > "$OUT_DIR/logs/real_replay.txt" 2>&1; }

movemetric () {
  python3 - "$1" <<'PY' 2>/dev/null
import sys, numpy as np, pandas as pd
try:
    df = pd.read_csv(sys.argv[1]); df["dt"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    for c in ("ask_price_1","bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.ask_price_1.abs()<9e9)&(df.bid_price_1.abs()<9e9)&(df.ask_price_1>0)&(df.bid_price_1>0)]
    df["mid"] = (df.ask_price_1+df.bid_price_1)/2/10000.0
    df = df[df.dt>=pd.Timestamp("2015-01-30 09:45:00")].dropna(subset=["dt","mid"])
    s = df.set_index("dt")["mid"].resample("1s").last().ffill().dropna(); r = np.log(s).diff().dropna()
    print("move: ret1s_std={:.2f}bp  lag1_acf={:.3f}  mid_range_tk={:.0f}  uniq_mid={}".format(
        r.std()*1e4, r.autocorr(1), (df.mid.max()-df.mid.min())*100, df.mid.round(3).nunique()))
except Exception as e:
    print("move: (failed:", e, ")")
PY
}

# run <tag> <extra>  -> generates a 30-min sim, records flow+move, saves the CSV path for the battery
run () {
  local TAG="$1" EXTRA="$2"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"; local CSVFILE="$OUT_DIR/logs/csv_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  echo "$CSV" > "$CSVFILE"
  { echo "## $TAG  (${SECS}s)"; echo '```'; echo "csv: ${CSV:-none}"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REAL" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REAL" --gen "$CSV" 2>&1
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

echo "════ the three present-worthy cells ════"
run "FROZEN"       ""                                   # no depth-noise -> the failure
run "dn0.3"        "--depth-noise 0.3"                  # headline fix
run "dn0.3_dd0.2"  "--depth-noise 0.3 --depth-drift 0.2"  # volatility variant

# ── stylized-facts battery: Real vs the three cells ────────────────────────────────────────────
echo "════ stylized-facts battery ════"
F_FROZEN=$(cat "$OUT_DIR/logs/csv_FROZEN" 2>/dev/null || true)
F_DN03=$(cat "$OUT_DIR/logs/csv_dn0.3" 2>/dev/null || true)
F_DD02=$(cat "$OUT_DIR/logs/csv_dn0.3_dd0.2" 2>/dev/null || true)
BATT=(python evaluation/stylized_custom/battery_reanchored.py
      --date 2015-01-30 --lob-dir "$LOB_DIR"
      --out analysis/plots/showcase_battery.png
      --title "INTC 2015-01-30 09:45-10:00 — the freeze and the fix")
[[ -n "$F_FROZEN" ]] && BATT+=(--series "Frozen (no noise)=$F_FROZEN")
[[ -n "$F_DN03"   ]] && BATT+=(--series "dn0.3=$F_DN03")
[[ -n "$F_DD02"   ]] && BATT+=(--series "dn0.3+dd0.2=$F_DD02")
echo "   ${BATT[*]}"
"${BATT[@]}" 2>&1 | tee "$OUT_DIR/logs/battery.txt"

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Table: $SUM"
echo "  Figure: analysis/plots/showcase_battery.png"
echo "  (email the PNG + summary.md to yourself for the slides — remote has no scp)"
echo "══════════════════════════════════════════"
