#!/bin/bash
# drift_sigma_sweep.sh — Stage 1 of the 3-day plan: close the cross-day drift question.
#
# The adaptive controller (--dn-target-exec) was refuted on drift days. The untried lever is a
# FIXED lower --depth-noise, with and without the book-balancing cancel. For each drift day we
# sweep sigma and ask: is there ANY setting that is alive-and-stable (not frozen, not drifting), or
# is the day's stable window empty? If empty on every day, the drift is decode-time-unfixable and
# the scheduled-sampling retrain (Stage 3) is the necessary fix.
#
# Each cell also runs drift_profile, whose per-bucket aggr_buy/aggr_sell/B-S closes the mechanism
# question (Stage 1b): the prediction is balanced aggression (B-S ~ 0) even on drift days.
#
# Clones the proven run()/flow_mix harness from scripts/exec_bracket.sh, adds movemetric +
# drift_profile + a per-day regime table. Decode-time only, no training.
#
# Usage:  bash scripts/drift_sigma_sweep.sh                          # 0107 & 0129, seed 30
#         bash scripts/drift_sigma_sweep.sh --days "20150107" --sigmas "0.15 0.20" --seed 31
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""; SEED="30"
DAYS="20150107 20150129"                          # the two clearest drift days
SIGMAS="0.10 0.15 0.20 0.25 0.30"
BOOK="--book-target-thick 2.0 --book-cancel-rate 0.5"   # the second setting per sigma; "" to skip
OUT_DIR="drift_sweep/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --days) DAYS="$2"; shift 2;; --sigmas) SIGMAS="$2"; shift 2;; --book) BOOK="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

if pgrep -f "main.py" > /dev/null; then echo "!! training (main.py) running — kill it first (GPU contention)."; exit 1; fi
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
COLLIDE=$(for f in "$CKPT_DIR"/*.ckpt; do valof "$f"; done | grep -Fxc "$ID" || true)
[[ "${COLLIDE:-0}" -le 1 ]] || { echo "!! $COLLIDE ckpts share val_ema=$ID — archive strays. Refusing."; exit 1; }
echo "# Drift sigma sweep — $(date '+%F %T')  ckpt val_ema=$ID  seed=$SEED  days=[$DAYS]" > "$SUM"

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_30/processed_orders.csv"; }
ensure_real () {
  local D="$1" RP; RP=$(real_for "$D" "$ET")
  [[ -f "$RP" ]] || { echo "  -- real replay $D" >&2; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
      > "$OUT_DIR/logs/real_${D}.txt" 2>&1; }
  echo "$RP"
}
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
    print("move: ret1s_std={:.2f}bp  lag1_acf={:.3f}  mid_range_tk={:.0f}  uniq_mid={}".format(
        r.std()*1e4, r.autocorr(1), (df.mid.max()-df.mid.min())*100, df.mid.round(3).nunique()))
except Exception as e:
    print("move: (failed:", e, ")")
PY
}

# run <tag> <date> <extra>
run () {
  local TAG="$1" D="$2" EXTRA="$3"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D")
  echo "-- $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID" -seed "$SEED"
           --size-reshape --type-decode prior)
  # shellcheck disable=SC2206
  A+=($EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)  [$D seed=$SEED]"; echo '```'; echo "csv: ${CSV:-none}"; echo "real: $REALP"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1
    if [[ -n "$CSV" && -f "$REALP" ]]; then
      echo ""; echo "-- drift_profile (B-S closes the mechanism question) --"
      python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REALP" 2>&1 | head -34
    fi
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

for D in $DAYS; do
  for s in $SIGMAS; do
    run "dd${D}_dn${s}"           "$D" "--depth-noise ${s}"
    [[ -n "$BOOK" ]] && run "dd${D}_dn${s}_bt2.0r0.5" "$D" "--depth-noise ${s} $BOOK"
  done
done

# regime table: per cell classify freeze / alive / drift from uniq_mid + ret1s_std
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read(); rows = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]; tag = head.split('(')[0].strip()
    if 'ERROR' in head: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    std = g(r'gen  move: ret1s_std=([\d.]+)bp'); uniq = g(r'uniq_mid=(\d+)')
    rng = g(r'mid_range_tk=(\d+)'); exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block)
    exe = exe[-1] if exe else '-'
    reg = '-'
    try:
        u = int(uniq); sd = float(std)
        reg = 'FREEZE' if (u <= 9 and sd < 1.0) else ('DRIFT' if u >= 90 else 'alive?')
    except Exception:
        pass
    rows.append((tag, uniq, std, exe, rng, reg))
hdr = f"{'cell':<26}{'uniqMid':>8}{'std':>7}{'exec%':>7}{'rng_tk':>8}{'regime':>9}"
tab = "\n".join([hdr, '-'*len(hdr)] +
                [f"{t:<26}{u:>8}{s:>7}{e:>7}{r:>8}{g:>9}" for t,u,s,e,r,g in rows])
print("\n==== DRIFT SIGMA SWEEP ====\n"
      "  Per drift day: is any sigma 'alive?' (not FREEZE, not DRIFT)? If none, drift is\n"
      "  decode-time-unfixable -> the retrain (Stage 3) is the necessary fix.\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY
echo ""; echo "Done. Summary: $SUM"
echo "READ: per day, is there an 'alive?' sigma, or is the stable window empty?"
