#!/bin/bash
# flow_balance_sweep.sh — the directional (flow-side) probe for cross-day drift, the pre-retrain
# cheap shot at generalisability. Stage 1 showed the drift is one-sided limit FLOW (limOFI ~-7000),
# NOT variance/execution, so the symmetric levers (depth-noise, book-thinning) can't touch it. The
# --flow-balance lever is the flow-side twin of the book-balancing cancel: it nudges the decoded
# direction of LIMIT orders toward the thin side when recent flow is one-sided. This sweeps its
# strength on the two clearest drift days, on the winning 30-min config.
#
# Success = a flow_balance value bounds the drift (uniq_mid toward real ~27-35, mid back in the real
# envelope, ret1s_std near real) on BOTH days -> decode-time cross-day fix, retrain becomes optional.
#
# Usage:  bash scripts/flow_balance_sweep.sh                       # 0107 & 0129, fb in {0,0.5,1,2,3}
#         bash scripts/flow_balance_sweep.sh --days 20150107 --flows "1.0 1.5"
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""; SEED="30"
DAYS="20150107 20150129"
FLOWS="0 0.5 1.0 2.0 3.0"                       # 0 = baseline (winning config, no flow-balance)
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
OUT_DIR="flow_sweep/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --days) DAYS="$2"; shift 2;; --flows) FLOWS="$2"; shift 2;;
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
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }
echo "# Flow-balance sweep — $(date '+%F %T')  ckpt val_ema=$ID  seed=$SEED  days=[$DAYS]  flows=[$FLOWS]" > "$SUM"

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$ET")_30/processed_orders.csv"; }
ensure_real () {
  local D="$1" RP; RP=$(real_for "$D")
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

run () { # run <tag> <date> <extra>
  local TAG="$1" D="$2" EXTRA="$3"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D")
  echo "-- $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID" -seed "$SEED")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)  [$D seed=$SEED]"; echo '```'; echo "csv: ${CSV:-none}"; echo "real: $REALP"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1
    grep -E "DIAG flow_balance" "$LOG" | tail -1
    if [[ -n "$CSV" && -f "$REALP" ]]; then
      echo ""; echo "-- drift_profile --"; python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REALP" 2>&1 | head -30
    fi
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

for D in $DAYS; do
  for f in $FLOWS; do
    if [[ "$f" == "0" || "$f" == "0.0" ]]; then run "fb${D}_f0" "$D" ""
    else run "fb${D}_f${f}" "$D" "--flow-balance ${f}"; fi
  done
done

# regime table
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read(); rows = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]; tag = head.split('(')[0].strip()
    if 'ERROR' in head: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    std = g(r'gen  move: ret1s_std=([\d.]+)bp'); uniq = g(r'uniq_mid=(\d+)'); rng = g(r'mid_range_tk=(\d+)')
    exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block); exe = exe[-1] if exe else '-'
    imb = g(r'final_limit_imbalance=([-\d.]+)')
    reg = '-'
    try:
        u = int(uniq); sd = float(std)
        reg = 'FREEZE' if (u <= 9 and sd < 1.0) else ('DRIFT' if u >= 90 else 'alive?')
    except Exception:
        pass
    rows.append((tag, uniq, std, exe, rng, imb, reg))
hdr = f"{'cell':<20}{'uniqMid':>8}{'std':>7}{'exec%':>7}{'rng_tk':>8}{'flowImb':>8}{'regime':>9}"
tab = "\n".join([hdr, '-'*len(hdr)] +
                [f"{t:<20}{u:>8}{s:>7}{e:>7}{r:>8}{i:>8}{g:>9}" for t,u,s,e,r,i,g in rows])
print("\n==== FLOW-BALANCE SWEEP ====\n"
      "  Real: 0107 ~27 mids, 0129 ~35 mids, ~13tk range. Does any flow_balance land 'alive?' on BOTH\n"
      "  days (uniq_mid toward real, mid in envelope)? flowImb -> ~0 means the lever balanced the flow.\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY
echo ""; echo "Done. Summary: $SUM"
echo "READ: does any flow_balance bound the drift on BOTH days? If yes -> decode-time cross-day fix."
