#!/bin/bash
# rerun_drift_controller.sh — the generalisation follow-up. Phase 3a of away_run.sh ran the BARE
# 30-min config (dn0.3, NO --dn-target-exec) across the month, and ~8 of 20 days "drifted": the
# fixed sigma over-executed (gen exec 15-17% vs real 3.5-8%) and the closed loop walked the mid out
# of the real envelope (gen uniq_mid 90-187 vs real 27-66). The hypothesis is that the sigma
# feedback controller, which Phase 3a did NOT use at 30-min, pins realised execution to each day's
# real share and so removes the drift. This script tests exactly that: the SAME winning config plus
# --dn-target-exec set to each day's real 30-min execution share, on the drift days, both seeds.
#
# All decode-time on the existing checkpoint (no training, cannot deadlock). Resumable via .done
# sentinels: rerun with --root <same dir> to continue. Compare the output table's exec%/uniq_mid
# against the DRIFT rows in generalisation_summary.md (the no-controller baseline).
#
# Usage:  bash scripts/rerun_drift_controller.sh
#         bash scripts/rerun_drift_controller.sh --root drift_ctrl/20260722_HHMMSS   # resume
#         bash scripts/rerun_drift_controller.sh --days "20150107 20150129" --seeds "30 31"
#         bash scripts/rerun_drift_controller.sh --target-mode fixed --fixed-target 0.05

set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior --depth-noise 0.3"
SEEDS="30 31"
# the 8 drift days from the generalisation run (gen uniq_mid > 50)
DAYS="20150102 20150105 20150107 20150108 20150109 20150123 20150127 20150129"
TARGET_MODE="realexec"     # realexec = pin to each day's real 30-min exec share; fixed = one value
FIXED_TARGET="0.05"
ROOT="drift_ctrl/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --root) ROOT="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;;
  --days) DAYS="$2"; shift 2;; --target-mode) TARGET_MODE="$2"; shift 2;;
  --fixed-target) FIXED_TARGET="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$ROOT/logs"; SUM="$ROOT/drift_controller_summary.md"

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
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$ROOT/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }
echo "# Drift re-run WITH sigma controller — $(date '+%F %T')  ckpt val_ema=$ID  seeds=[$SEEDS]  target=$TARGET_MODE" > "$SUM"

etdash () { echo "${1//:/-}"; }
ymd_dash () { local D="$1"; echo "${D:0:4}-${D:4:2}-${D:6:2}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_30/processed_orders.csv"; }
ensure_real () {
  local D="$1" ETv="$2" RP; RP=$(real_for "$D" "$ETv")
  [[ -f "$RP" ]] || { echo "  ── real replay $D $ETv" >&2; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv" \
      > "$ROOT/logs/real_${D}_$(etdash "$ETv").txt" 2>&1; }
  echo "$RP"
}
# real 30-min execution share (post-09:45), used as the controller target when target-mode=realexec
real_exec_share () {
  python3 - "$1" <<'PY' 2>/dev/null
import sys, pandas as pd, datetime as dt
try:
    df = pd.read_csv(sys.argv[1]); df["dt"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    df = df.dropna(subset=["dt"]); df = df[df.dt.dt.time >= dt.time(9,45,0)]
    tc = "TYPE" if "TYPE" in df.columns else df.columns[[c.upper()=="TYPE" for c in df.columns].index(True)]
    share = (df[tc].astype(str).str.upper() == "ORDER_EXECUTED").mean()
    print("{:.4f}".format(share) if share == share else "")   # blank on NaN
except Exception:
    print("")
PY
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

# run <day> <seed>
runcell () {
  local D="$1" S="$2"; local TAG="ctrl_${D}_s${S}"
  local LOG="$ROOT/logs/${TAG}.txt"; local DONE="$ROOT/logs/.done_${TAG}"; local CSVFILE="$ROOT/logs/csv_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D" "$ET")
  # decide the controller target
  local TGT="$FIXED_TARGET"
  if [[ "$TARGET_MODE" == "realexec" && -f "$REALP" ]]; then
    local RE; RE=$(real_exec_share "$REALP")
    [[ -n "$RE" ]] && TGT="$RE"
  fi
  echo "── $TAG  (target_exec=$TGT)"
  local T; T=$(mktemp); touch "$T"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID" -seed "$S")
  # shellcheck disable=SC2206
  A+=($BASE --dn-target-exec "$TGT"); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$T"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$T" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$T"
  echo "$CSV" > "$CSVFILE"
  { echo "## $TAG  (${SECS}s)  [$D $ST-$ET seed=$S  target_exec=$TGT]"; echo '```'; echo "csv: ${CSV:-none}"; echo "real: $REALP"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

echo "════ drift days WITH controller: $DAYS ════"
for D in $DAYS; do
  for S in $SEEDS; do
    runcell "$D" "$S"
  done
done

# ════ comparison table: new (controller) numbers, with the no-controller baseline for reference ════
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
# baseline (no-controller) DRIFT rows from the Phase-3a generalisation run, for side-by-side reading
BASE = {  # day: (gen_exec_noctrl, gen_uniqmid_s31_noctrl, real_exec)
 "20150102": (16.0, 133, 3.7), "20150105": (15.1, 176, 5.8), "20150107": (15.3, 187, 3.5),
 "20150108": (15.7, 165, 4.3), "20150109": (14.8, 75, 6.0),  "20150123": (15.4, 124, 4.0),
 "20150127": (16.9, 90, 8.1),  "20150129": (15.2, 185, 4.1),
}
rows = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]; tag = head.split('(')[0].strip()
    if 'ERROR' in head or not tag.startswith('ctrl_'): continue
    m = re.match(r'ctrl_(\d{8})_s(\d+)', tag); day, seed = (m.group(1), m.group(2)) if m else ('?','?')
    tgt = re.search(r'target_exec=([\d.]+)', head); tgt = tgt.group(1) if tgt else '-'
    exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block); exe = exe[-1] if exe else '-'
    can = re.findall(r'ORDER_CANCELLED\s+([\d.]+)', block); can = can[-1] if can else '-'
    gu  = re.search(r'gen  move:.*uniq_mid=(\d+)', block); gu = gu.group(1) if gu else '-'
    std = re.search(r'ret1s_std=([\d.]+)bp', block); std = std.group(1) if std else '-'
    b = BASE.get(day, ('-','-','-'))
    rows.append((day[4:], seed, tgt, exe, str(b[0]), gu, str(b[1]), can, std, str(b[2])))
hdr = (f"{'day':>5}{'seed':>5}{'tgt':>7}{'exc':>6}{'exc0':>6}{'gMid':>6}{'gMid0':>7}"
       f"{'can':>6}{'std':>6}{'rExc':>6}")
tab = "\n".join([hdr, '-'*len(hdr)] +
      [f"{d:>5}{s:>5}{t:>7}{e:>6}{e0:>6}{g:>6}{g0:>7}{c:>6}{st:>6}{re:>6}"
       for d,s,t,e,e0,g,g0,c,st,re in rows])
key = ("\n  cols: tgt=controller target  exc/gMid=WITH controller  exc0/gMid0=no-controller baseline"
       "\n  rExc=real exec share.  WIN = exc drops towards rExc AND gMid collapses towards the pinned regime.\n")
print("\n════ CONTROLLER vs BASELINE (drift days) ════"+key+tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n"+key+"\n"+text)
PY

echo ""; echo "══════════════════════════════════════════"
echo "  DRIFT CONTROLLER RE-RUN COMPLETE. Root: $ROOT"
echo "  Summary + comparison table: $SUM"
echo "  Resume: bash scripts/rerun_drift_controller.sh --root $ROOT"
echo "══════════════════════════════════════════"
