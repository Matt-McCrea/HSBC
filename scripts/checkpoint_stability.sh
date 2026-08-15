#!/bin/bash
# checkpoint_stability.sh — trial EVERY checkpoint in data/checkpoints/TRADES on a drift day, on the
# winning 30-min config, to find which one generalises (fixes the cross-day drift). Uses --ckpt-path,
# so checkpoints sharing a rounded val_ema are distinguishable and NOTHING is moved (safe to run while
# training keeps writing new checkpoints). Newest-first (retrain checkpoints trialled first),
# resumable via .done sentinels — partial results are usable if the GPU window runs out.
#
# Real targets: 0107 ~27 unique mids / ~13tk range. STABLE = uniq_mid near real, mid in envelope,
# ret1s_std ~1.5-2.5. DRIFT = uniq_mid >~50, mid walks out. FREEZE = uniq_mid <~10, std <1.
#
# Usage:  bash scripts/checkpoint_stability.sh                       # all ckpts on 0107
#         bash scripts/checkpoint_stability.sh --days "20150107 20150129"
#         bash scripts/checkpoint_stability.sh --oldest-first
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"
CKPT_DIR="data/checkpoints/TRADES"; DAYS="20150107"; SEED="30"; NEWEST_FIRST=1; PRIORITY=""
BASE="--depth-noise 0.3 --size-reshape --type-decode prior"
OUT_DIR="ckpt_stability/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --days) DAYS="$2"; shift 2;; --seed) SEED="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --ckpt-dir) CKPT_DIR="$2"; shift 2;; --oldest-first) NEWEST_FIRST=0; shift;;
  --priority) PRIORITY="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

if pgrep -f "main.py" > /dev/null; then
  echo "!! NOTE: training (main.py) is running — these sims share the GPU and will run slower (contention)."
fi
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRECHECK" == "True True" ]] || { echo "!! flags not True True — refusing. Got: $PRECHECK"; exit 1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }

if [[ "$NEWEST_FIRST" == "1" ]]; then mapfile -t CKPTS < <(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null)
else mapfile -t CKPTS < <(ls "$CKPT_DIR"/*.ckpt 2>/dev/null); fi
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "!! no .ckpt in $CKPT_DIR"; exit 1; }

# --priority "0.724_epoch=0 0.7_epoch=2" pulls matching checkpoints to the front, in the order given
# (e.g. known-good baselines first for a sanity check), leaving the rest in their existing order.
if [[ -n "$PRIORITY" ]]; then
  declare -a PRI_CKPTS=() REST_CKPTS=() ORDERED_PRI=()
  for CK in "${CKPTS[@]}"; do
    matched=0
    for P in $PRIORITY; do [[ "$CK" == *"$P"* ]] && { matched=1; break; }; done
    if [[ $matched -eq 1 ]]; then PRI_CKPTS+=("$CK"); else REST_CKPTS+=("$CK"); fi
  done
  for P in $PRIORITY; do
    for CK in "${PRI_CKPTS[@]}"; do [[ "$CK" == *"$P"* ]] && ORDERED_PRI+=("$CK"); done
  done
  CKPTS=("${ORDERED_PRI[@]}" "${REST_CKPTS[@]}")
  echo "priority order applied: ${ORDERED_PRI[*]##*/}"
fi
echo "# Checkpoint stability — $(date '+%F %T')  days=[$DAYS] seed=$SEED  ${#CKPTS[@]} checkpoints" > "$SUM"
echo "checkpoints: ${#CKPTS[@]}  days: $DAYS  (newest-first=$NEWEST_FIRST)  out: $OUT_DIR"

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

run () {  # run <ckptfile> <day>
  local CK="$1" D="$2"
  local cname; cname=$(basename "$CK" .ckpt)
  local tag; tag="${D}__$(echo "$cname" | sed -E 's/^val_ema=//; s/_INTC.*//')"   # e.g. 20150107__0.715_epoch=3
  local DONE="$OUT_DIR/logs/.done_${tag}"
  [[ -f "$DONE" ]] && { echo "  SKIP $tag"; return; }
  local REALP; REALP=$(ensure_real "$D")
  echo "-- $tag"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  # each cell writes its OWN small file instead of repeatedly appending to one shared
  # summary.md — a single flaky NFS write then only costs one cell, not the whole run, and
  # there's far less repeated open/append contention on one file. Merged into $SUM at the end.
  local CELL="$OUT_DIR/logs/${tag}.summary.md"
  if ! python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
        -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK" -seed "$SEED" $BASE \
        > "$OUT_DIR/logs/${tag}.txt" 2>&1; then
    echo "  ERROR — see logs/${tag}.txt"; echo "## $tag — ERROR" > "$CELL"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  # parse the CSV path straight out of the run's own log first (fast); only fall back to a
  # full ABIDES/log tree scan if that fails — the scan is very slow over NFS on a log dir
  # that's accumulated months of run folders.
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${tag}.txt" | tail -1)
  [[ -n "$CSV" && -f "$CSV" ]] || CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*market_replay*" 2>/dev/null | sort | tail -1)
  rm -f "$S"
  { echo "## $tag  (${SECS}s)"; echo '```'; echo "ckpt: $CK"; echo "csv: ${CSV:-none}"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1 | grep -E "ORDER_EXECUTED|ORDER_CANCELLED|LIMIT_ORDER|unique mid"
    echo '```'; echo ""; } > "$CELL"
  touch "$DONE"; echo "  done ${SECS}s"
}

for D in $DAYS; do
  for CK in "${CKPTS[@]}"; do run "$CK" "$D"; done
done

# merge every cell's own summary file into $SUM now, once, at the end — avoids the repeated
# per-cell append that stalled earlier.
for f in "$OUT_DIR"/logs/*.summary.md; do [[ -f "$f" ]] && cat "$f" >> "$SUM"; done

# ranking table: classify each checkpoint's day-cell freeze / stable / drift
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
    reg = '-'
    try:
        u = int(uniq); sd = float(std); rg = int(rng)
        reg = 'FREEZE' if (u <= 10 and sd < 1.0) else ('DRIFT' if (u >= 50 or rg >= 40) else 'STABLE')
    except Exception:
        pass
    rows.append((tag, uniq, std, rng, exe, reg))
# sort: STABLE first, then by uniq_mid
order = {'STABLE':0,'FREEZE':1,'DRIFT':2,'-':3}
rows.sort(key=lambda r: (order.get(r[5],3), int(r[1]) if r[1].isdigit() else 999))
hdr = f"{'day__checkpoint':<30}{'uniqMid':>8}{'std':>7}{'rng_tk':>8}{'exec%':>7}{'regime':>9}"
tab = "\n".join([hdr, '-'*len(hdr)] +
                [f"{t:<30}{u:>8}{s:>7}{r:>8}{e:>7}{g:>9}" for t,u,s,r,e,g in rows])
print("\n==== CHECKPOINT STABILITY (winning 30-min config on drift day) ====\n"
      "  STABLE = uniq_mid near real (0107~27), mid in envelope. Pick the STABLE checkpoint(s) — NOT\n"
      "  the best val_ema. Then validate cross-day + LOB-Bench.\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY
echo ""; echo "Done (or truncated). Summary: $SUM"
echo "READ: which checkpoint(s) are STABLE on the drift day? That's the generalising one."
