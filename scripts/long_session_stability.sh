#!/bin/bash
# long_session_stability.sh — test the two NEW long-horizon stability levers against the 90-min
# lopsided-book divergence. Both are decode-time, additive, default-off (winning 30-min config
# byte-for-byte unchanged), revertible via `git checkout -- ABIDES/`.
#
#   --book-target-thick T (--book-cancel-rate r)  book-balancing spontaneous cancellation: when a
#       side top-of-book size exceeds T x real mean level size, cancel our own resting touch orders
#       to remove r x the excess. Recreates real cancel churn -> targets under-cancel AND divergence.
#   --cond-clip C   clip the z-scored book SIZE conditioning to [-C, C] -> keeps the fed-back state
#       in training support, arresting the drift that grows the touch OOD.
#
# READ (morning table): the ask1 column is the wall-watch. Success = ask1/bid1 stay within ~2x real
# (3899/2117) and non-lopsided, ret1s_std holds near real ~1.5bp (not collapsing < 1), no
# drift_profile cliff. STAGE 1 (30-min smokes) must show the levers DON'T regress the 30-min config.
#
# Usage:  bash scripts/long_session_stability.sh            # auto-discover 0.627
#         bash scripts/long_session_stability.sh --id 0.627

set -uo pipefail
TICKER="INTC"; JAN30="20150130"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior"
OUT_DIR="long_stability/$(date +%Y%m%d_%H%M%S)"
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
echo "# Long-horizon stability — $(date '+%F %T')  ckpt val_ema=$ID" > "$SUM"

[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }

etdash () { echo "${1//:/-}"; }
ymd_dash () { local D="$1"; echo "${D:0:4}-${D:4:2}-${D:6:2}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_30/processed_orders.csv"; }
ensure_real () {
  local D="$1" ETv="$2" RP; RP=$(real_for "$D" "$ETv")
  [[ -f "$RP" ]] || { echo "  ── real replay $D $ETv"; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv" \
      > "$OUT_DIR/logs/real_${D}_$(etdash "$ETv").txt" 2>&1; }
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

# run <tag> <et> <extra> <drift:0|1>
run () {
  local TAG="$1" ETv="$2" EXTRA="$3" DODRIFT="${4:-0}"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"; local CSVFILE="$OUT_DIR/logs/csv_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$JAN30" "$ETv")
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$JAN30" -st "$ST" -et "$ETv"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  echo "$CSV" > "$CSVFILE"
  { echo "## $TAG  (${SECS}s)  [$ST-$ETv]"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1; echo "csv: ${CSV:-none}"; echo "real: $REALP"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    if [[ "$DODRIFT" == "1" && -n "$CSV" && -f "$REALP" ]]; then
      echo ""; echo "-- drift_profile (STABILITY: watch for a cliff / lopsided touch walls) --"
      python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REALP" 2>&1 | head -40
    fi
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

# ── STAGE 1: 30-min smoke — the levers must NOT regress the 30-min winning config ─────────────
echo "════ STAGE 1: 30-min smokes (regression guard) ════"
run "SMOKE_BASE"      "$ET" "--depth-noise 0.3"                              0
run "SMOKE_bt2.0"     "$ET" "--depth-noise 0.3 --book-target-thick 2.0 --book-cancel-rate 0.5" 0
run "SMOKE_cc5"       "$ET" "--depth-noise 0.3 --cond-clip 5"                0

# ── STAGE 2: 90-min stability cells (the point). All carry the exec controller. ───────────────
echo "════ STAGE 2: 90-min stability sweep ════"
run "LH_BASE"         "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045"                                        1
run "LH_bt2.0_r0.5"   "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 --book-target-thick 2.0 --book-cancel-rate 0.5" 1
run "LH_bt2.0_r1.0"   "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 --book-target-thick 2.0 --book-cancel-rate 1.0" 1
run "LH_cc5"          "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 --cond-clip 5"                          1
run "LH_bt2.0_cc5"    "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 --book-target-thick 2.0 --book-cancel-rate 0.5 --cond-clip 5" 1
run "LH_bt3.0_r0.5"   "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 --book-target-thick 3.0 --book-cancel-rate 0.5" 1

# ── STAGE 3: morning table ────────────────────────────────────────────────────────────────────
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read(); rows = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]; tag = head.split('(')[0].strip()
    if 'ERROR' in head: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    can = re.findall(r'ORDER_CANCELLED\s+([\d.]+)', block); can = can[-1] if can else '-'
    exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block); exe = exe[-1] if exe else '-'
    std = g(r'gen  move: ret1s_std=([\d.]+)bp'); lag = g(r'lag1_acf=([-\d.]+)')
    umid = g(r'gen  move:.*uniq_mid=(\d+)')
    bid = re.findall(r'bid_size_1:.*mean=(\d+)', block); bid = bid[-1] if bid else '-'
    ask = re.findall(r'ask_size_1:.*mean=(\d+)', block); ask = ask[-1] if ask else '-'
    bkc = g(r'book_cancels_issued=(\d+)'); clip = g(r'cond_clipped_count=(\d+)')
    rows.append((tag, can, exe, std, lag, umid, bid, ask, bkc, clip))
hdr = f"{'cell':<18}{'Can%':>6}{'Exc%':>6}{'std':>6}{'lag1':>7}{'mids':>6}{'bid1':>7}{'ask1':>8}{'bkCxl':>7}{'clip':>7}"
tab = "\n".join([hdr, '-'*len(hdr),
                 f"{'REAL(30m)':<18}{'43.8':>6}{'7.0':>6}{'1.53':>6}{'-0.09':>7}{'69':>6}{'3899':>7}{'2117':>8}{'-':>7}{'-':>7}"] +
                [f"{t:<18}{c:>6}{e:>6}{s:>6}{l:>7}{u:>6}{b:>7}{a:>8}{bk:>7}{cl:>7}" for t,c,e,s,l,u,b,a,bk,cl in rows])
print("\n════ STABILITY TABLE ════\n"
      "  ask1/bid1 = wall-watch: should stay near real (2117/3899) and NOT go lopsided.\n"
      "  bkCxl = book-balancing cancels issued; clip = z-size entries clipped.\n"
      "  SMOKE_* (30m) must match SMOKE_BASE — levers should not regress the 30-min config.\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary (table at top): $SUM"
echo "  WALL-WATCH: STAGE 2 ask1/bid1 columns + each cell's drift_profile."
echo "  If book-cancel bounds the walls while std holds -> the fix works; LOB-Bench the winner locally."
echo "══════════════════════════════════════════"
