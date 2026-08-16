#!/bin/bash
# away_run.sh — fill a long UNATTENDED GPU window (compute booked until Thursday). Chains, in order
# of information value, everything that is safe to leave running. ALL phases are decode-time
# generation on the existing checkpoint — NO training — so there is no dataloader-deadlock risk (the
# failure mode that hung the epoch-15 retrain). Fully resumable via .done sentinels: if the box is
# reclaimed or you relaunch, pass --root <the same dir> and it continues where it stopped.
#
#   PHASE 1  diagnostics + coverage        scripts/long_session.sh
#              (Stage-1 divergence isolation, under-cancel probes, cross-day coverage)
#   PHASE 2  stability lever sweep          scripts/long_session_stability.sh
#              (the new --book-target-thick / --cond-clip cells at 90 min)
#   PHASE 3  generalisation + robustness    (this script, inline)
#              3a: headline dn0.3 across the WHOLE month x seeds (30-min)  -> kills "one day" limitation
#              3b: the stability combo at 90 min on flat/median/trend days -> do the levers generalise?
#
# Est. ~18-22 h on one free GPU (Phase 3a dominates). If you have >1 GPU, launch the phases in
# separate terminals with different --root subdirs.
#
# Usage:  bash scripts/away_run.sh
#         bash scripts/away_run.sh --root away_run/20260720_HHMMSS   # resume
#         bash scripts/away_run.sh --seeds "30 31 32"                # more robustness seeds

set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior"
LOB_DIR="data/INTC/INTC_2015-01-02_2015-01-30_10"
SEEDS="30 31"
STAB="--book-target-thick 2.0 --book-cancel-rate 0.5 --cond-clip 5"   # stability combo for Phase 3b
ROOT="away_run/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --root) ROOT="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$ROOT/logs"; SUM="$ROOT/generalisation_summary.md"

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
echo "# Away-run generalisation — $(date '+%F %T')  ckpt val_ema=$ID  seeds=[$SEEDS]" > "$SUM"

# ════ PHASE 1 & 2: delegate to the committed sub-scripts (each resumable in its own subdir) ════
echo "════════ PHASE 1: diagnostics + coverage ════════"
bash scripts/long_session.sh           --id "$ID" --out-dir "$ROOT/1_diagnostics" || echo "  (phase 1 returned nonzero — continuing)"
echo "════════ PHASE 2: stability lever sweep ════════"
bash scripts/long_session_stability.sh --id "$ID" --out-dir "$ROOT/2_stability"   || echo "  (phase 2 returned nonzero — continuing)"

# ════ PHASE 3: generalisation + robustness (inline) ════
etdash () { echo "${1//:/-}"; }
ymd_dash () { local D="$1"; echo "${D:0:4}-${D:4:2}-${D:6:2}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_30/processed_orders.csv"; }
ensure_real () {
  local D="$1" ETv="$2" RP; RP=$(real_for "$D" "$ETv")
  [[ -f "$RP" ]] || { echo "  ── real replay $D $ETv"; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv" \
      > "$ROOT/logs/real_${D}_$(etdash "$ETv").txt" 2>&1; }
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
# run3 <tag> <date> <et> <extra> <seed> <drift:0|1>
run3 () {
  local TAG="$1" D="$2" ETv="$3" EXTRA="$4" S="$5" DODRIFT="${6:-0}"
  local LOG="$ROOT/logs/${TAG}.txt"; local DONE="$ROOT/logs/.done_${TAG}"; local CSVFILE="$ROOT/logs/csv_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D" "$ETv")
  echo "── $TAG"
  local T=$(mktemp); touch "$T"; local T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID" -seed "$S")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$T"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$T" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$T"
  echo "$CSV" > "$CSVFILE"
  { echo "## $TAG  (${SECS}s)  [$D $ST-$ETv seed=$S]"; echo '```'; echo "csv: ${CSV:-none}"; echo "real: $REALP"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    [[ -n "$CSV" && -f "$REALP" ]] && python -m evaluation.quantitative_eval.flow_mix --real "$REALP" --gen "$CSV" 2>&1
    if [[ "$DODRIFT" == "1" && -n "$CSV" && -f "$REALP" ]]; then
      echo ""; echo "-- drift_profile --"; python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REALP" 2>&1 | head -40
    fi
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

# enumerate every Jan-2015 trading day present (robust to the LOBSTER folder-name suffix)
DAYS=$(ls data/INTC/*/*message*.csv 2>/dev/null | grep -oE '2015-01-[0-9]{2}' | tr -d '-' | sort -u)
[[ -n "$DAYS" ]] || DAYS="20150130"
echo "days: $DAYS"

echo "════════ PHASE 3a: headline dn0.3 across the month x seeds (30-min) ════════"
for D in $DAYS; do
  for S in $SEEDS; do
    run3 "gen_${D}_s${S}" "$D" "$ET" "--depth-noise 0.3" "$S" 0
  done
done

echo "════════ PHASE 3b: stability combo at 90-min on flat/median/trend days ════════"
# pick flat / median / trend days by realized 09:45-10:00 mid range
mapfile -t TRIO < <(python3 - "$LOB_DIR" <<'PY'
import glob, os, re, sys, numpy as np, pandas as pd
lob = sys.argv[1]; rows = []
cands = sorted(glob.glob(os.path.join(lob, "*message*.csv")) + glob.glob("data/INTC/*/*message*.csv"))
for msgf in dict.fromkeys(cands):
    m = re.search(r'(2015-01-\d{2})', os.path.basename(msgf))
    if not m: continue
    ymd = m.group(1).replace("-", ""); obf = msgf.replace("message", "orderbook")
    if not os.path.exists(obf): continue
    try:
        t = pd.read_csv(msgf, header=None, usecols=[0]).iloc[:, 0].values
        ob = pd.read_csv(obf, header=None, usecols=[0, 2]); ob.columns = ["a", "b"]
        n = min(len(t), len(ob)); mid = (ob["a"].values[:n] + ob["b"].values[:n]) / 2 / 10000.0
        sel = (t[:n] >= 35100) & (t[:n] < 36000); mid = mid[sel]; mid = mid[mid > 0]
        if len(mid) < 10: continue
        rows.append((ymd, (mid.max() - mid.min()) * 100))
    except Exception:
        continue
rows.sort(key=lambda r: r[1])
if rows:
    pick = [rows[0][0], rows[len(rows)//2][0], rows[-1][0]]
    print("\n".join(dict.fromkeys(pick)))
PY
)
[[ ${#TRIO[@]} -gt 0 ]] || TRIO=(20150130)
echo "flat/median/trend: ${TRIO[*]}"
for D in "${TRIO[@]}"; do
  run3 "stab90_${D}" "$D" "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045 $STAB" "30" 1
done

# ════ generalisation table + LOB-Bench manifest ════
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read(); rows = []; manifest = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]; tag = head.split('(')[0].strip()
    if 'ERROR' in head: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    can = re.findall(r'ORDER_CANCELLED\s+([\d.]+)', block); can = can[-1] if can else '-'
    exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block); exe = exe[-1] if exe else '-'
    umids = re.findall(r'unique mid-prices: (\d+)', block)
    ru = umids[0] if len(umids) >= 2 else '-'; gu = g(r'gen  move:.*uniq_mid=(\d+)')
    std = g(r'gen  move: ret1s_std=([\d.]+)bp'); lag = g(r'lag1_acf=([-\d.]+)')
    rows.append((tag, can, exe, std, lag, gu, ru))
    m = re.search(r'^csv: (.+)$', block, flags=re.M)
    if m and m.group(1) != 'none': manifest.append(f"  --gen {tag}={m.group(1).strip()} \\")
hdr = f"{'cell':<20}{'Can%':>6}{'Exc%':>6}{'std':>6}{'lag1':>7}{'genMid':>7}{'realMid':>8}"
tab = "\n".join([hdr, '-'*len(hdr)] +
                [f"{t:<20}{c:>6}{e:>6}{s:>6}{l:>7}{gu:>7}{ru:>8}" for t,c,e,s,l,gu,ru in rows])
print("\n════ GENERALISATION TABLE ════\n"
      "  Per day/seed: genMid vs realMid (coverage), std/lag1 (return moments), Can%/Exc% (flow).\n"
      "  Robustness = agreement across seeds for the same day; generalisation = holding across days.\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
if manifest:
    mf = open(sys.argv[1].replace("generalisation_summary.md", "lobbench_manifest.txt"), "w")
    mf.write("# score locally on the Mac (lob_bench env is not on the remote):\n")
    mf.write("external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \\\n")
    mf.write("\n".join(manifest) + "\n  --out-dir lob_bench_away_run --window 09:45\n")
    print("\nLOB-Bench manifest written:", sys.argv[1].replace('generalisation_summary.md','lobbench_manifest.txt'))
PY

echo ""; echo "══════════════════════════════════════════"
echo "  AWAY-RUN COMPLETE (or truncated gracefully). Root: $ROOT"
echo "  Phase 1: $ROOT/1_diagnostics/summary.md    Phase 2: $ROOT/2_stability/summary.md"
echo "  Phase 3: $ROOT/generalisation_summary.md   LOB-Bench: $ROOT/lobbench_manifest.txt"
echo "  Resume anytime: bash scripts/away_run.sh --root $ROOT"
echo "══════════════════════════════════════════"
