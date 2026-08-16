#!/bin/bash
# long_session.sh — the investigative session behind analysis/PROJECT_STATUS.md open items.
# Three probes, ordered by information-per-minute, resumable via .done sentinels:
#
#   STAGE 1  LONG-HORIZON STABILITY CONTROL (the thing we most need to isolate)
#            The 90-min divergence (lopsided touch walls, ask1~16k) was seen with cb1.0+dd0.2.
#            Both are suspects (dd0.2's AR(1) has phi=0.995, long memory). So run the PLAIN
#            winning config at 90 min — dn0.3 + controller, NO drift, NO cancel-boost — and read
#            the drift_profile. If it STILL diverges -> genuine book accumulation (treat the
#            cancel side). If it does NOT -> the drift was causing it (de-persist/drop drift long).
#
#   STAGE 2  UNDER-CANCEL HYPOTHESES (cheap 30-min probes, no code changes)
#            H1 dilution: cancel% is low partly because too many limits CROSS (channel B) and
#               dilute the mix. Lower sigma -> fewer crossings -> does cancel% rise toward 44%?
#               (dn0.5/0.6 lowered it; dn0.16/0.2 should raise it if the dilution story holds.)
#            H2 type head: does --type-decode l1/l2 decode more cancels than 'prior'?
#
#   STAGE 3  MID-COVERAGE ACROSS DAYS (is low uniq_mid real, or a Jan-30 trend artefact?)
#            Scan the month for realized 09:45-10:00 range, then run the headline config on the
#            FLATTEST couple of days + the median + the most-trending + Jan-30. If real ALSO
#            visits ~15-20 mids on a flat day and we match it, coverage is fine.
#
#   STAGE 4  LONG-HORIZON VARIANTS (expensive, last): controller on/off, lower sigma.
#
# All cells: DDIM10, size-reshape + type prior, ckpt auto-discovered. Nothing destructive; the
# new-flag code is additive + default-off, revert with `git checkout -- ABIDES/`.
#
# Usage:  bash scripts/long_session.sh            # auto-discover 0.627
#         bash scripts/long_session.sh --id 0.627

set -uo pipefail
TICKER="INTC"; JAN30="20150130"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior"
LOB_DIR="data/INTC/INTC_2015-01-02_2015-01-30_10"
OUT_DIR="long_session/$(date +%Y%m%d_%H%M%S)"
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
echo "# Long session — $(date '+%F %T')  ckpt val_ema=$ID" > "$SUM"

[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }

etdash () { echo "${1//:/-}"; }
ymd_dash () { local D="$1"; echo "${D:0:4}-${D:4:2}-${D:6:2}"; }
real_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_30/processed_orders.csv"; }
ensure_real () { # ensure_real <date> <et>
  local D="$1" ETv="$2" RP; RP=$(real_for "$D" "$ETv")
  [[ -f "$RP" ]] || { echo "  ── real replay $D $ETv"; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv" \
      > "$OUT_DIR/logs/real_${D}_$(etdash "$ETv").txt" 2>&1; }
  echo "$RP"
}

# date-agnostic movement metric (filters by time-of-day >= 09:45, works on any trading day)
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

# run <tag> <date> <et> <extra> <drift:0|1>
run () {
  local TAG="$1" D="$2" ETv="$3" EXTRA="$4" DODRIFT="${5:-0}"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"; local CSVFILE="$OUT_DIR/logs/csv_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D" "$ETv")
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA); echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  echo "$CSV" > "$CSVFILE"
  { echo "## $TAG  (${SECS}s)  [$D $ST-$ETv]"; echo '```'
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

# ── STAGE 0: day-range scan → pick coverage days; real replays for Jan-30 ──────────────────────
echo "════ STAGE 0: day-range scan ════"
python3 - "$LOB_DIR" "$OUT_DIR/selected_days.txt" <<'PY' | tee "$OUT_DIR/logs/day_scan.txt"
import glob, os, re, sys, numpy as np, pandas as pd
lob, outf = sys.argv[1], sys.argv[2]; JAN30 = "20150130"; rows = []
for msgf in sorted(glob.glob(os.path.join(lob, "*_message_10.csv"))):
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', os.path.basename(msgf))
    if not m: continue
    ymd = "".join(m.groups()); obf = msgf.replace("message", "orderbook")
    if not os.path.exists(obf): continue
    try:
        t = pd.read_csv(msgf, header=None, usecols=[0]).iloc[:, 0].values
        ob = pd.read_csv(obf, header=None, usecols=[0, 2]); ob.columns = ["ask", "bid"]
        n = min(len(t), len(ob)); t = t[:n]; ob = ob.iloc[:n]
        mid = (ob["ask"].values + ob["bid"].values) / 2 / 10000.0
        sel = (t >= 35100) & (t < 36000); mid = mid[sel]; mid = mid[mid > 0]
        if len(mid) < 10: continue
        rows.append((ymd, (mid.max()-mid.min())*100, len(np.unique(np.round(mid, 3))), len(mid)))
    except Exception:
        continue
rows.sort(key=lambda r: r[1])  # flattest first
print("day-range scan (09:45-10:00), flattest first:")
print(f"  {'day':10}{'range_tk':>10}{'uniq_mid':>10}{'events':>9}")
for ymd, rng, uniq, ne in rows: print(f"  {ymd:10}{rng:>10.0f}{uniq:>10}{ne:>9}")
sel = []
if rows:
    sel += [rows[0][0]]
    if len(rows) > 1: sel += [rows[1][0]]
    sel += [rows[len(rows)//2][0], rows[-1][0]]
sel = list(dict.fromkeys(sel))
if JAN30 not in sel: sel.append(JAN30)
sel = sel[:5]
open(outf, "w").write(" ".join(sel)); print("selected coverage days (flattest first):", " ".join(sel))
PY
SELECTED_DAYS=$(cat "$OUT_DIR/selected_days.txt" 2>/dev/null || echo "$JAN30")
FLAT_DAY=$(echo "$SELECTED_DAYS" | awk '{print $1}')

# ── STAGE 1: long-horizon stability control (isolate — no drift, no cancel-boost) ──────────────
echo "════ STAGE 1: long-horizon stability CONTROL (plain config, 90 min) ════"
run "LH_dn0.3_ctrl" "$JAN30" "$ET_LONG" "--depth-noise 0.3 --dn-target-exec 0.045" 1

# ── STAGE 2: under-cancel hypotheses (30-min, cheap) ───────────────────────────────────────────
echo "════ STAGE 2: under-cancel probes ════"
run "dn0.16"        "$JAN30" "$ET" "--depth-noise 0.16"          # H1 dilution: fewer crossings -> cancel% up?
run "dn0.2"         "$JAN30" "$ET" "--depth-noise 0.2"
run "dn0.3_typeL1"  "$JAN30" "$ET" "--depth-noise 0.3 --type-decode l1"   # H2: does l1 head cancel more?
run "dn0.3_typeL2"  "$JAN30" "$ET" "--depth-noise 0.3 --type-decode l2"

# ── STAGE 3: mid-coverage across days (headline config on the selected days) ───────────────────
echo "════ STAGE 3: cross-day coverage (dn0.3 vs real per day) ════"
for D in $SELECTED_DAYS; do
  run "cov_${D}" "$D" "$ET" "--depth-noise 0.3"
done

# ── STAGE 4: long-horizon variants (expensive, last) ───────────────────────────────────────────
echo "════ STAGE 4: long-horizon variants ════"
run "LH_dn0.3_noctrl" "$JAN30" "$ET_LONG" "--depth-noise 0.3" 1                        # isolate the controller
run "LH_dn0.2_ctrl"   "$JAN30" "$ET_LONG" "--depth-noise 0.2 --dn-target-exec 0.045" 1  # thinner book -> less divergence?

# ── STAGE 5: morning table + a coverage figure on the flattest day ─────────────────────────────
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
    umids = re.findall(r'unique mid-prices: (\d+)', block)
    real_u = umids[0] if len(umids) >= 2 else '-'; gen_u = umids[-1] if umids else '-'
    std = g(r'gen  move: ret1s_std=([\d.]+)bp'); lag = g(r'lag1_acf=([-\d.]+)')
    ask = re.findall(r'ask_size_1:.*mean=(\d+)', block); ask = ask[-1] if ask else '-'
    rows.append((tag, can, exe, std, lag, gen_u, real_u, ask))
hdr = f"{'cell':<22}{'Can%':>6}{'Exc%':>6}{'std_bp':>8}{'lag1':>7}{'genMid':>7}{'realMid':>8}{'ask1':>8}"
tab = "\n".join([hdr, '-'*len(hdr),
                 f"{'REAL(Jan30 30m)':<22}{'43.8':>6}{'7.0':>6}{'1.53':>8}{'-0.09':>7}{'-':>7}{'69':>8}{'2117':>8}"] +
                [f"{t:<22}{c:>6}{e:>6}{s:>8}{l:>7}{gu:>7}{ru:>8}{a:>8}" for t,c,e,s,l,gu,ru,a in rows])
print("\n════ LONG SESSION TABLE ════\n"
      "  STAGE1/4 (LH_*): read ask1 — if it balloons/goes lopsided the book still diverges.\n"
      "  STAGE2 (dn0.16/0.2, typeL1/L2): does Can% rise toward 43.8?\n"
      "  STAGE3 (cov_*): compare genMid vs realMid PER DAY — if real is also low on a flat day and\n"
      "                  we match it, coverage is fine (Jan-30's 69 is trend-inflated).\n" + tab)
open(sys.argv[1], 'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY

# coverage figure on the flattest day (Real vs dn0.3), if that cell produced a CSV
FLAT_CSV=$(cat "$OUT_DIR/logs/csv_cov_${FLAT_DAY}" 2>/dev/null || true)
if [[ -n "$FLAT_CSV" ]]; then
  echo "════ coverage battery on flattest day $FLAT_DAY ════"
  python evaluation/stylized_custom/battery_reanchored.py \
    --date "$(ymd_dash "$FLAT_DAY")" --lob-dir "$LOB_DIR" \
    --series "dn0.3=$FLAT_CSV" \
    --out "analysis/plots/coverage_${FLAT_DAY}.png" \
    --title "Coverage check — INTC $(ymd_dash "$FLAT_DAY") 09:45-10:00 (flattest day)" \
    2>&1 | tee "$OUT_DIR/logs/coverage_battery.txt"
fi

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary (table at top): $SUM"
echo "  Coverage figure (if made): analysis/plots/coverage_${FLAT_DAY}.png"
echo "  KEY READS: STAGE1 drift_profile (does plain config diverge?), STAGE2 Can% (dilution?),"
echo "             STAGE3 genMid-vs-realMid per day (is coverage real?)."
echo "══════════════════════════════════════════"
