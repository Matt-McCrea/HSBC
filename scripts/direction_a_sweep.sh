#!/bin/bash
# direction_a_sweep.sh — DIRECTION A: make price impact realistic + raise mid-price variance,
# WITHOUT faking the trend, and WITHOUT breaking the known-good config.
#
# THE TWO NEW LEVERS (both decode-time, default OFF, so dn0.3 is byte-for-byte unchanged):
#   --cancel-boost B    bias type-decode toward CANCEL -> thinner book -> higher price impact
#                       (targets the under-cancel/thick-book gap; the impact fix for exec-agent use)
#   --depth-drift A     AR(1) directional persistence on depth -> transient mid excursions that
#                       mean-revert (no net trend) -> higher realized vol, less over-bounce
#
# REVERT SAFETY:
#   * The current best config (dn0.3) runs FIRST as the reference cell — every lever is judged
#     against it in the SAME run. If a lever regresses flow/stability, you just don't use its flag;
#     the winning config is untouched (the code changes are additive + default-off).
#   * Nothing here is destructive. To fully revert the code: `git checkout -- ABIDES/agent/WorldAgent.py
#     ABIDES/config/world_agent_sim.py` (or don't pass the new flags).
#
# WHAT TO READ (the A-specific metrics, in the morning table):
#   impact  -> bid1/ask1 mean should FALL toward real (3899/2117); thinner = more impact
#   move    -> ret1s_std (bp) and mid_range/uniq_mids should RISE vs baseline; |lag1_acf| should FALL
#              toward real's ~-0.10 (baseline over-bounces at ~-0.19..-0.22)
#   flow    -> cancel% should RISE toward real 44% (esp. under --cancel-boost)
#   STABILITY: the drift_profile block per cell — watch for a min-XX cliff, spread blow-up, or
#              cond_z[price] leaving support. Higher variance CAN tip into teleport/divergence; if a
#              cell's drift shows that, that lever/level is too strong.
#
# Usage:  bash scripts/direction_a_sweep.sh                 # auto-discover 0.627
#         bash scripts/direction_a_sweep.sh --id 0.627

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
CKPT_DIR="data/checkpoints/TRADES"; ID=""
BASE="--size-reshape --type-decode prior"                 # shared with the winning config
REAL="ABIDES/log/market_replay_${TICKER}_2015-01-30_10-00-00_30/processed_orders.csv"
OUT_DIR="direction_a/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

if pgrep -f "main.py" > /dev/null; then
  echo "!! training (main.py) running — kill it first. Refusing."; exit 1
fi
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
echo "# Direction A sweep — $(date '+%F %T')  ckpt val_ema=$ID" > "$SUM"

[[ -f data/quantile_targets/real_size_limit.npy ]] || python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing — refusing."; exit 1; }
[[ -f "$REAL" ]] || { echo "── regenerating real replay CSV ──"; \
  python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET" > "$OUT_DIR/logs/real_replay.txt" 2>&1; }

# movement metric: 1s-return std (bp), lag-1 autocorr, mid range (ticks), unique mids, from a CSV.
movemetric () {
  python3 - "$1" <<'PY' 2>/dev/null
import sys, numpy as np, pandas as pd
try:
    df = pd.read_csv(sys.argv[1])
    df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.ask_price_1.abs() < 9e9) & (df.bid_price_1.abs() < 9e9) & (df.ask_price_1 > 0) & (df.bid_price_1 > 0)]
    df["mid"] = (df.ask_price_1 + df.bid_price_1) / 2 / 10000.0
    df = df[df.dt >= pd.Timestamp("2015-01-30 09:45:00")].dropna(subset=["dt", "mid"])
    s = df.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
    r = np.log(s).diff().dropna()
    print("move: ret1s_std={:.2f}bp  lag1_acf={:.3f}  mid_range_tk={:.0f}  uniq_mid={}".format(
        r.std()*1e4, r.autocorr(1), (df.mid.max()-df.mid.min())*100, df.mid.round(3).nunique()))
except Exception as e:
    print("move: (failed:", e, ")")
PY
}
echo "── REAL movement reference (trend-inflated upper bound) ──"
[[ -f "$REAL" ]] && { echo -n "REAL "; movemetric "$REAL"; } | tee -a "$SUM"

run () { # run <tag> <extra> [et_override]
  local TAG="$1" EXTRA="$2" ET_RUN="${3:-$ET}"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET_RUN"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id "$ID")
  # shellcheck disable=SC2206
  A+=($BASE $EXTRA)
  echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1; echo "csv: ${CSV:-none}"
    [[ -n "$CSV" ]] && { echo -n "gen  "; movemetric "$CSV"; }
    if [[ -n "$CSV" && -f "$REAL" ]]; then python -m evaluation.quantitative_eval.flow_mix --real "$REAL" --gen "$CSV" 2>&1
    elif [[ -n "$CSV" ]]; then python -m evaluation.quantitative_eval.flow_mix --gen "$CSV" 2>&1; fi
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo ""; echo "-- drift_profile (STABILITY: watch for a cliff / spread blow-up) --"
    [[ -n "$CSV" && -f "$REAL" ]] && python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REAL" 2>&1 | head -30
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

echo "════ REFERENCE: current best (revert target) ════"
run "BASE_dn0.3"          "--depth-noise 0.3"

echo "════ IMPACT lever: --cancel-boost (thin book -> more impact) ════"
run "dn0.3_cb0.5"         "--depth-noise 0.3 --cancel-boost 0.5"
run "dn0.3_cb1.0"         "--depth-noise 0.3 --cancel-boost 1.0"
run "dn0.3_cb2.0"         "--depth-noise 0.3 --cancel-boost 2.0"

echo "════ VOLATILITY lever: --depth-drift (persistence -> more move, no trend) ════"
run "dn0.3_dd0.1"         "--depth-noise 0.3 --depth-drift 0.1"
run "dn0.3_dd0.2"         "--depth-noise 0.3 --depth-drift 0.2"
run "dn0.3_dd0.3"         "--depth-noise 0.3 --depth-drift 0.3"

echo "════ COMBINED (impact + volatility) ════"
run "dn0.3_cb1.0_dd0.2"   "--depth-noise 0.3 --cancel-boost 1.0 --depth-drift 0.2"

echo "════ crossing-frequency ceiling (cheap σ check) ════"
run "dn0.5"               "--depth-noise 0.5"
run "dn0.6"               "--depth-noise 0.6"

echo "════ one 75-min stability check on the combined (does it survive the long horizon?) ════"
[[ -f "ABIDES/log/market_replay_${TICKER}_2015-01-30_11-00-00_30/processed_orders.csv" ]] || \
  python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET_LONG" > "$OUT_DIR/logs/real_replay_1100.txt" 2>&1
run "dn0.3_cb1.0_dd0.2_ET1100" "--depth-noise 0.3 --cancel-boost 1.0 --depth-drift 0.2 --dn-target-exec 0.045" "$ET_LONG"

# ── morning table + LOB-Bench manifest ────────────────────────────────────────────────────────
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
rows, manifest = [], []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    tag = block.splitlines()[0].split('(')[0].strip()
    if 'ERROR' in block.splitlines()[0]: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    can = re.findall(r'ORDER_CANCELLED\s+([\d.]+)', block); can = can[-1] if can else '-'
    exe = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block); exe = exe[-1] if exe else '-'
    std = g(r'ret1s_std=([\d.]+)bp'); lag = g(r'lag1_acf=([-\d.]+)')
    rng = g(r'mid_range_tk=([\d.]+)'); umid = g(r'uniq_mid=(\d+)')
    bid = re.findall(r'bid_size_1:.*mean=(\d+)', block); bid = bid[-1] if bid else '-'
    ask = re.findall(r'ask_size_1:.*mean=(\d+)', block); ask = ask[-1] if ask else '-'
    rows.append((tag, can, exe, std, lag, rng, umid, bid, ask))
    m = re.search(r'^csv: (.+)$', block, flags=re.M)
    if m and m.group(1) != 'none': manifest.append(f"  --gen {tag}={m.group(1)} \\")
hdr = f"{'cell':<26}{'Can%':>6}{'Exc%':>6}{'std_bp':>8}{'lag1':>7}{'rng_tk':>7}{'mids':>6}{'bid1':>8}{'ask1':>8}"
tab = "\n".join([hdr, '-'*len(hdr),
                 f"{'REAL(target)':<26}{'43.8':>6}{'7.0':>6}{'1.53':>8}{'-0.09':>7}{'44*':>7}{'88*':>6}{'3899':>8}{'2117':>8}"] +
                [f"{t:<26}{c:>6}{e:>6}{s:>8}{l:>7}{r:>7}{u:>6}{b:>8}{a:>8}" for t,c,e,s,l,r,u,b,a in rows])
print("\n════ DIRECTION A TABLE ════\n"
      "  std_bp is ALREADY ~real (baseline 1.4 vs real 1.5) — do NOT chase it.\n"
      "  REAL GAPS: |lag1| too big (baseline -0.19 vs real -0.09 = over-bounce -> --depth-drift),\n"
      "             bid1/ask1 too thick (impact too cheap -> --cancel-boost), cancel% too low.\n"
      "  *range/mids are TREND-inflated in real — not a target; ignore.\n" + tab)
if manifest:
    print("\n════ LOB-Bench manifest (score locally) ════")
    print("external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \\")
    print("  --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-30_34140000_57660000_orderbook_10.csv \\")
    print("\n".join(manifest)); print("  --out-dir lob_bench_direction_a --window 09:45")
open(sys.argv[1],'w').write("# TABLE\n```\n"+tab+"\n```\n\n"+text)
PY

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary (table at top): $SUM"
echo "  REVERT: baseline is cell BASE_dn0.3; if no lever beats it on impact+move WITHOUT breaking"
echo "  stability (drift_profile) or flow, keep dn0.3. Code reverts via: git checkout -- ABIDES/"
echo "══════════════════════════════════════════"
