#!/bin/bash
# tsla_paper_runs.sh — the two-hour deliverables for a new ticker, plus the vanilla DDPM baseline.
#
# PRODUCES, on the two held-out test days at a two-hour window:
#   A) DDIM-10 + decode corrections, with the type prior set to THIS stock's real marginals
#   B) DDPM-100 with NO decode interventions — the un-accelerated baseline to measure A against
# and a short prior-validation cell first, so a mis-set prior is caught in ~5 minutes rather than
# after four two-hour runs.
#
# WHY THE PRIOR MATTERS HERE. --type-decode prior used INTC's marginals [0.49,0.48,0.03] for every
# stock. On TSLA (real flow 44.9/38.4/16.7) that pinned generated market orders to Intel's 3%: flow
# sat at 51/45/3 and execution share at 2.7-4.2% for EVERY sigma from 0.15 to 3.0. The prior is
# derived here from the stock's own real data rather than hardcoded again.
#
# ⚠️ WHAT "VANILLA" MEANS IN CELL B. No decode-time interventions (no depth-noise, size-reshape or
# type-decode). The data-pipeline flags UNCLAMP_DEPTH / PRICE_REANCHOR stay ON because the
# checkpoint was TRAINED with them and simulation must match training. So B is the "no decode fixes"
# baseline, NOT a from-scratch replication of the published TRADES pipeline — that would need a
# checkpoint trained without the data fixes. Label it accordingly in the write-up.
#
#   bash scripts/tsla_paper_runs.sh --dry-run
#   nohup bash scripts/tsla_paper_runs.sh > tsla_paper.log 2>&1 & disown
set -uo pipefail

TICKER="TSLA"; START="2015-01-02"; END="2015-01-30"
DAYS=""                       # default: the two held-out test days (last two of the period)
ST="10:00:00"; ET="12:00:00"  # the two-hour window
SMOKE_ST="09:30:00"; SMOKE_ET="10:00:00"
SEED="30"; SIGMA="0.6"
CKPT=""                       # default: newest checkpoint in the dir
CAP_SECS=10800                # 3h per cell; TSLA 30-min cells ran 89-587s, 2h is unmeasured
TYPE_PRIOR=""                 # default: derived from this stock's real data
SKIP_SMOKE=0; DRY=0
PY="${PY:-python}"
OUT_DIR=""                    # defaulted after arg parsing so --ticker is reflected in the path

while [[ $# -gt 0 ]]; do case "$1" in
  --ticker) TICKER="$2"; shift 2;; --start) START="$2"; shift 2;; --end) END="$2"; shift 2;;
  --days) DAYS="$2"; shift 2;; --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --sigma) SIGMA="$2"; shift 2;; --ckpt-path) CKPT="$2"; shift 2;;
  --type-prior) TYPE_PRIOR="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --cap-secs) CAP_SECS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --skip-smoke) SKIP_SMOKE=1; shift;; --dry-run) DRY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

[[ -n "$OUT_DIR" ]] || OUT_DIR="paper_runs/${TICKER}_$(date +%Y%m%d_%H%M%S)"
export TICKER TRADING_START="$START" TRADING_END="$END"

DATA_DIR="data/${TICKER}/${TICKER}_${START}_${END}"
[[ -d "$DATA_DIR" ]] || DATA_DIR=$(ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1)

# Held-out test days = the last two of the period (SPLIT_RATES .85/.05/.10, chronological by day).
if [[ -z "$DAYS" ]]; then
  DAYS=$(ls "$DATA_DIR" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u | tail -2 | tr -d '-' | tr '\n' ' ')
fi
[[ -n "$CKPT" ]] || CKPT=$(ls -t data/checkpoints/TRADES/*.ckpt 2>/dev/null | head -1)

# Hard requirements are enforced only for a real run; --dry-run still prints the plan so the
# layout can be checked before any data is in place.
if [[ "$DRY" != "1" ]]; then
  [[ -n "$DAYS" ]] || { echo "!! no days found in ${DATA_DIR:-<missing>}"; exit 1; }
  [[ -n "$CKPT" && -f "$CKPT" ]] || { echo "!! no checkpoint found (pass --ckpt-path)"; exit 1; }
fi

say () { echo "[$(date +%T)] $*"; }
if [[ -n "$CKPT" ]]; then CKPT_LABEL=$(basename "$CKPT"); else CKPT_LABEL="<none found — pass --ckpt-path>"; fi

if [[ "$DRY" == "1" ]]; then
  cat <<EOF
=== $TICKER paper runs ===
data     : ${DATA_DIR:-MISSING}
ckpt     : $CKPT_LABEL
test days: ${DAYS:-<none — data not found>}
window   : $ST-$ET   (smoke: $SMOKE_ST-$SMOKE_ET)
sigma    : $SIGMA
prior    : ${TYPE_PRIOR:-<derived from real data>}
out      : $OUT_DIR

cells:
  smoke              DDIM-10 + fixes + prior, 30-min, first test day     ~5-10 min
$(for d in $DAYS; do echo "  A_ddim10_$d       DDIM-10 + fixes + prior, 2h                        ~30-60 min"; done)
$(for d in $DAYS; do echo "  B_ddpm100_van_$d  DDPM-100, NO interventions, 2h                     ~1-3 h"; done)

Cell B keeps UNCLAMP_DEPTH/PRICE_REANCHOR (the checkpoint was trained with them) but passes no
decode flags. It is the "no decode fixes" baseline, not a published-TRADES replication.
EOF
  exit 0
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/real"
SUM="$OUT_DIR/summary.md"; : > "$SUM"

# ---- preflight -------------------------------------------------------------
pgrep -f "main.py" >/dev/null && { echo "!! training running — kill it first (GPU contention)"; exit 1; }
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRE=$("$PY" -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
[[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }
say "ckpt: $(basename "$CKPT")   days: $DAYS"

# ---- real references + the type prior --------------------------------------
# Both are CPU-only. The prior is [limit, cancel, market] from the stock's own real next-event
# marginals: LOBSTER type 1 -> limit, 2/3 -> cancel, 4/5/6 -> execution (i.e. an incoming
# marketable order), which is the quantity the model's MARKET class represents.
for D in $DAYS; do
  R="$OUT_DIR/real/real_${TICKER}_${D}.csv"
  [[ -f "$R" ]] || "$PY" -m evaluation.stylized_custom.lobster_real_reference \
      --ticker "$TICKER" --date "$D" --st "$ST" --et "$ET" --out "$R" \
      > "$OUT_DIR/logs/real_${D}.txt" 2>&1 || { echo "!! real reference failed for $D"; cat "$OUT_DIR/logs/real_${D}.txt"; exit 1; }
  say "real reference $D -> $R"
done

FIRST_DAY=$(awk '{print $1}' <<< "$DAYS")
if [[ -z "$TYPE_PRIOR" ]]; then
  TYPE_PRIOR=$("$PY" - "$OUT_DIR/real/real_${TICKER}_${FIRST_DAY}.csv" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
m = d.TYPE.value_counts(normalize=True)
lim, can, ex = (float(m.get(k, 0.0)) for k in ("LIMIT_ORDER", "ORDER_CANCELLED", "ORDER_EXECUTED"))
t = lim + can + ex
print(f"{lim/t:.4f},{can/t:.4f},{ex/t:.4f}")
PY
)
  say "type prior derived from real $FIRST_DAY: $TYPE_PRIOR  (INTC's is 0.49,0.48,0.03)"
fi
echo "# $TICKER paper runs — $(date '+%F %T')" >> "$SUM"
echo "ckpt \`$(basename "$CKPT")\` · window $ST-$ET · sigma $SIGMA · type-prior \`$TYPE_PRIOR\`" >> "$SUM"
echo "" >> "$SUM"

# ---- runner ----------------------------------------------------------------
run () { # run <tag> <date> <st> <et> <sampler> <nsteps> <extra flags>
  local TAG="$1" D="$2" S="$3" E="$4" SAMP="$5" NST="$6" EXTRA="$7"
  local DONE="$OUT_DIR/logs/.done_${TAG}" LOG="$OUT_DIR/logs/${TAG}.txt"
  [[ -f "$DONE" ]] && { say "SKIP $TAG (done)"; return 0; }
  say "-- $TAG   [$SAMP-$NST $D $S-$E]"
  local T0; T0=$(date +%s)
  local A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$S" -et "$E"
           -d True -m TRADES -type "$SAMP" -nsteps "$NST" -eta 0.0 --ckpt-path "$CKPT" -seed "$SEED")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA" ]] && A+=($EXTRA)
  if ! timeout -k 30 "$CAP_SECS" "${A[@]}" > "$LOG" 2>&1; then
    local RC=$?; local SECS=$(( $(date +%s) - T0 ))
    local M="ERROR rc=$RC"; [[ $RC -eq 124 || $RC -eq 137 ]] && M="TIMEOUT ${CAP_SECS}s"
    say "   $M after $((SECS/60))m"; echo "## $TAG — **$M**" >> "$SUM"; return 1
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$LOG" | tail -1)
  echo "$CSV" > "$OUT_DIR/logs/.csv_${TAG}"; touch "$DONE"
  say "   done $((SECS/60))m $((SECS%60))s"
  { echo "## $TAG  ($((SECS/60))m)"; echo '```'
    grep -E "^\[WorldAgent\] type prior" "$LOG" | tail -1
    echo "csv: ${CSV:-none}"
    local R="$OUT_DIR/real/real_${TICKER}_${D}.csv"
    [[ -n "$CSV" && -f "$R" ]] && "$PY" -m evaluation.quantitative_eval.flow_mix --real "$R" --gen "$CSV" 2>&1
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo '```'; echo ""; } >> "$SUM"
}

FIXES="--depth-noise $SIGMA --size-reshape --type-decode prior --type-prior $TYPE_PRIOR"

# ---- 1. prior smoke: catch a mis-set prior in minutes, not after four 2h runs
if [[ "$SKIP_SMOKE" == "0" ]]; then
  run "smoke_prior_${FIRST_DAY}" "$FIRST_DAY" "$SMOKE_ST" "$SMOKE_ET" DDIM 10 "$FIXES" || true
  say "CHECK the smoke cell in $SUM: generated flow should move toward real, NOT sit at 51/45/3."
  say "If it still reads ~51/45/3 the prior is not taking effect — stop and investigate."
fi

# ---- 2. A: the model, two hours, both test days
for D in $DAYS; do run "A_ddim10_${D}" "$D" "$ST" "$ET" DDIM 10 "$FIXES" || true; done

# ---- 3. B: DDPM-100, no decode interventions, two hours, both test days
for D in $DAYS; do run "B_ddpm100_vanilla_${D}" "$D" "$ST" "$ET" DDPM 100 "" || true; done

# ---- summary table ---------------------------------------------------------
"$PY" - "$OUT_DIR" "$TICKER" <<'PY' | tee -a "$SUM"
import sys, os, glob, numpy as np, pandas as pd, datetime as dt
out, tick = sys.argv[1], sys.argv[2]
def stat(p, warm=15):
    df = pd.read_csv(p); df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1"): df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df[(df.ask_price_1 > 0) & (df.bid_price_1 > 0) & (df.ask_price_1.abs() < 9e9) & (df.bid_price_1.abs() < 9e9)].copy()
    d["mid"] = (d.ask_price_1 + d.bid_price_1) / 2 / 1e4
    d = d.dropna(subset=["dt", "mid"]); d = d[d.dt >= d.dt.min() + pd.Timedelta(minutes=warm)]
    s = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
    r = np.log(s).diff().dropna(); mx = d.TYPE.value_counts(normalize=True).mul(100).round(1)
    return (int(s.round(3).nunique()), float((s.max()-s.min())*100), float(r.std()*1e4),
            mx.get("LIMIT_ORDER", float("nan")), mx.get("ORDER_CANCELLED", float("nan")),
            mx.get("ORDER_EXECUTED", float("nan")))
rows = []
for f in sorted(glob.glob(os.path.join(out, "logs", ".csv_*"))):
    tag = os.path.basename(f).replace(".csv_", ""); csv = open(f).read().strip()
    if not csv or not os.path.exists(csv): continue
    try: rows.append((tag,) + stat(csv))
    except Exception as e: print(f"  {tag}: score failed ({e})")
for f in sorted(glob.glob(os.path.join(out, "real", "real_*.csv"))):
    try: rows.append(("REAL " + os.path.basename(f).split("_")[-1].replace(".csv", ""),) + stat(f))
    except Exception: pass
print("\n| cell | mids | range(tk) | 1s vol(bp) | limit% | cancel% | exec% |")
print("|---|---|---|---|---|---|---|")
for t, m, rg, v, l, c, e in rows:
    print(f"| {t} | {m} | {rg:.0f} | {v:.2f} | {l} | {c} | {e} |")
PY

say "COMPLETE — $SUM"
