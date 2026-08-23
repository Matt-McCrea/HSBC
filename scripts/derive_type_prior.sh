#!/bin/bash
# derive_type_prior.sh — compute a stock's real next-event marginals for --type-prior / TYPE_PRIOR.
#
# WHY THIS MATTERS. The prior-corrected type decode uses [limit, cancel, market] class priors, and
# they default to INTC's 0.49/0.48/0.03. That default is applied in TWO places:
#   * simulation  — ABIDES/agent/WorldAgent.py, via --type-prior
#   * TRAINING    — the scheduled-sampling rollout decodes its own generated block with it before
#                   feeding it back as conditioning (diffusion_engine.py:_decode_type)
# so on any stock whose marginals differ from Intel's it must be set BEFORE an SS retrain, not just
# before simulation. On TSLA the INTC prior pinned generated market orders to 3% and held execution
# share at 2.7-4.2% across sigma 0.15-3.0 while real was 16.7%.
#
# TRAINING DAYS ONLY, by default. The prior is a model configuration parameter, so deriving it from
# a held-out test day is leakage — and it matters here: real TSLA execution share is 17.7% on
# 2015-01-02 and 7.1% on 2015-01-29. With SPLIT_RATES .85/.05/.10 applied chronologically by day,
# the last two days of the period are the test set and the third-from-last is validation; this
# script excludes all three.
#
# CPU only — reads the raw LOBSTER files, runs no simulation.
#
#   bash scripts/derive_type_prior.sh --ticker TSLA
#   bash scripts/derive_type_prior.sh --ticker TSLA --st 09:30:00 --et 16:00:00
set -uo pipefail

TICKER="TSLA"; START="2015-01-02"; END="2015-01-30"
ST="09:30:00"; ET="16:00:00"          # full session: the most data for a stable estimate
KEEP=0                                 # keep the per-day reference CSVs instead of deleting them
PY="${PY:-python}"
WORK="$(mktemp -d)"

while [[ $# -gt 0 ]]; do case "$1" in
  --ticker) TICKER="$2"; shift 2;; --start) START="$2"; shift 2;; --end) END="$2"; shift 2;;
  --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --keep) KEEP=1; WORK="type_prior_${TICKER}"; mkdir -p "$WORK"; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

export TICKER TRADING_START="$START" TRADING_END="$END"
trap '[[ "$KEEP" == "1" ]] || rm -rf "$WORK"' EXIT

DATA_DIR="data/${TICKER}/${TICKER}_${START}_${END}"
[[ -d "$DATA_DIR" ]] || DATA_DIR=$(ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1)
[[ -n "$DATA_DIR" && -d "$DATA_DIR" ]] || { echo "!! no data for $TICKER $START..$END"; exit 1; }

ALL=$(ls "$DATA_DIR" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u)
N=$(wc -l <<< "$ALL" | tr -d ' ')
[[ "$N" -ge 4 ]] || { echo "!! only $N days found; need at least 4 to hold out val+test"; exit 1; }

# drop the last three: 2 test + 1 validation
TRAIN_DAYS=$(head -n $((N - 3)) <<< "$ALL")
HELD=$(tail -n 3 <<< "$ALL" | tr '\n' ' ')

echo "=== type prior for $TICKER  ($ST-$ET) ==="
echo "days total : $N"
echo "training   : $(wc -l <<< "$TRAIN_DAYS" | tr -d ' ') days"
echo "held out   : $HELD (excluded — using these would leak)"
echo ""

OK=0
for D in $TRAIN_DAYS; do
  DD="${D//-/}"
  if "$PY" -m evaluation.stylized_custom.lobster_real_reference \
        --ticker "$TICKER" --date "$DD" --st "$ST" --et "$ET" \
        --out "$WORK/tp_${DD}.csv" > "$WORK/tp_${DD}.log" 2>&1; then
    OK=$((OK+1)); printf '  %s ok\n' "$D"
  else
    printf '  %s FAILED (%s)\n' "$D" "$(tail -1 "$WORK/tp_${DD}.log")"
  fi
done
[[ "$OK" -gt 0 ]] || { echo "!! no days could be read"; exit 1; }
echo ""

"$PY" - "$WORK" "$TICKER" <<'PY'
import glob, os, sys
import pandas as pd

work, ticker = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(work, "tp_*.csv")))

per_day, frames = [], []
for f in files:
    d = pd.read_csv(f, usecols=["TYPE"])
    frames.append(d)
    m = d.TYPE.value_counts(normalize=True)
    l, c, e = (float(m.get(k, 0.0)) for k in ("LIMIT_ORDER", "ORDER_CANCELLED", "ORDER_EXECUTED"))
    t = l + c + e
    per_day.append((os.path.basename(f)[3:11], l / t, c / t, e / t, len(d)))

print(f"{'day':>10} {'limit':>8} {'cancel':>8} {'market':>8} {'rows':>10}")
print("-" * 48)
for day, l, c, e, n in per_day:
    print(f"{day:>10} {l:8.4f} {c:8.4f} {e:8.4f} {n:10,}")

# Pooled over all events, so busier days weight more — the prior should reflect the event
# distribution the model actually sees, not an unweighted mean of days.
allev = pd.concat(frames)
m = allev.TYPE.value_counts(normalize=True)
l, c, e = (float(m.get(k, 0.0)) for k in ("LIMIT_ORDER", "ORDER_CANCELLED", "ORDER_EXECUTED"))
t = l + c + e
l, c, e = l / t, c / t, e / t

mk = [p[3] for p in per_day]
print("-" * 48)
print(f"{'POOLED':>10} {l:8.4f} {c:8.4f} {e:8.4f} {len(allev):10,}")
print(f"\nmarket share across days: min {min(mk):.4f}  max {max(mk):.4f}  spread {max(mk)-min(mk):.4f}")
if max(mk) - min(mk) > 0.05:
    print("  ^ wide day-to-day spread: the pooled value is the right single choice, but note the")
    print("    variation when comparing any single day's generated output against its own real row.")

prior = f"{l:.4f},{c:.4f},{e:.4f}"
print(f"\n  TYPE_PRIOR = {prior}")
print(f"  INTC default = 0.4900,0.4800,0.0300"
      f"{'   (materially different — do NOT use the default)' if abs(e - 0.03) > 0.02 else ''}")
print(f"""
Use it in both places:

  training (scheduled sampling):
    bash scripts/new_ticker_pipeline.sh --ticker {ticker} --phase ss-retrain --type-prior {prior}

  simulation:
    --type-decode prior --type-prior {prior}
""")
PY
