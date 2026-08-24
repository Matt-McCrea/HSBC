#!/bin/bash
# ss_quick_finish.sh — one-shot for a SHORT remaining GPU window: grab the ticker's baseline
# checkpoint, scheduled-sampling train it for a fixed number of minutes, then simulate the newest
# NEW checkpoint. Honest by construction: if training saves no new checkpoint it says so and does
# NOT pass the baseline off as a scheduled-sampling result (the failure we just fixed elsewhere).
#
# It relies on the same mechanics as new_ticker_pipeline.sh:
#   RESUME_TRAINING_FLAG      -> Lightning resumes from the NEWEST checkpoint in the dir
#   SCHEDULED_SAMPLING_FLAG   -> conditions training on the model's own generated block (needs a prior)
#   KEEP_EPOCH_CHECKPOINTS    -> one checkpoint per (short) epoch, so checkpoints land fast
#   LIMIT_TRAIN_BATCHES=0.10  -> ~10% epochs; a checkpoint every few minutes instead of ~1.7h
#
#   nohup bash scripts/ss_quick_finish.sh --train-min 35 > ss_quick.log 2>&1 & disown
set -uo pipefail

TICKER="TSLA"; START="2015-01-02"; END="2015-01-30"
BASE_CKPT=""
TYPE_PRIOR=""                        # empty => auto-derive from TRAINING days (scripts/derive_type_prior.sh)
LIMIT_TRAIN_BATCHES="0.10"           # short epochs -> a checkpoint lands in minutes (run.py:141)
TRAIN_MIN=35                         # minutes to train before handing to the sim
SIGMA="0.3"                          # sim depth-noise. NB: TSLA DIVERGES at sigma=1.0 — keep it low
SIM_DAY="20150130"; ST="10:00:00"; ET="10:30:00"   # a 30-min sanity sim, not the 2h deliverable
SIM_CAP=900
PY="${PY:-python}"
CKDIR="data/checkpoints/TRADES"
OUT_DIR="paper_runs/ss_quick_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --ticker) TICKER="$2"; shift 2;; --base-ckpt) BASE_CKPT="$2"; shift 2;;
  --type-prior) TYPE_PRIOR="$2"; shift 2;; --limit-train-batches) LIMIT_TRAIN_BATCHES="$2"; shift 2;;
  --train-min) TRAIN_MIN="$2"; shift 2;; --sigma) SIGMA="$2"; shift 2;;
  --sim-day) SIM_DAY="$2"; shift 2;; --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"
say () { echo "[$(date +%T)] $*"; }

pgrep -f "main.py" >/dev/null && { echo "!! main.py already running — kill it first, or let it finish."; exit 1; }

# 1) baseline checkpoint: newest of the ticker's (or --base-ckpt). Bump its mtime so RESUME_TRAINING
#    resumes from THIS one and not a stray (INTC etc.) that happens to be newer.
[[ -n "$BASE_CKPT" ]] || BASE_CKPT=$(ls -t "$CKDIR"/*"$TICKER"*.ckpt 2>/dev/null | head -1)
[[ -n "$BASE_CKPT" && -f "$BASE_CKPT" ]] || { echo "!! no $TICKER baseline checkpoint in $CKDIR — pass --base-ckpt"; exit 1; }
touch "$BASE_CKPT"
say "baseline (resume parent): $(basename "$BASE_CKPT")"

# 1b) type prior. The scheduled-sampling rollout decodes its own block with this prior and feeds it
#     back as conditioning, so it is a TRAINING input — it must be the ticker's own TRAINING-day
#     marginals, never INTC's default and never a test-window value (that would leak). Auto-derive
#     it (CPU, leakage-safe: excludes the val+test days) unless one was passed explicitly.
if [[ -z "$TYPE_PRIOR" ]]; then
  say "deriving $TICKER type prior from TRAINING days (full session, leakage-safe)…"
  DLOG="$OUT_DIR/logs/derive_prior.log"
  bash scripts/derive_type_prior.sh --ticker "$TICKER" --start "$START" --end "$END" > "$DLOG" 2>&1 || true
  TYPE_PRIOR=$(grep -oE 'TYPE_PRIOR = [0-9.,]+' "$DLOG" | tail -1 | sed 's/.*= //')
  [[ -n "$TYPE_PRIOR" ]] || { echo "!! prior derivation failed — see $DLOG. Pass --type-prior to override."; exit 1; }
  say "derived TYPE_PRIOR = $TYPE_PRIOR  (INTC default is 0.49,0.48,0.03)"
else
  say "using provided TYPE_PRIOR = $TYPE_PRIOR"
fi

# snapshot the checkpoints that already exist, so NEW (SS-lineage) ones are told apart by name
SNAP="$OUT_DIR/.baseline_ckpts"; ls "$CKDIR"/*.ckpt 2>/dev/null | sort > "$SNAP"
new_ckpts () { ls "$CKDIR"/*.ckpt 2>/dev/null | sort | comm -13 "$SNAP" - ; }

# 2) scheduled-sampling training, time-boxed
touch SCHEDULED_SAMPLING_FLAG RESUME_TRAINING_FLAG KEEP_EPOCH_CHECKPOINTS_FLAG UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
export TICKER TRADING_START="$START" TRADING_END="$END" TYPE_PRIOR LIMIT_TRAIN_BATCHES
TLOG="$OUT_DIR/logs/ss_train.log"
say "SS training $TICKER for ${TRAIN_MIN}min (LIMIT_TRAIN_BATCHES=$LIMIT_TRAIN_BATCHES, prior=$TYPE_PRIOR) -> $TLOG"
nohup "$PY" -u main.py > "$TLOG" 2>&1 &
TPID=$!
END_T=$(( $(date +%s) + TRAIN_MIN*60 ))
while [[ $(date +%s) -lt "$END_T" ]]; do
  kill -0 "$TPID" 2>/dev/null || { say "training process exited early — see $TLOG"; break; }
  say "  training… new checkpoints so far: $(new_ckpts | wc -l | tr -d ' '), $(( (END_T - $(date +%s))/60 ))min left"
  sleep 120
done
pkill -f main.py 2>/dev/null; sleep 10
rm -f SCHEDULED_SAMPLING_FLAG RESUME_TRAINING_FLAG   # the sim must NOT resume training
say "training stopped; $(new_ckpts | wc -l | tr -d ' ') NEW checkpoint(s) from the SS lineage"

# 3) pick the newest NEW checkpoint; fail honest if none
SS_CKPT=""; while IFS= read -r f; do [[ -z "$SS_CKPT" || "$f" -nt "$SS_CKPT" ]] && SS_CKPT="$f"; done < <(new_ckpts)
if [[ -z "$SS_CKPT" ]]; then
  say "!! NO new checkpoint saved in ${TRAIN_MIN}min. Simulating the BASELINE (clearly labelled), and"
  say "   rerun with a smaller --limit-train-batches (e.g. 0.05) or a longer --train-min next time."
  SIM_CKPT="$BASE_CKPT"; TAG="baseline_NO_ss_ckpt"
else
  say "SS checkpoint: $(basename "$SS_CKPT")"; SIM_CKPT="$SS_CKPT"; TAG="ss"
fi

# 4) short sanity sim at the corrected decode config
say "-- sim $TAG  [DDIM-10 $SIM_DAY $ST-$ET  sigma=$SIGMA]"
A=("$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$SIM_DAY" -st "$ST" -et "$ET"
   -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$SIM_CKPT" -seed 30
   --depth-noise "$SIGMA" --size-reshape --type-decode prior --type-prior "$TYPE_PRIOR")
if timeout -k 30 "$SIM_CAP" "${A[@]}" > "$OUT_DIR/logs/sim_${TAG}.txt" 2>&1; then
  CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/sim_${TAG}.txt" | tail -1)
  say "sim done -> ${CSV:-<no csv path in log>}"
else
  rc=$?; CSV=""
  say "sim $([[ $rc -eq 124 || $rc -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$rc") — see $OUT_DIR/logs/sim_${TAG}.txt"
fi

# 5) quick read
if [[ -n "${CSV:-}" && -f "$CSV" ]]; then
  "$PY" - "$CSV" "$TAG" <<'PY'
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); d["dt"] = pd.to_datetime(d.iloc[:,0], errors="coerce")
for c in ("ask_price_1","bid_price_1"): d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[(d.ask_price_1>0)&(d.bid_price_1>0)&(d.ask_price_1.abs()<9e9)&(d.bid_price_1.abs()<9e9)].dropna(subset=["dt"])
d["mid"] = (d.ask_price_1+d.bid_price_1)/2/1e4
d = d[d.dt >= d.dt.min()+pd.Timedelta(minutes=15)]
s = d.set_index("dt")["mid"].resample("1s").last().ffill().dropna(); r = np.log(s).diff().dropna()
m = d.TYPE.value_counts(normalize=True).mul(100).round(1)
print("\nSIM RESULT [{}]: mids={} range={:.0f}tk 1s_vol={:.2f}bp limit/cancel/exec={}/{}/{}".format(
  sys.argv[2], s.round(3).nunique(), (s.max()-s.min())*100, r.std()*1e4,
  m.get("LIMIT_ORDER","-"), m.get("ORDER_CANCELLED","-"), m.get("ORDER_EXECUTED","-")))
print("(real TSLA ~1.3-1.5bp, exec ~2%; high vol/range/exec = the sigma-driven divergence, lower --sigma)")
PY
fi
say "COMPLETE — $OUT_DIR/   train log: $TLOG   sim log: logs/sim_${TAG}.txt"
