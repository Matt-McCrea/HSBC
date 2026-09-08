#!/bin/bash
# exp2_survival_sweep.sh — Experiment 2 (instability paper), the paper's headline evidence.
# Runs DDPM-100 on the pinned "model under test" checkpoint (analysis/model_under_test.md --
# THIS SCRIPT REFUSES TO RUN UNTIL THAT FILE'S "Confirmed checkpoint" SECTION IS FILLED IN),
# NO decode-time flags, for every (day, seed, price-reanchor variant), for as long as practically
# possible, and scores each run against the pre-registered freeze/diverge definitions in
# rl_execution/survival_metrics.py. Single-GPU, strictly serial (see
# rl_execution/RUNBOOK_instability.md for the tmux layout this fits into).
#
# Resumable via .done sentinels. GPU REQUIRED (loads and samples the model) -- do not run this
# locally, per this project's own convention (see rl_execution/train.py, evaluate.py, benchmark.py).
#
# Usage (pilot, per the plan's own suggestion -- run this FIRST):
#   bash scripts/exp2_survival_sweep.sh --days "20150130 20150107 20150115" --seeds "30 31 32 33 34" \
#       --et 10:30:00 --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt
# Usage (full sweep, only after the pilot's failure pattern looks like the dissertation's):
#   bash scripts/exp2_survival_sweep.sh --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="13:30:00"   # ET default: 4h horizon, as long as practically possible
DAYS="20150102 20150105 20150106 20150107 20150108 20150109 20150112 20150113 20150114 20150115 \
20150116 20150120 20150121 20150122 20150123 20150126 20150127 20150128 20150129 20150130"
SEEDS="30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"  # 20 seeds default
REANCHOR="both"   # on | off | both -- the price-reanchoring open question (see model_under_test.md)
CKPT_PATH=""
OUT_DIR="exp2_results/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --days) DAYS="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --reanchor) REANCHOR="$2"; shift 2;; --ckpt-path) CKPT_PATH="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

[[ -n "$CKPT_PATH" ]] || { echo "!! --ckpt-path required. See analysis/model_under_test.md for"; \
  echo "   which checkpoint (its 'Confirmed checkpoint' section must be filled in FIRST)."; exit 1; }
[[ -f "$CKPT_PATH" ]] || { echo "!! checkpoint not found: $CKPT_PATH"; exit 1; }
grep -q "^\*(to be filled in" analysis/model_under_test.md 2>/dev/null && \
  { echo "!! analysis/model_under_test.md's 'Confirmed checkpoint' section is still blank."; \
    echo "   Fill it in (file path + flag state actually used) before running Exp 2 for real."; exit 1; }

if pgrep -f "main.py" > /dev/null; then echo "!! training (main.py) running — kill it first (single GPU)."; exit 1; fi
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
echo "# Experiment 2 — survival sweep — $(date '+%F %T')" > "$SUM"
echo "ckpt: $CKPT_PATH" >> "$SUM"
N_DAYS=$(echo $DAYS | wc -w | tr -d ' '); N_SEEDS=$(echo $SEEDS | wc -w | tr -d ' ')
echo "days: $N_DAYS  seeds: $N_SEEDS  reanchor: $REANCHOR  et: $ST-$ET  out: $OUT_DIR"

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
real_dir_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$ET")_30"; }
ensure_real () {
  local D="$1" RD; RD=$(real_dir_for "$D")
  [[ -f "$RD/processed_orders.csv" ]] || { echo "  -- real replay $D -> $ET" >&2; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" -seed 30 \
      > "$OUT_DIR/logs/real_${D}.txt" 2>&1; }
  echo "$RD/processed_orders.csv"
}

run_variant () {  # run_variant <day> <seed> <reanchor_on:0|1>
  local D="$1" S="$2" RA="$3"
  local RA_TAG; RA_TAG=$([[ "$RA" == "1" ]] && echo "reanchor" || echo "noreanchor")
  local TAG="${D}__seed${S}__${RA_TAG}"
  local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "SKIP $TAG"; return; }
  local REALP; REALP=$(ensure_real "$D")

  # PRICE_REANCHOR is decided at import time (constants.py), gated by a file flag -- flip it via
  # the flag file for the duration of this one run only, then restore whatever was there before.
  local HAD_FLAG=0; [[ -f PRICE_REANCHOR_FLAG ]] && HAD_FLAG=1
  if [[ "$RA" == "1" ]]; then touch PRICE_REANCHOR_FLAG; else rm -f PRICE_REANCHOR_FLAG; fi

  echo "-- $TAG"
  local S0; S0=$(mktemp); touch "$S0"; local T0; T0=$(date +%s)
  python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
      -d True -m TRADES -type DDPM -nsteps 100 --ckpt-path "$CKPT_PATH" -seed "$S" \
      > "$OUT_DIR/logs/${TAG}.txt" 2>&1
  local RC=$?
  [[ "$HAD_FLAG" == "1" ]] && touch PRICE_REANCHOR_FLAG || rm -f PRICE_REANCHOR_FLAG
  local SECS=$(( $(date +%s) - T0 ))
  if [[ "$RC" -ne 0 ]]; then
    echo "  ERROR — see logs/${TAG}.txt"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S0"; return
  fi
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)
  [[ -n "$CSV" && -f "$CSV" ]] || CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S0" ! -path "*market_replay*" 2>/dev/null | sort | tail -1)
  rm -f "$S0"
  local SCORE_JSON="$OUT_DIR/logs/${TAG}.score.json"
  if [[ -n "$CSV" && -f "$REALP" ]]; then
    python -m rl_execution.survival_metrics --gen "$CSV" --real "$REALP" --out "$SCORE_JSON" \
      >> "$OUT_DIR/logs/${TAG}.txt" 2>&1
  fi
  { echo "## $TAG  (${SECS}s)"; echo '```'; echo "csv: ${CSV:-none}"
    [[ -f "$SCORE_JSON" ]] && cat "$SCORE_JSON"
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

for D in $DAYS; do
  for S in $SEEDS; do
    [[ "$REANCHOR" == "on" || "$REANCHOR" == "both" ]] && run_variant "$D" "$S" 1
    [[ "$REANCHOR" == "off" || "$REANCHOR" == "both" ]] && run_variant "$D" "$S" 0
  done
done

# Master survival-fraction rollup: pooled + per-day, over elapsed time since arrival.
python3 - "$OUT_DIR" "$SUM" <<'PY'
import glob, json, os, sys
rows = []
for jf in sorted(glob.glob(os.path.join(sys.argv[1], "logs", "*.score.json"))):
    tag = os.path.basename(jf).replace(".score.json", "")
    day, seed_part, ra = tag.split("__")
    with open(jf) as f:
        d = json.load(f)
    rows.append({"day": day, "seed": seed_part.replace("seed", ""), "reanchor": ra, **d})

n = len(rows)
survived = sum(1 for r in rows if r.get("survived_to_end"))
print(f"\n==== EXPERIMENT 2 — {n} scored runs ====")
print(f"survived to end of horizon: {survived}/{n} ({100*survived/max(n,1):.1f}%)")
with open(sys.argv[2], "a") as f:
    f.write(f"\n# ROLLUP\n{n} scored runs, {survived} survived to end of horizon "
            f"({100*survived/max(n,1):.1f}%).\n")
PY

echo ""; echo "Done. Summary: $SUM"
echo "READ: build the survival-fraction-over-time curve (pooled + per-day) from *.score.json --"
echo "the plan's headline chart. Each run's full CSV is under its own ABIDES/log/world_agent_... dir."
