#!/bin/bash
# exp4_checkpoint_sweep.sh — Experiment 4 (instability paper): is Exp 2's failure specific to one
# unlucky checkpoint, or does it show up across checkpoints from the same training run? Adapted
# from scripts/checkpoint_stability.sh's resumable checkpoint-iteration pattern, but scored
# against rl_execution/survival_metrics.py's pre-registered freeze/diverge definitions (the same
# ones Exp 2 uses) instead of that script's own cross-day-drift heuristic -- run AFTER Exp 2's
# main result is in hand, to support it. Short battery: a handful of seeds, capped horizon.
#
# GPU required -- run on the remote box. Resumable via .done sentinels.
#
# By default EXCLUDES known-decoy checkpoints -- 0.627/0.681/0.719 (pre-fix, per
# analysis/MASTER_RESULTS.md's phase table) and 0.7_epoch (documented fallback, commit b0f449c) --
# so a directory holding checkpoints from more than one training run doesn't silently corrupt
# "robustness across checkpoints from THIS training run" into "robustness across unrelated runs."
# Pass --include-decoys to sweep everything anyway.
#
# Usage:
#   bash scripts/exp4_checkpoint_sweep.sh --day 20150107 --seeds "30 31 32 33 34" --et 10:30:00
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; ET="10:30:00"; DAY="20150107"
SEEDS="30 31 32 33 34"; CKPT_DIR="data/checkpoints/TRADES"; INCLUDE_DECOYS=0
OUT_DIR="exp4_results/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --day) DAY="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --ckpt-dir) CKPT_DIR="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --include-decoys) INCLUDE_DECOYS=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
echo "# Experiment 4 — checkpoint sweep — $(date '+%F %T')  day=$DAY seeds=[$SEEDS]" > "$SUM"

if pgrep -f "main.py" > /dev/null; then echo "!! training (main.py) running — kill it first (single GPU)."; exit 1; fi

mapfile -t ALL_CKPTS < <(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null)
[[ ${#ALL_CKPTS[@]} -gt 0 ]] || { echo "!! no .ckpt in $CKPT_DIR"; exit 1; }
if [[ "$INCLUDE_DECOYS" == "1" ]]; then
  CKPTS=("${ALL_CKPTS[@]}")
else
  CKPTS=()
  for CK in "${ALL_CKPTS[@]}"; do
    case "$(basename "$CK")" in
      val_ema=0.627*|val_ema=0.681*|val_ema=0.719*|val_ema=0.7_epoch*)
        echo "  excluding known decoy: $CK  (pre-fix or documented fallback -- pass --include-decoys to sweep it anyway)" ;;
      *) CKPTS+=("$CK") ;;
    esac
  done
fi
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "!! no checkpoints left after excluding known decoys -- pass --include-decoys, or place the real training-run checkpoints."; exit 1; }
echo "checkpoints found: ${#ALL_CKPTS[@]}  (sweeping: ${#CKPTS[@]})"

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
real_dir_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$ET")_30"; }
ensure_real () {
  local RD; RD=$(real_dir_for "$DAY")
  [[ -f "$RD/processed_orders.csv" ]] || { echo "  -- real replay $DAY -> $ET" >&2; \
    python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DAY" -st "$ST" -et "$ET" -seed 30 \
      > "$OUT_DIR/logs/real_${DAY}.txt" 2>&1; }
  echo "$RD/processed_orders.csv"
}
REALP=$(ensure_real)

run () {  # run <ckptfile> <seed>
  local CK="$1" S="$2"
  local cname; cname=$(basename "$CK" .ckpt | sed -E 's/^val_ema=//; s/_INTC.*//')
  local TAG="${cname}__seed${S}"
  local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "SKIP $TAG"; return; }
  echo "-- $TAG"
  local S0; S0=$(mktemp); touch "$S0"; local T0; T0=$(date +%s)
  if ! python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DAY" -st "$ST" -et "$ET" \
        -d True -m TRADES -type DDPM -nsteps 100 --ckpt-path "$CK" -seed "$S" \
        > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    echo "  ERROR — see logs/${TAG}.txt"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S0"; return
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)
  [[ -n "$CSV" && -f "$CSV" ]] || CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S0" ! -path "*market_replay*" 2>/dev/null | sort | tail -1)
  rm -f "$S0"
  local SCORE_JSON="$OUT_DIR/logs/${TAG}.score.json"
  [[ -n "$CSV" && -f "$REALP" ]] && python -m rl_execution.survival_metrics --gen "$CSV" --real "$REALP" \
      --out "$SCORE_JSON" >> "$OUT_DIR/logs/${TAG}.txt" 2>&1
  { echo "## $TAG  (${SECS}s)"; echo '```'; echo "ckpt: $CK"; echo "csv: ${CSV:-none}"
    [[ -f "$SCORE_JSON" ]] && cat "$SCORE_JSON"
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

for CK in "${CKPTS[@]}"; do for S in $SEEDS; do run "$CK" "$S"; done; done

python3 - "$OUT_DIR" "$SUM" <<'PY'
import glob, json, os, sys
from collections import defaultdict
by_ckpt = defaultdict(list)
for jf in sorted(glob.glob(os.path.join(sys.argv[1], "logs", "*.score.json"))):
    tag = os.path.basename(jf).replace(".score.json", "")
    ckpt = tag.rsplit("__seed", 1)[0]
    with open(jf) as f:
        by_ckpt[ckpt].append(json.load(f))
hdr = f"{'checkpoint':<24}{'n':>4}{'survived':>10}{'survival%':>11}"
tab = [hdr, "-" * len(hdr)]
for ckpt, runs in by_ckpt.items():
    n = len(runs); surv = sum(1 for r in runs if r.get("survived_to_end"))
    tab.append(f"{ckpt:<24}{n:>4}{surv:>10}{100*surv/max(n,1):>10.1f}%")
table = "\n".join(tab)
print("\n==== EXPERIMENT 4 — PER-CHECKPOINT SURVIVAL ====\n" + table)
n_ckpts = len(by_ckpt)
n_fail_all = sum(1 for runs in by_ckpt.values() if all(not r.get("survived_to_end") for r in runs))
print(f"\n{n_fail_all}/{n_ckpts} checkpoints failed on EVERY seed tested "
      f"({'most/all checkpoints fail -- not one unlucky snapshot' if n_fail_all >= max(1, n_ckpts - 1) else 'mixed -- report honestly, this changes the story'}).")
with open(sys.argv[2], "a") as f:
    f.write("\n# PER-CHECKPOINT TABLE\n```\n" + table + "\n```\n")
PY

echo ""; echo "Done. Summary: $SUM"
