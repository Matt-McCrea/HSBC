#!/bin/bash
# open_loop_sweep.sh — CHEAP checkpoint × sampler sensitivity map (minutes/cell, no ABIDES loop).
#
# open_loop_eval.py samples the model on REAL test-window conditioning and reports the collapse
# mechanism directly (depth histogram, market-decode rate, all 3 type-decode variants). Because
# there's no closed-loop feedback it runs in minutes — so we can afford the FULL matrix here and
# reserve the 30-min closed-loop runs (eval_new_checkpoint.sh) for the cells that look promising.
#
# Answers the sensitivity question overnight: as checkpoint quality varies (val 0.656 → 2.869),
# does the deterministic-sampler depth-collapse ease, worsen, or stay flat? And does DDPM stay
# healthy on every checkpoint?
#
# Usage:
#   bash scripts/open_loop_sweep.sh                       # all 7 ckpts × 4 samplers, 2048 windows
#   bash scripts/open_loop_sweep.sh --ids "0.656 2.869" --n-windows 4096
#
# Reads results back into one table at the end (market% + depth-0% per cell).

set -uo pipefail
IDS="0.656 0.668 0.671 0.67 0.681 0.719 2.869"
NWIN=2048; BATCH=256; SPLIT="test"; STOCK="INTC"
OUT="open_loop_sweep/$(date +%Y%m%d_%H%M%S)"
# sampler cells: "TYPE NSTEPS ETA [extra args passed straight to open_loop_eval.py]"
SAMPLERS=(
  "DDPM 100 0.0"
  "DDIM 10 0.0"
  "DDIM 10 1.0"
  "DPM_SOLVER_PP 10 0.0"
  "CHURN 10 0.0 --churn-steps 3 --churn-strength 0.3"
  "CHURN 10 0.0 --churn-steps 4 --churn-strength 0.5"
)
while [[ $# -gt 0 ]]; do case "$1" in
  --ids) IDS="$2"; shift 2;; --n-windows) NWIN="$2"; shift 2;;
  --batch) BATCH="$2"; shift 2;; --split) SPLIT="$2"; shift 2;;
  --out-dir) OUT="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT"
echo "open-loop matrix → $OUT   (ids: $IDS  |  $NWIN windows)"

for ID in $IDS; do
  for CELL in "${SAMPLERS[@]}"; do
    read -r TYPE NS ETA EXTRA <<< "$CELL"     # EXTRA = any remaining tokens (e.g. churn args)
    SUFFIX=$(echo "$EXTRA" | tr -d ' -' | tr -s ' ' '_')   # sanitise extra args into the tag
    TAG="ckpt${ID}__${TYPE}_${NS}_eta${ETA}${SUFFIX:+_$SUFFIX}"
    JSON="$OUT/${TAG}.json"
    [[ -f "$JSON" ]] && { echo "  SKIP $TAG"; continue; }
    echo "── $TAG"
    # shellcheck disable=SC2086
    python evaluation/diagnostics/open_loop_eval.py \
      --type "$TYPE" --nsteps "$NS" --eta "$ETA" --id "$ID" $EXTRA \
      --stock "$STOCK" --split "$SPLIT" --n-windows "$NWIN" --batch "$BATCH" \
      --out "$JSON" > "$OUT/${TAG}.log" 2>&1 \
      || { echo "  ERROR — see $OUT/${TAG}.log"; continue; }
  done
done

echo ""; echo "════ SUMMARY TABLE ════"
python - "$OUT" <<'PY'
import json, sys, glob, os
root = sys.argv[1]
rows = {}   # (ckpt) -> {sampler -> (market%, depth0%)}  using L1 decode (the shipped decode)
cols = []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    d = json.load(open(f))
    c = d["config"]
    ck = os.path.basename(d["config"]["checkpoint"]).split("=")[1].split("_")[0]
    samp = f"{c['type']}{c['nsteps']}e{c['eta']}"
    if c["type"] == "CHURN":
        samp = f"CHURN{c['nsteps']}_s{c.get('churn_steps','?')}k{c.get('churn_strength','?')}"
    g = d["generated_l1_decode"]
    rows.setdefault(ck, {})[samp] = (g["type_hist"]["market"], g["depth_hist"]["0"])
    if samp not in cols: cols.append(samp)
# real reference (same across cells)
real = json.load(open(sorted(glob.glob(os.path.join(root,'*.json')))[0]))["real"]
print(f"REAL next-events:  market={real['type_hist']['market']:.1%}  depth0={real['depth_hist']['0']:.1%}\n")
w = 10
print("market-order decode %   (real={:.1%})".format(real['type_hist']['market']))
print("ckpt".ljust(8) + "".join(s.rjust(14) for s in cols))
for ck in sorted(rows, key=float):
    print(ck.ljust(8) + "".join(f"{rows[ck].get(s,(float('nan'),))[0]:>13.1%}" for s in cols))
print("\ndepth-0 share %   (real={:.1%};  higher = more collapsed)".format(real['depth_hist']['0']))
print("ckpt".ljust(8) + "".join(s.rjust(14) for s in cols))
for ck in sorted(rows, key=float):
    print(ck.ljust(8) + "".join(f"{rows[ck].get(s,(float('nan'),))[1]:>13.1%}" for s in cols))
PY
echo ""; echo "Full per-cell JSON + logs in $OUT"
