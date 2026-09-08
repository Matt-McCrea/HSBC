#!/bin/bash
# exp3_teacher_forced.sh — Experiment 3 (instability paper): is the failure Exp 2 measures the
# model's own mistakes compounding (exposure bias), or does per-step quality degrade even when the
# model is always shown REAL history? Wraps evaluation/diagnostics/open_loop_eval.py (already does
# exactly the teacher-forced sampling this needs -- no ABIDES loop, real conditioning throughout),
# patched with --bucket-by-time so early- vs late-session accuracy is directly comparable.
#
# Note: open_loop_eval.py samples windows from the whole test-split dataset (LOBDataset), not a
# specific (day, seed) pair like the ABIDES-driven experiments -- there is no day/seed grid to
# match here, only the checkpoint. GPU required -- run on the remote box.
#
# The checkpoint does NOT need to be typed. --variant <baseline|reanchor|ss> is the normal path --
# reads the right CKPT_PATH_<VARIANT> from analysis/model_under_test.md and forces price-
# reanchoring to match how that checkpoint was TRAINED (not an independent choice -- see
# exp2_survival_sweep.sh's header for why). --ckpt-path overrides.
#
# ONE COMMAND PER VARIANT:
#   bash scripts/exp3_teacher_forced.sh --variant baseline
#   bash scripts/exp3_teacher_forced.sh --variant reanchor
#   bash scripts/exp3_teacher_forced.sh --variant ss
set -uo pipefail
source "$(dirname "$0")/_ckpt_lib.sh"
CKPT_PATH=""; VARIANT=""; REANCHOR="off"; N_WINDOWS=4096; OUT_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in
  --ckpt-path) CKPT_PATH="$2"; shift 2;; --reanchor) REANCHOR="$2"; shift 2;;
  --variant) VARIANT="$2"; REANCHOR=$(reanchor_for_variant "$2"); shift 2;;
  --n-windows) N_WINDOWS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

if [[ -n "$VARIANT" ]]; then
  [[ -n "$CKPT_PATH" ]] || CKPT_PATH=$(resolve_ckpt_path_for_variant "$VARIANT") || exit 1
else
  CKPT_PATH=$(resolve_ckpt_path "$CKPT_PATH") || exit 1
fi
[[ -f "$CKPT_PATH" ]] || { echo "!! checkpoint not found: $CKPT_PATH"; exit 1; }
[[ -n "$OUT_DIR" ]] || OUT_DIR="exp3_results/${VARIANT:-adhoc}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

RA_TAG=$([[ "$REANCHOR" == "on" ]] && echo "reanchor" || echo "noreanchor")
HAD_FLAG=0; [[ -f PRICE_REANCHOR_FLAG ]] && HAD_FLAG=1
[[ "$REANCHOR" == "on" ]] && touch PRICE_REANCHOR_FLAG || rm -f PRICE_REANCHOR_FLAG
echo "-- open-loop DDPM-100, ckpt=$CKPT_PATH, $RA_TAG"
python -u evaluation/diagnostics/open_loop_eval.py --type DDPM --nsteps 100 --ckpt-path "$CKPT_PATH" \
    --split test --n-windows "$N_WINDOWS" --bucket-by-time \
    --out "$OUT_DIR/open_loop_${RA_TAG}.json" \
    2>&1 | tee "$OUT_DIR/open_loop_${RA_TAG}.txt"
[[ "$HAD_FLAG" == "1" ]] && touch PRICE_REANCHOR_FLAG || rm -f PRICE_REANCHOR_FLAG

echo ""; echo "Done. Next: cat $OUT_DIR/open_loop_${RA_TAG}.txt   (read bucket_early vs bucket_late)"
