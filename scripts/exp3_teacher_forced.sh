#!/bin/bash
# exp3_teacher_forced.sh — Experiment 3 (instability paper): is the failure Exp 2 measures the
# model's own mistakes compounding (exposure bias), or does per-step quality degrade even when the
# model is always shown REAL history? Wraps evaluation/diagnostics/open_loop_eval.py (already does
# exactly the teacher-forced sampling this needs -- no ABIDES loop, real conditioning throughout),
# patched with --bucket-by-time so early- vs late-session accuracy is directly comparable.
#
# Note: open_loop_eval.py samples windows from the whole test-split dataset (LOBDataset), not a
# specific (day, seed) pair like the ABIDES-driven experiments -- there is no day/seed grid to
# match here, only the checkpoint + PRICE_REANCHOR variant (which changes preprocessing/
# conditioning, so needs its own run, same as Exp 2). GPU required -- run on the remote box.
#
# The checkpoint does NOT need to be typed -- it's read automatically from the CKPT_PATH= line in
# analysis/model_under_test.md, written by `bash scripts/exp1_pin_checkpoint.sh`. Pass --ckpt-path
# only to override that (never a bare val_ema number -- two checkpoints can share a rounded
# val_ema, see analysis/MASTER_RESULTS.md 1.4).
#
# ONE COMMAND, ZERO ARGUMENTS:
#   bash scripts/exp3_teacher_forced.sh
set -uo pipefail
CKPT_PATH=""; REANCHOR="both"; N_WINDOWS=4096; OUT_DIR="exp3_results/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --ckpt-path) CKPT_PATH="$2"; shift 2;; --reanchor) REANCHOR="$2"; shift 2;;
  --n-windows) N_WINDOWS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

source "$(dirname "$0")/_ckpt_lib.sh"
CKPT_PATH=$(resolve_ckpt_path "$CKPT_PATH") || exit 1
[[ -f "$CKPT_PATH" ]] || { echo "!! checkpoint not found: $CKPT_PATH"; exit 1; }
mkdir -p "$OUT_DIR"

run_variant () {  # run_variant <reanchor_on:0|1>
  local RA="$1"
  local RA_TAG; RA_TAG=$([[ "$RA" == "1" ]] && echo "reanchor" || echo "noreanchor")
  local HAD_FLAG=0; [[ -f PRICE_REANCHOR_FLAG ]] && HAD_FLAG=1
  if [[ "$RA" == "1" ]]; then touch PRICE_REANCHOR_FLAG; else rm -f PRICE_REANCHOR_FLAG; fi
  echo "-- open-loop DDPM-100, ckpt=$CKPT_PATH, $RA_TAG"
  python -u evaluation/diagnostics/open_loop_eval.py --type DDPM --nsteps 100 --ckpt-path "$CKPT_PATH" \
      --split test --n-windows "$N_WINDOWS" --bucket-by-time \
      --out "$OUT_DIR/open_loop_${RA_TAG}.json" \
      2>&1 | tee "$OUT_DIR/open_loop_${RA_TAG}.txt"
  [[ "$HAD_FLAG" == "1" ]] && touch PRICE_REANCHOR_FLAG || rm -f PRICE_REANCHOR_FLAG
}

[[ "$REANCHOR" == "on" || "$REANCHOR" == "both" ]] && run_variant 1
[[ "$REANCHOR" == "off" || "$REANCHOR" == "both" ]] && run_variant 0

echo ""; echo "Done. Next: cat $OUT_DIR/open_loop_noreanchor.txt   (read bucket_early vs bucket_late)"
