#!/usr/bin/env bash
#
# battery_0724.sh — unattended battery on the 0.724 checkpoint.
#
#   tmux new -s bat
#   ./battery_0724.sh
#   # ctrl-b d to detach, go to lunch, `tmux attach -t bat` to come back
#
# or:  nohup ./battery_0724.sh > battery_console.log 2>&1 &
#      tail -f battery_console.log
#
# Everything here is on ONE checkpoint, so the flag files stay ON throughout
# and there is no toggling. Runs are sequential (GPU contention) and a failure
# in one does NOT stop the others.
#
# Deliberately NO `set -e`. Every previous version of this script died silently
# on a false test in a return position. Errors are handled explicitly instead.
#
# Runs are ordered by value, so if you come back early the important ones are
# already done. Total is roughly 2 hours.
#
#   1 typedecode_l1   corrections minus --type-decode   ~16m  BLOCKING for 5.4.2
#   2 eta1            DDIM-10 eta=1, no corrections     ~16m  substitutes the lost R2
#   3 vanilla         no corrections at all             ~16m  neg-depth baseline
#   4 sigma030        adopted config                    ~16m  DIAG for the shipped setup
#   5 sigma015        low end of the dial               ~16m  dose-response on a RETAINED ckpt
#   6 sigma050        high end of the dial              ~16m  ditto
#   7 ddim100         DDIM-100 eta=0                    ~22m  divergence regime
#
# NOTE: the DIAG lines appear to be printed at END of simulation, so a killed
# run yields nothing. Let each one finish.

CK="data/checkpoints/TRADES/val_ema=0.724_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt"
TICKER=INTC; DATE=20150130; ST=09:30:00; ET=10:00:00; SEED=30

OUT="logs_0724_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
PROGRESS="$OUT/PROGRESS.txt"

note() { echo "$(date '+%H:%M:%S')  $*" | tee -a "$PROGRESS"; }

# ---------------------------------------------------------------- preflight --
note "battery starting, output -> $OUT"

if [ ! -f ABIDES/abides.py ]; then
  note "FATAL: not in the repo root (no ABIDES/abides.py). Nothing run."
  exit 1
fi

# 0.724 was trained WITH the pipeline fixes, so both flags must be present.
for f in UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG; do
  if [ -e ".flagstash/$f" ] && [ ! -e "$f" ]; then mv ".flagstash/$f" .; fi
  if [ ! -e "$f" ]; then touch "$f"; fi
  note "flag $f present"
done

if [ ! -f "$CK" ]; then
  note "WARNING: no local copy of $CK -- continuing anyway, it may be remote"
fi

# ---------------------------------------------------------------- run helper -
# run <name> <extra abides args...>
run() {
  name="$1"; shift
  log="$OUT/${name}.log"
  note "START $name"
  t0=$(date +%s)

  python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" \
    -st "$ST" -et "$ET" -d True -m TRADES -seed "$SEED" \
    --ckpt-path "$CK" "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}

  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    note "DONE  $name  ($((t1-t0))s)"
  else
    note "FAIL  $name  (exit $rc, $((t1-t0))s) -- see $log"
  fi
  # keep going regardless
  return 0
}

# -------------------------------------------------------------------- runs ---

# 1. The blocking one: l1 type decode = omit --type-decode, keep the other two.
run typedecode_l1  -type DDIM -nsteps 10 -eta 0.0 --depth-noise 0.3 --size-reshape

# 2. eta=1, no decode corrections. Stands in for the eta=1 run on the lost
#    checkpoint: does restoring injected noise bring movement back through the
#    type channel rather than the depth channel?
run eta1           -type DDIM -nsteps 10 -eta 1.0

# 3. Fully vanilla. Gives the neg-depth share with no corrections, which is the
#    baseline every other row is measured against.
run vanilla        -type DDIM -nsteps 10 -eta 0.0

# 4-6. Dose-response on a RETAINED checkpoint. The version in 5.4.2 is on a
#      checkpoint that no longer exists, so this would let the dial argument
#      stand on a model anyone can re-run.
run sigma030       -type DDIM -nsteps 10 -eta 0.0 --depth-noise 0.30 --size-reshape --type-decode prior
run sigma015       -type DDIM -nsteps 10 -eta 0.0 --depth-noise 0.15 --size-reshape --type-decode prior
run sigma050       -type DDIM -nsteps 10 -eta 0.0 --depth-noise 0.50 --size-reshape --type-decode prior

# 7. Many-step deterministic: the divergence end of the operating range.
run ddim100        -type DDIM -nsteps 100 -eta 0.0 --depth-noise 0.3 --size-reshape --type-decode prior

# ----------------------------------------------------------------- summary ---
note "battery finished"
echo
echo "================ SUMMARY ================"
cat "$PROGRESS"
echo
echo "================ DIAG LINES ================"
for log in "$OUT"/*.log; do
  case "$log" in *PROGRESS*) continue;; esac
  echo
  echo "--- $(basename "$log")"
  grep -E 'decoded_pre_drop|placed|drops|execution_channels|depth_pre_drop|size_pre_drop|resample|cond_z|Time taken' \
       "$log" 2>/dev/null || echo "(no DIAG lines)"
done
echo
echo "Unique mids and flow mix are not in stdout -- score the CSVs separately."
