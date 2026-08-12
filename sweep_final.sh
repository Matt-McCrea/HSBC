#!/usr/bin/env bash
#
# sweep_final.sh — everything the paper still needs from the GPU, in one script.
#
#   tmux new -s sweep
#   ./sweep_final.sh all
#   # ctrl-b d, come back with: tmux attach -t sweep
#
# Phases (each can be run alone; `all` runs them in this order):
#
#   fills          ~4 h    short runs. Closes coverage so BOTH test days carry all
#                          four configurations, plus the 0.724 battery.
#   month-corr    ~8-25 h  full-month DDPM-100 WITH corrections. Fixes the gap
#                          section 5.4.3 admits to: the step-count claim currently
#                          rests on the single day both were run on.
#   month-vanilla ~8-25 h  full-month DDPM-100 vanilla. The inherited baseline
#                          across the month rather than one day.
#   predictive     ~1 h    predictive score (MAE) for all four configurations
#                          against one real day, with a shared replay baseline.
#
# WHAT IS NOT HERE: LOB-Bench scoring. It needs jax and python >= 3.10 and runs
# locally only -- pull the CSVs and score them on your machine afterwards.
#
# FLAGS: every checkpoint used here was trained WITH the pipeline fixes, so
# UNCLAMP_DEPTH_FLAG and PRICE_REANCHOR_FLAG stay ON throughout and there is no
# toggling. "vanilla" below means no DECODE-time corrections, which is a
# command-line matter, not a flag matter. This is why nothing in this script
# touches .flagstash.
#
# DESIGN NOTES, all of them learned the hard way:
#   - no `set -e`. A failure in one run must never stop an unattended sweep.
#   - `python -u` and a plain pipe. Without it python block-buffers and a live
#     run looks dead for minutes.
#   - completion is marked by a .done file written only on exit 0. A crashed run
#     leaves no marker and re-runs. Never skip merely because a log exists.
#   - every run is wrapped in `timeout` so one pathological day cannot eat the
#     session. 40 min matches the cap used in the checkpoint search.
#
set -u

# ------------------------------------------------------------------ config --
CKDIR="data/checkpoints/TRADES"
CK_BASE="$CKDIR/val_ema=0.724_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt"
CK_FINAL="$CKDIR/val_ema=0.69_epoch=4_INTC_se_256_au_64_CD_8_seed_30.ckpt"

TICKER=INTC
SEED=30
# Per-run wall-clock cap. NOT 40 min: that was the checkpoint-search cap on the
# RTX 4070, where DDIM-10 costs ~11.7 ms/order. On the UCL box the same run
# measures ~54.8 ms/order and a 30-minute session takes ~55 min, so a 40-minute
# cap would kill every run. Override with RUN_TIMEOUT=... if the box changes.
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"          # 2 h, DDIM-10 sessions
RUN_TIMEOUT_DDPM="${RUN_TIMEOUT_DDPM:-10800}"   # 3 h, DDPM-100 is ~10x the NN work
CORR="--depth-noise 0.3 --size-reshape --type-decode prior"

# The 20 January 2015 trading days.
DAYS=(20150102 20150105 20150106 20150107 20150108 20150109 20150112 20150113 \
      20150114 20150115 20150116 20150120 20150121 20150122 20150123 20150126 \
      20150127 20150128 20150129 20150130)

OUT="sweep_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
PROG="$OUT/PROGRESS.txt"

note() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$PROG"; }

# ------------------------------------------------------------------ runner --
# sim <name> <date> <start> <end> <args...>
sim() {
  local name="$1" date="$2" st="$3" et="$4"; shift 4
  local log="$OUT/${name}.log" done="$OUT/${name}.done"

  if [ -e "$done" ]; then note "skip  $name (already completed)"; return 0; fi
  [ -e "$log" ] && mv "$log" "${log%.log}.$(date +%H%M%S).old.log"

  # DDPM-100 runs get the longer cap.
  local cap="$RUN_TIMEOUT"
  case " $* " in *" -type DDPM "*|*" -nsteps 100 "*) cap="$RUN_TIMEOUT_DDPM";; esac

  note "START $name (cap ${cap}s)"
  local t0; t0=$(date +%s)

  timeout "$cap" python -u ABIDES/abides.py -c world_agent_sim \
    -t "$TICKER" -date "$date" -st "$st" -et "$et" -d True -m TRADES \
    -seed "$SEED" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  local dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then
    touch "$done"; note "DONE  $name (${dt}s)"
  elif [ "$rc" -eq 124 ]; then
    note "TIMEOUT $name (${dt}s, cap ${cap}s) -- continuing"
  else
    note "FAIL  $name (exit $rc, ${dt}s) -- see $log"
  fi
  return 0
}

# ------------------------------------------------------- phase: fills -------
# Coverage so both released test days carry all four configurations at 30 min,
# plus the 0.724 battery. Ordered so the blocking item runs first.
phase_fills() {
  note "=== PHASE fills ==="

  # BLOCKING for 5.4.2: l1 type decode is `omit --type-decode`, keep the rest.
  sim typedecode_l1 20150130 09:30:00 10:00:00 \
      -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK_BASE" \
      --depth-noise 0.3 --size-reshape

  # Coverage fills: DDIM-1 and vanilla DDPM-100 on BOTH days, so the four-way
  # comparison is not restricted to 2015-01-29.
  for d in 20150129 20150130; do
    sim "ddim1_vanilla_$d"   "$d" 09:30:00 10:00:00 \
        -type DDIM -nsteps 1   -eta 0.0 --ckpt-path "$CK_BASE"
    sim "ddpm100_vanilla_$d" "$d" 09:30:00 10:00:00 \
        -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK_BASE"
  done

  # Replication second day: their released window is 09:45-11:00.
  sim repl_ddpm100_vanilla_0129 20150129 09:45:00 11:00:00 \
      -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK_BASE"

  # 0.724 battery. Harmless if yesterday's already landed -- .done markers skip.
  sim eta1_vanilla   20150130 09:30:00 10:00:00 -type DDIM -nsteps 10  -eta 1.0 --ckpt-path "$CK_BASE"
  sim vanilla        20150130 09:30:00 10:00:00 -type DDIM -nsteps 10  -eta 0.0 --ckpt-path "$CK_BASE"
  sim ddim100_corr   20150130 09:30:00 10:00:00 -type DDIM -nsteps 100 -eta 0.0 --ckpt-path "$CK_BASE" $CORR

  # Dose-response on a RETAINED checkpoint. The version in 5.4.2 is on a
  # checkpoint that no longer exists in usable form.
  for s in 0.15 0.30 0.50; do
    sim "sigma${s}" 20150130 09:30:00 10:00:00 \
        -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK_BASE" \
        --depth-noise "$s" --size-reshape --type-decode prior
  done

  note "=== PHASE fills complete ==="
}

# ------------------------------------------------- phase: full months -------
# $1 = label, $2... = extra args
month() {
  local label="$1"; shift
  note "=== PHASE month-$label (${#DAYS[@]} days) ==="
  local i=0
  for d in "${DAYS[@]}"; do
    i=$((i+1))
    note "[$i/${#DAYS[@]}] $d"
    sim "month_${label}_${d}" "$d" 09:30:00 10:00:00 "$@"
  done
  note "=== PHASE month-$label complete ==="
}

# NOT in `all` (dropped 2026-08-12): month-final does the same step-count
# ablation on the checkpoint actually shipped, which is the comparison that
# matters. Still callable as `./sweep_final.sh month-corr` if the baseline
# month is wanted too.
phase_month_corr()    { month corr    -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK_BASE" $CORR; }
phase_month_vanilla() { month vanilla -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK_BASE"; }
# Step-count ablation on the model actually shipped, rather than on the
# pre-retrain baseline it is currently established on.
phase_month_final()   { month final   -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK_FINAL" $CORR; }

# ------------------------------------------- phase: extra long-horizon ------
# The two-hour claim rests on two days, while 5.4.3 makes cross-day discipline a
# stated contribution. Each extra day is ~1 h.
LONGDAYS=("${LONGDAYS[@]:-20150107 20150123 20150116}")
phase_longhorizon() {
  note "=== PHASE longhorizon (${#LONGDAYS[@]} days, 2 h each) ==="
  for d in ${LONGDAYS[@]}; do
    sim "long_final_${d}" "$d" 10:00:00 12:00:00 \
        -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK_FINAL" $CORR
    sim "long_base_${d}"  "$d" 10:00:00 12:00:00 \
        -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK_BASE" $CORR
  done
  note "=== PHASE longhorizon complete ==="
}

# -------------------------------------------------- phase: predictive -------
# All four configurations against one real day, with the shared real-on-real
# baseline that makes the numbers interpretable without matching the paper's
# absolutes. Uses whatever run directories exist -- edit REAL/GEN if the names
# differ.
phase_predictive() {
  note "=== PHASE predictive ==="
  local L=ABIDES/log
  local REAL="$L/market_replay_INTC_2015-01-29_10-00-00_30/processed_orders.csv"

  if [ ! -f "$REAL" ]; then
    note "SKIP predictive: no real reference at $REAL"
    note "  set it to the matched market_replay CSV and re-run: $0 predictive"
    return 0
  fi

  # Resolve generated files by glob so exact suffixes do not have to be guessed.
  local args=()
  add() {  # add <label> <glob>
    local p; p=$(ls -d $2 2>/dev/null | head -1)
    if [ -n "$p" ] && [ -f "$p/processed_orders.csv" ]; then
      args+=(--gen "$1=$p/processed_orders.csv"); note "  predictive: $1 <- $p"
    else
      note "  predictive: $1 NOT FOUND ($2)"
    fi
  }
  add inherited_ddim1        "$L/world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_1_*"
  add inherited_ddpm100      "$L/world_agent_INTC_2015-01-29_10-00-00_30_DDPM_0.0_100_*"
  add ours_0724_fixed        "$L/world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.724*tdprior_sr_dn0.3"
  add ours_ss_e4_fixed       "$L/world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.69*tdprior_sr_dn0.3"

  if [ ${#args[@]} -eq 0 ]; then note "SKIP predictive: no generated files matched"; return 0; fi

  python -u -m evaluation.quantitative_eval.predictive_batch \
    --real "$REAL" "${args[@]}" --seeds 3 --out-dir "$OUT/predictive" \
    2>&1 | tee "$OUT/predictive.log"
  note "predictive done -> $OUT/predictive"
}

# ------------------------------------------------------------- summary ------
summary() {
  echo; echo "================= PROGRESS ================="; cat "$PROG"
  echo; echo "================= DIAG LINES ================="
  for log in "$OUT"/*.log; do
    case "$log" in *old.log|*predictive*) continue;; esac
    [ -e "$log" ] || continue
    echo; echo "--- $(basename "$log")"
    grep -E 'decoded_pre_drop|placed|drops|execution_channels|depth_pre_drop|resample|cond_z|Time taken|per order' \
         "$log" 2>/dev/null | head -20 || echo "(none)"
  done
  echo
  echo "NEXT, on your own machine (needs jax, python >= 3.10):"
  echo "  pull the run directories and score LOB-Bench"
  echo "  score unique mids / flow mix from the CSVs -- they are not in stdout"
}

# ---------------------------------------------------------------- preflight -
note "sweep starting -> $OUT"
[ -f ABIDES/abides.py ] || { note "FATAL: not in repo root"; exit 1; }
for f in UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG; do
  [ -e ".flagstash/$f" ] && [ ! -e "$f" ] && mv ".flagstash/$f" .
  [ -e "$f" ] || touch "$f"
  note "flag $f ON"
done
for c in "$CK_BASE" "$CK_FINAL"; do
  [ -f "$c" ] || note "WARNING: no local copy of $c"
done

case "${1:-all}" in
  fills)          phase_fills ;;
  month-corr)     phase_month_corr ;;
  month-vanilla)  phase_month_vanilla ;;
  month-final)    phase_month_final ;;
  longhorizon)    phase_longhorizon ;;
  predictive)     phase_predictive ;;
  all)            phase_fills; phase_month_final; phase_predictive; \
                  phase_longhorizon; phase_month_vanilla ;;
  summary)        summary; exit 0 ;;
  *) sed -n '3,30p' "$0"; exit 1 ;;
esac

note "sweep finished"
summary
