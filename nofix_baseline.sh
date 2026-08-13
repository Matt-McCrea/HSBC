#!/usr/bin/env bash
#
# nofix_baseline.sh — build a TRADES-default (no-fixes) checkpoint and run the
# baseline sims on it. Run from the repo root.
#
#   tmux new -s nofix
#   ./nofix_baseline.sh all
#
# Stages (each runs alone; `all` runs them in order):
#
#   config    edit configuration.py to the pre-fix hyperparameters. BACKS UP.
#   prep      rebuild preprocessing with ALL THREE fixes off, then verify
#             the regenerated normalisation stats look like the originals.
#   train     train for at most 5 h, saving a checkpoint every epoch.
#   pick      report the newest checkpoint produced.
#   sims      freeze demo + controls on that checkpoint, then the full month.
#   restore   put configuration.py back. RUN THIS WHEN DONE.
#
# WHAT "NO FIXES" MEANS -- all three, and they are independent:
#   UNCLAMP_DEPTH=0   depth targets clamped at 0 again
#   PRICE_REANCHOR=0  absolute price z-scoring, not deviation from the day open
#   DEPTH_INDEX_FIX=0 the self-referential index=j for event_type==1
#
# constants.py reads UNCLAMP_DEPTH and PRICE_REANCHOR as
#   (env == "1") OR os.path.exists(<FLAG file>)
# which is an OR, not an override: a leftover flag file wins regardless of the
# env var. sweep_final.sh and battery_0724.sh both leave these files PRESENT
# when they finish (0.724 needs them on). So this script explicitly moves them
# out of the way too, not just sets env vars -- discovered 2026-08-13 when the
# rebuilt depth stats matched the original exactly (expected: depth is a
# same-day price DIFFERENCE, so a constant anchor cancels out of it and depth
# alone can never prove PRICE_REANCHOR is off) but price stats still showed the
# reanchored shape (mean ~0, not the ~3620 the code's own comment documents for
# the absolute convention).
#
# WHY THIS IS WORTH A TRAINING RUN: with the clamp and the index bug, 0.00% of
# real depth targets are negative, so the model cannot learn marketable orders
# at all. That is the pathology section 5.3 describes. This checkpoint should
# therefore FREEZE under deterministic sampling -- which both gives the
# TRADES-default month baseline and restores the freeze demonstration that went
# with the lost checkpoints.
#
# No `set -e`: a failure in one stage must not abandon the rest.
set -u

NOFIX_ENV=(UNCLAMP_DEPTH=0 PRICE_REANCHOR=0 DEPTH_INDEX_FIX=0)
# RESUME_TRAINING_FLAG and SCHEDULED_SAMPLING_FLAG are unrelated to the
# no-fixes DATA pipeline but corrupt a from-scratch baseline just as badly:
# RESUME_TRAINING_FLAG (run.py, file-only, no env escape) makes training
# resume from whatever .ckpt sits in data/checkpoints/TRADES -- someone
# else's converged, fixes-trained weights -- instead of starting fresh.
# SCHEDULED_SAMPLING_FLAG turns on conditioning-on-own-output during
# training, which is not part of a baseline reproduction. Both must be
# ABSENT for `train`, caught 2026-08-13 after a resumed, scheduled-sampling
# run on no-fixes data produced the same instability as the original paper
# reproduction attempts (loss jumping 1.0 -> 2.9 around step 11k and
# staying there -- a converged model's input distribution pulled out from
# under it, compounded by partly conditioning on its own output).
FILE_FLAGS=(UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG RESUME_TRAINING_FLAG SCHEDULED_SAMPLING_FLAG)
STASH=.flagstash

# Move any of FILE_FLAGS out of the way. Must run before prep/train/sims --
# their presence overrides the env vars above regardless of value.
nofix_flags_off() {
  mkdir -p "$STASH"
  for f in "${FILE_FLAGS[@]}"; do
    [ -e "$f" ] && mv "$f" "$STASH/" && note "  stashed $f (was overriding the env var)"
  done
  for f in "${FILE_FLAGS[@]}"; do
    [ -e "$f" ] && { note "FATAL: $f still present, cannot proceed"; return 1; }
  done
  note "  confirmed absent: ${FILE_FLAGS[*]}"
  return 0
}

# Restore for anything downstream that expects them (e.g. sweep_final.sh, or
# just leaving the repo in its normal state).
nofix_flags_on() {
  for f in "${FILE_FLAGS[@]}"; do
    [ -e "$STASH/$f" ] && mv "$STASH/$f" . && note "  restored $f"
    [ -e "$f" ] || { touch "$f"; note "  created $f"; }
  done
}
TRAIN_SECONDS="${TRAIN_SECONDS:-18000}"        # 5 h
SIM_TIMEOUT="${SIM_TIMEOUT:-7200}"
SIM_TIMEOUT_DDPM="${SIM_TIMEOUT_DDPM:-10800}"

CFG=configuration.py
CKDIR=data/checkpoints
STATS=data/INTC/normalization_stats.json
OUT="nofix_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
PROG="$OUT/PROGRESS.txt"
BACKUP="$OUT/configuration.py.orig"

DAYS=(20150102 20150105 20150106 20150107 20150108 20150109 20150112 20150113 \
      20150114 20150115 20150116 20150120 20150121 20150122 20150123 20150126 \
      20150127 20150128 20150129 20150130)

note() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$PROG"; }

# ------------------------------------------------------------- stage config -
# The pre-fix pair. These are TRADES's own values, from their repository.
stage_config() {
  note "=== config ==="
  [ -f "$CFG" ] || { note "FATAL: no $CFG -- run from the repo root"; return 1; }
  cp "$CFG" "$BACKUP"; note "backed up $CFG -> $BACKUP"

  python3 - "$CFG" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s2 = s.replace('LearningHyperParameter.LEARNING_RATE] = 0.00025',
               'LearningHyperParameter.LEARNING_RATE] = 0.001')
s2 = s2.replace('LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1',
                'LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.0')
open(p,'w').write(s2)
print("  LEARNING_RATE      0.00025 -> 0.001")
print("  CONDITIONAL_DROPOUT    0.1 -> 0.0")
PY
  note "NOTE: plain DROPOUT (line ~73) left at 0.1 -- only CONDITIONAL_DROPOUT and"
  note "      LEARNING_RATE are the documented pre-fix differences. Change by hand"
  note "      if the TRADES repo had a different value."
  grep -nE "LEARNING_RATE\]|CONDITIONAL_DROPOUT\]" "$CFG" | tee -a "$PROG"
}

stage_restore() {
  note "=== restore ==="
  if [ -f "$BACKUP" ]; then cp "$BACKUP" "$CFG"; note "restored $CFG from $BACKUP"
  else note "no backup at $BACKUP -- nothing restored"; fi
  grep -nE "LEARNING_RATE\]|CONDITIONAL_DROPOUT\]" "$CFG" | tee -a "$PROG"
  nofix_flags_on
}

# --------------------------------------------------------------- stage prep -
stage_prep() {
  note "=== prep (rebuilding with all three fixes OFF) ==="
  nofix_flags_off || return 1
  note "contents of data/INTC before:"
  ls -la data/INTC 2>/dev/null | tee -a "$PROG"
  [ -f "$STATS" ] && { cp "$STATS" "$OUT/normalization_stats.before.json"; note "saved old stats"; }

  note "launching build -- if the dataset is cached this will be quick and the"
  note "stats will NOT change, which is how you will know a rebuild is needed."
  env "${NOFIX_ENV[@]}" python -u main.py 2>&1 | tee "$OUT/prep.log" &
  local pid=$!
  # The build happens before training starts; give it time then check the stats.
  sleep 60
  while kill -0 $pid 2>/dev/null; do
    if [ -f "$STATS" ] && [ "$STATS" -nt "$OUT/normalization_stats.before.json" ]; then break; fi
    sleep 30
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null

  note "--- STATS CHECK ---"
  if [ -f "$STATS" ]; then
    cp "$STATS" "$OUT/normalization_stats.after.json"
    python3 - "$STATS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
def find(sub):
    return {k: v for k, v in d.items() if sub in k.lower()}
print("depth entries:", json.dumps(find('depth'), indent=2)[:600])
print()
print("EXPECTED for a genuine no-fixes rebuild: depth mean ~1.3847, std ~2.6777")
print("(these are the constants behind z* = -0.517 in the draft).")
print("If std is still in the tens of thousands, the rebuild did NOT happen or")
print("a fix is still on -- do not train until this is right.")
PY
  else
    note "no stats file at $STATS"
  fi
}

# -------------------------------------------------------------- stage train -
stage_train() {
  note "=== train (cap ${TRAIN_SECONDS}s, one checkpoint per epoch) ==="
  nofix_flags_off || return 1
  ls -1 "$CKDIR"/TRADES/*.ckpt 2>/dev/null | wc -l | xargs -I{} note "checkpoints before: {}"
  timeout "$TRAIN_SECONDS" env "${NOFIX_ENV[@]}" KEEP_EPOCH_CHECKPOINTS=1 \
      python -u main.py 2>&1 | tee "$OUT/train.log"
  local rc=${PIPESTATUS[0]}
  case $rc in
    0)   note "training finished on its own" ;;
    124) note "training hit the ${TRAIN_SECONDS}s cap -- expected, checkpoints kept" ;;
    *)   note "training exited $rc -- see $OUT/train.log" ;;
  esac
}

# --------------------------------------------------------------- stage pick -
newest_ckpt() { ls -t "$CKDIR"/TRADES/*.ckpt 2>/dev/null | head -1; }

stage_pick() {
  note "=== pick ==="
  local c; c=$(newest_ckpt)
  if [ -z "$c" ]; then note "no checkpoint found under $CKDIR/TRADES"; return 1; fi
  note "most advanced checkpoint: $c"
  ls -lt "$CKDIR"/TRADES/*.ckpt 2>/dev/null | head -10 | tee -a "$PROG"
  echo "$c" > "$OUT/CHECKPOINT.txt"
}

# --------------------------------------------------------------- stage sims -
sim() {
  local name="$1" date="$2" st="$3" et="$4"; shift 4
  local log="$OUT/${name}.log" done="$OUT/${name}.done"
  [ -e "$done" ] && { note "skip  $name"; return 0; }
  [ -e "$log" ] && mv "$log" "${log%.log}.$(date +%H%M%S).old.log"
  local cap="$SIM_TIMEOUT"
  case " $* " in *" -type DDPM "*) cap="$SIM_TIMEOUT_DDPM";; esac
  note "START $name (cap ${cap}s)"
  local t0; t0=$(date +%s)
  timeout "$cap" env "${NOFIX_ENV[@]}" python -u ABIDES/abides.py -c world_agent_sim \
      -t INTC -date "$date" -st "$st" -et "$et" -d True -m TRADES -seed 30 "$@" \
      2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]} dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then touch "$done"; note "DONE  $name (${dt}s)"
  elif [ "$rc" -eq 124 ]; then note "TIMEOUT $name (${dt}s)"
  else note "FAIL  $name (exit $rc, ${dt}s)"; fi
  return 0
}

stage_sims() {
  note "=== sims ==="
  nofix_flags_off || return 1
  local CK; CK=$(cat "$OUT/CHECKPOINT.txt" 2>/dev/null || newest_ckpt)
  if [ -z "$CK" ]; then note "no checkpoint -- run pick first"; return 1; fi
  note "using $CK"

  # 1. THE FREEZE DEMONSTRATION. This is the point of the whole exercise.
  sim freeze_ddim10_eta0 20150130 09:30:00 10:00:00 \
      -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CK"
  note "CHECK: under ~10 unique mids with executions near zero = it freezes."
  note "       That reproduces section 5.3 on a checkpoint that still exists."

  # 2. The control that makes the argument: same weights, stochastic sampler.
  sim control_ddpm100 20150130 09:30:00 10:00:00 \
      -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK"

  # 3. eta=1, for the remedy (i) comparison in 5.3.2.
  sim eta1 20150130 09:30:00 10:00:00 \
      -type DDIM -nsteps 10 -eta 1.0 --ckpt-path "$CK"

  # 4. The TRADES-default month. This is what replaces `month-vanilla` in
  #    sweep_final.sh with a genuine published-configuration baseline.
  note "=== month (TRADES-default, ${#DAYS[@]} days) ==="
  local i=0
  for d in "${DAYS[@]}"; do
    i=$((i+1)); note "[$i/${#DAYS[@]}] $d"
    sim "month_nofix_${d}" "$d" 09:30:00 10:00:00 \
        -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CK"
  done
}

summary() {
  echo; echo "============ PROGRESS ============"; cat "$PROG" 2>/dev/null
  echo; echo "============ DIAG ============"
  for l in "$OUT"/freeze_*.log "$OUT"/control_*.log "$OUT"/eta1*.log; do
    [ -e "$l" ] || continue
    echo; echo "--- $(basename "$l")"
    grep -E 'decoded_pre_drop|placed|drops|execution_channels|depth_pre_drop|cond_z|per order' "$l" | head -15
  done
  echo
  echo "REMEMBER: ./nofix_baseline.sh restore  -- configuration.py is still edited."
}

note "nofix baseline -> $OUT"
[ -f ABIDES/abides.py ] || { note "FATAL: run from the repo root"; exit 1; }

case "${1:-}" in
  config)  stage_config ;;
  prep)    stage_prep ;;
  train)   stage_train ;;
  pick)    stage_pick ;;
  sims)    stage_sims ;;
  restore) stage_restore ;;
  summary) summary; exit 0 ;;
  all)     stage_config && stage_prep; stage_train; stage_pick; stage_sims; summary ;;
  *) sed -n '3,20p' "$0"; exit 1 ;;
esac

note "done"
