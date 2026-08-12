#!/usr/bin/env bash
#
# run_diagnostics.sh — section 5.3 diagnosis on an old no-fixes checkpoint,
# plus the type-decode cell on the adopted checkpoint. Run from the repo root.
#
#   ./run_diagnostics.sh check          preflight, changes nothing
#   ./run_diagnostics.sh gate  [CKPT]   does it freeze? ~16 min. FLAGS OFF.
#   ./run_diagnostics.sh core  [CKPT]   eta=1 and DDPM-100. ~38 min. FLAGS OFF.
#   ./run_diagnostics.sh extra [CKPT]   DDIM-100 divergence. ~22 min. FLAGS OFF.
#   ./run_diagnostics.sh r1             type-decode l1 on 0.724. ~16 min. FLAGS ON.
#   ./run_diagnostics.sh summary        DIAG lines from every log so far
#
# THE RULE: flag files at simulation must match flag files at training.
#   old checkpoints (0.7_epoch=2, 0.704_epoch=0)  -> trained WITHOUT -> flags OFF
#   adopted checkpoint (0.724_epoch=0)            -> trained WITH    -> flags ON
# The stages set this for you. Flags are moved to .flagstash/, never deleted.
#
# WHY gate IS SEPARATE: the battery is only worth running if the checkpoint
# freezes under deterministic sampling. A model that does not freeze cannot be
# used to diagnose a freeze. gate answers that in one run.
#
set -euo pipefail

CKDIR=data/checkpoints/TRADES
CK_OLD_DEFAULT="$CKDIR/val_ema=0.7_epoch=2_INTC_se_256_au_64_CD_8_seed_30.ckpt"
CK_OLD_ALT="$CKDIR/val_ema=0.704_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt"
CK_0724="$CKDIR/val_ema=0.724_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt"
# Checkpoints live on the remote filestore. A "not here" in `check` means only
# that there is no local copy -- it says nothing about the remote.

TICKER=INTC; DATE=20150130; ST=09:30:00; ET=10:00:00; SEED=30
FLAGS=(UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG)
STASH=.flagstash

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }
tag()  { basename "${1%.ckpt}" | sed 's/_INTC.*//; s/val_ema=//'; }

flags_show() {
  for f in "${FLAGS[@]}"; do
    printf '  %-22s %s\n' "$f" "$([ -e "$f" ] && echo PRESENT || echo absent)"
  done
}
flags_off() {
  mkdir -p "$STASH"
  for f in "${FLAGS[@]}"; do [ -e "$f" ] && mv "$f" "$STASH/" && echo "  stashed $f"; done
  return 0
}
flags_on() {
  for f in "${FLAGS[@]}"; do
    [ -e "$STASH/$f" ] && mv "$STASH/$f" . && echo "  restored $f"
    [ -e "$f" ] || { touch "$f"; echo "  created $f"; }
  done
  return 0
}
need_off() {
  for f in "${FLAGS[@]}"; do
    [ -e "$f" ] && die "$f present. Old checkpoints were trained without it. Run: $0 check"
  done
  # MUST return 0 explicitly. Without this the loop's final failing [ -e ] test
  # becomes the function's exit status, and under `set -e` bash exits silently
  # -- in the case where everything is CORRECT and no flag is present.
  return 0
}

# run <logname> <extra args...>
run() {
  local name="$1"; shift
  local log="logs_diag/${name}.log"
  mkdir -p logs_diag
  [ -e "$log" ] && { echo "SKIP $name (log exists)"; return 0; }
  say "RUN $name"
  flags_show
  local t0=$SECONDS
  # python -u and a plain pipe: without them Python block-buffers stdout when it
  # is not a terminal and the run looks dead for minutes. pipefail (set above)
  # makes the pipeline return python's exit status, not tee's.
  PYTHONUNBUFFERED=1 python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" \
    -st "$ST" -et "$ET" -d True -m TRADES -seed "$SEED" "$@" \
    2>&1 | tee "$log" || die "$name failed -- see $log"
  printf 'elapsed %s s\n' "$(( SECONDS - t0 ))" | tee -a "$log"
}

stage_check() {
  say PREFLIGHT
  [ -f ABIDES/abides.py ] || die "run me from the HSBC repo root"
  echo "root: $(pwd)"
  echo; echo "flag files:"; flags_show
  echo; echo "checkpoints (local copy only -- they live on the remote):"
  for c in "$CK_OLD_DEFAULT" "$CK_OLD_ALT" "$CK_0724"; do
    printf '  %-14s %s\n' "$(tag "$c")" "$([ -f "$c" ] && echo 'local copy present' || echo 'no local copy')"
  done
  echo; echo "window: $TICKER $DATE $ST-$ET seed $SEED"
  echo "logs:   logs_diag/"
}

stage_gate() {
  local ck="${1:-$CK_OLD_DEFAULT}"
  [ -f "$ck" ] || die "checkpoint not found: $ck"
  flags_off; need_off
  run "gate_$(tag "$ck")_eta0" -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$ck"
  say "GO / NO-GO"
  cat <<'EOF'
Score the CSV for unique mid-prices and executed share.

  under ~10 mids, executions ~0  -> IT FREEZES. run:  ./run_diagnostics.sh core
  13+ mids                       -> it does not. Try the other old checkpoint:
                                    ./run_diagnostics.sh gate <other ckpt path>

In the log, the same regime as the original diagnosis looks like:
  decoded_pre_drop market share  ~2-3%   (type channel is NOT the problem)
  drops: size_range              ~a fifth to a quarter of decodes
  depth_pre_drop neg bin         at or near zero
  execution_channels             B_crossing_limit at or near zero
  cond_z[price] mean             strongly negative (no PRICE_REANCHOR)
EOF
  echo
  grep -E 'decoded_pre_drop|drops|execution_channels|depth_pre_drop|cond_z|resample' \
       "logs_diag/gate_$(tag "$ck")_eta0.log" || echo "(no DIAG lines -- is the agent instrumented?)"
}

stage_core() {
  local ck="${1:-$CK_OLD_DEFAULT}"
  [ -f "$ck" ] || die "checkpoint not found: $ck"
  need_off
  run "core_$(tag "$ck")_eta1"    -type DDIM -nsteps 10  -eta 1.0 --ckpt-path "$ck"
  run "core_$(tag "$ck")_ddpm100" -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$ck"
  say "CORE DONE"
  cat <<'EOF'
eta=1 gives the neg-depth share that 5.3.2 remedy (i) is missing. Expect it
BETWEEN the eta=0 value and the DDPM value -- not zero. eta=1 does inject noise
into the depth channel, just less than DDPM's learned variance does.

DDPM-100 is the control that makes the argument: the marketable tail exists in
the learned distribution and the deterministic sampler does not reach it.
EOF
}

stage_extra() {
  local ck="${1:-$CK_OLD_DEFAULT}"
  need_off
  run "extra_$(tag "$ck")_ddim100" -type DDIM -nsteps 100 -eta 0.0 --ckpt-path "$ck"
}

stage_r1() {
  say "FLAGS ON for the adopted checkpoint"
  flags_on; flags_show
  [ -f "$CK_0724" ] || die "not found: $CK_0724"
  # l1 type decode = omit --type-decode. Keep the other two corrections.
  run "r1_0724_typedecode_l1" -type DDIM -nsteps 10 -eta 0.0 \
      --ckpt-path "$CK_0724" --depth-noise 0.3 --size-reshape
  say "R1 DONE"
  echo "Compare the decoded market share against the existing"
  echo "_tdprior_sr_dn0.3 run at matched sigma. Real is near 3%."
}

stage_summary() {
  say "DIAG LINES BY RUN"
  shopt -s nullglob
  for log in logs_diag/*.log; do
    printf '\n\033[1m--- %s\033[0m\n' "$(basename "$log")"
    grep -E 'decoded_pre_drop|placed|drops|execution_channels|depth_pre_drop|size_pre_drop|resample|cond_z|Time taken' \
         "$log" || echo "(none)"
  done
  echo
  echo "Unique mids and flow mix are not in stdout -- score the CSVs with"
  echo "evaluation/quantitative_eval/flow_mix.py or the movemetric helper."
}

case "${1:-}" in
  check)   stage_check ;;
  gate)    stage_gate  "${2:-}" ;;
  core)    stage_core  "${2:-}" ;;
  extra)   stage_extra "${2:-}" ;;
  r1)      stage_r1 ;;
  summary) stage_summary ;;
  *) sed -n '3,20p' "$0"; exit 1 ;;
esac
