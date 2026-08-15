#!/usr/bin/env bash
#
# diagnostic_battery.sh — re-run the section 5.3 diagnosis on a retained
# pre-fix vanilla checkpoint, and the type-decode cell on 0.724.
#
# Copy to the HSBC repo root and run from there.
#
#   ./diagnostic_battery.sh check    # preflight only, changes nothing
#   ./diagnostic_battery.sh gate     # run 1, then print the go/no-go numbers
#   ./diagnostic_battery.sh core     # runs 2 and 3 (eta=1, DDPM-100)
#   ./diagnostic_battery.sh extra    # hybrids + DDIM-100 divergence
#   ./diagnostic_battery.sh r1       # type-decode l1 on 0.724 (FLAGS BACK ON)
#   ./diagnostic_battery.sh summary  # print DIAG lines from every log so far
#
# WHY THE STAGES: the whole battery is only worth running if the checkpoint
# actually freezes under deterministic sampling. Stage `gate` answers that in
# ~16 minutes. If it does not freeze, stop — a model that does not freeze
# cannot be used to diagnose a freeze.
#
set -euo pipefail

# ---------------------------------------------------------------- config ----
# FILL THIS IN. The pre-fix vanilla checkpoint (the very first one).
CKPT="${CKPT:-data/checkpoints/TRADES/CHANGE_ME.ckpt}"

# The adopted checkpoint, for stage r1 only.
CKPT_0724="${CKPT_0724:-data/checkpoints/TRADES/val_ema=0.724_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt}"

TICKER=INTC
DATE=20150130
ST=09:30:00
ET=10:00:00
SEED=30
OUT="diag_runs_$(date +%Y%m%d)"
STASH=".flagstash"

FLAGS=(UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG)

# ------------------------------------------------------------- utilities ----
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

flags_state() {
  local present=() absent=()
  for f in "${FLAGS[@]}"; do
    [[ -e "$f" ]] && present+=("$f") || absent+=("$f")
  done
  printf 'present: %s\n' "${present[*]:-<none>}"
  printf 'absent : %s\n' "${absent[*]:-<none>}"
}

flags_off() {
  mkdir -p "$STASH"
  for f in "${FLAGS[@]}"; do
    [[ -e "$f" ]] && { mv "$f" "$STASH/"; echo "stashed $f"; }
  done
  return 0
}

flags_on() {
  for f in "${FLAGS[@]}"; do
    [[ -e "$STASH/$f" ]] && { mv "$STASH/$f" .; echo "restored $f"; }
    [[ -e "$f" ]] || { touch "$f"; echo "created $f"; }
  done
  return 0
}

require_flags_absent() {
  for f in "${FLAGS[@]}"; do
    [[ -e "$f" ]] && die "$f is present. This checkpoint was trained WITHOUT the pipeline fixes, so both flags must be absent. Run: $0 check"
  done
}

# Run one simulation, capturing stdout. Never overwrites an existing log.
run() {
  local name="$1"; shift
  local log="$OUT/${name}.log"
  mkdir -p "$OUT"
  [[ -e "$log" ]] && { echo "SKIP $name (log exists: $log)"; return 0; }
  say "RUN $name"
  echo "flags at launch:"; flags_state
  echo "cmd: python ABIDES/abides.py $*"
  local t0=$SECONDS
  # tee so you can watch it and still keep the log
  python ABIDES/abides.py "$@" > >(tee "$log") 2>&1 || die "$name failed, see $log"
  printf 'elapsed: %s s\n' "$(( SECONDS - t0 ))" | tee -a "$log"
}

base_args() {
  echo -c world_agent_sim -t "$TICKER" -date "$DATE" \
       -st "$ST" -et "$ET" -d True -m TRADES -seed "$SEED"
}

# --------------------------------------------------------------- stages -----
stage_check() {
  say "PREFLIGHT"
  [[ -f ABIDES/abides.py ]] || die "run me from the HSBC repo root (no ABIDES/abides.py here)"
  echo "repo root: $(pwd)"
  echo
  echo "flag files:"; flags_state
  echo
  if [[ "$CKPT" == *CHANGE_ME* ]]; then
    die "set CKPT at the top of this script, or export CKPT=... before running"
  fi
  [[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
  echo "vanilla ckpt : $CKPT"
  [[ -f "$CKPT_0724" ]] && echo "0.724 ckpt   : $CKPT_0724" \
                        || echo "0.724 ckpt   : NOT FOUND (stage r1 will fail)"
  echo
  echo "window       : $TICKER $DATE $ST-$ET seed $SEED"
  echo "output dir   : $OUT"
  echo
  say "Both flags must be ABSENT for stages gate/core/extra."
  echo "Stage r1 restores them. Nothing is deleted -- flags are moved to $STASH/."
}

stage_gate() {
  stage_check
  flags_off
  require_flags_absent
  run "01_ddim10_eta0_vanilla" $(base_args) -type DDIM -nsteps 10 -eta 0.0 --ckpt-path "$CKPT"

  say "GO / NO-GO"
  cat <<'EOF'
Check the log against the 0.681 diagnosis. Looking for:

  decoded_pre_drop  market share ~2-3%      (type channel NOT the problem)
  drops: size_range ~a fifth to a quarter of decodes
  depth_pre_drop    neg bin at or near zero
  execution_channels  B_crossing_limit at or near zero
  cond_z[price]     mean strongly negative (no PRICE_REANCHOR)

Then score the CSV for unique mid-prices and executed share:
  3-6 mids, executions ~0   -> IT FREEZES. run: ./diagnostic_battery.sh core
  13+ mids                  -> it does not freeze. STOP, this checkpoint
                               cannot carry the section 5.3 diagnosis.
EOF
  echo
  grep -E 'DIAG|decoded_pre_drop|drops|execution_channels|depth_pre_drop|cond_z|resample' \
       "$OUT/01_ddim10_eta0_vanilla.log" || echo "(no DIAG lines found -- check the agent is instrumented)"
}

stage_core() {
  require_flags_absent
  run "02_ddim10_eta1_vanilla" $(base_args) -type DDIM -nsteps 10 -eta 1.0 --ckpt-path "$CKPT"
  run "03_ddpm100_vanilla"     $(base_args) -type DDPM -nsteps 100 -eta 0.0 --ckpt-path "$CKPT"
  say "CORE DONE"
  cat <<'EOF'
Run 02 gives the neg-depth share under eta=1 -- the number 5.3.2 remedy (i)
is currently missing. Expect it BETWEEN the eta=0 value and the DDPM value,
not zero: eta=1 does inject noise into the depth channel, just less than
DDPM's learned variance does. A zero would contradict the theory argument.

Run 03 is the control that makes the whole argument: the marketable tail
exists in the learned distribution and the deterministic sampler does not
reach it. Compare its neg-depth share against run 01's.
EOF
}

stage_extra() {
  require_flags_absent
  # !! VERIFY THESE SAMPLER TOKENS against the sweep scripts before trusting.
  # The run-directory convention implies HYBRID_DDPM_PP / HYBRID_PP_DDPM but
  # the head/tail split may need an extra argument.
  run "04_hybrid_ddpm_head" $(base_args) -type HYBRID_DDPM_PP -nsteps 10 -eta 0.0 --ckpt-path "$CKPT" || true
  run "05_hybrid_ode_head"  $(base_args) -type HYBRID_PP_DDPM -nsteps 10 -eta 0.0 --ckpt-path "$CKPT" || true
  run "06_ddim100_eta0"     $(base_args) -type DDIM -nsteps 100 -eta 0.0 --ckpt-path "$CKPT"
  say "EXTRA DONE"
  echo "Note: the hybrid result is known checkpoint-dependent (113 mids on one"
  echo "checkpoint, 12 on another). If it does not reproduce that is the"
  echo "checkpoint-dependence finding of 5.3.3, not a failed run."
}

stage_r1() {
  say "RESTORING FLAGS for the 0.724 run"
  flags_on
  flags_state
  [[ -f "$CKPT_0724" ]] || die "0.724 checkpoint not found: $CKPT_0724"
  # l1 type decode = omit --type-decode entirely. Keep the other two corrections.
  run "07_0724_typedecode_l1" $(base_args) -type DDIM -nsteps 10 -eta 0.0 \
      --ckpt-path "$CKPT_0724" --depth-noise 0.3 --size-reshape
  say "R1 DONE"
  echo "Compare the decoded market share against the existing _tdprior_sr_dn0.3"
  echo "run at matched sigma. Real is near 3%."
}

stage_summary() {
  say "DIAG LINES BY RUN"
  for log in "$OUT"/*.log; do
    [[ -e "$log" ]] || continue
    printf '\n\033[1m--- %s\033[0m\n' "$(basename "$log")"
    grep -E 'decoded_pre_drop|placed|drops|execution_channels|depth_pre_drop|size_pre_drop|resample|cond_z|Time taken' \
         "$log" || echo "(none)"
  done
  echo
  echo "Unique mid-prices and flow mix are NOT in stdout -- score the CSVs with"
  echo "evaluation/quantitative_eval/flow_mix.py or your movemetric helper."
}

# ----------------------------------------------------------------- main -----
case "${1:-}" in
  check)   stage_check ;;
  gate)    stage_gate ;;
  core)    stage_core ;;
  extra)   stage_extra ;;
  r1)      stage_r1 ;;
  summary) stage_summary ;;
  *) sed -n '3,20p' "$0"; exit 1 ;;
esac
