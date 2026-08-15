#!/bin/bash
# long_horizon_and_replication.sh
#
# PRIORITY 1 --- long-horizon test of the final-model candidates.
#   Two-hour runs (10:00-12:00, INTC 2015-01-29) matching the window in which the
#   single-step replication collapsed (~minute 73). Until this exists we cannot claim the
#   decode-time fix survives the regime where the replication failed --- every evaluation of
#   the final models to date stops at minute 30.
#
# PRIORITY 2 --- sampler ABLATION on the final-model checkpoint. NOT a replication.
#   Runs vanilla DDIM-1 and vanilla DDIM-10 on ckpt 0.724_epoch=0, which already carries
#   PRICE_REANCHOR and UNCLAMP_DEPTH. The question is NOT "what does TRADES do" --- 0.724
#   contains our own data-pipeline fixes and cannot stand in for the published model. The
#   question is:
#
#       do the data-pipeline fixes ALONE rescue accelerated sampling, or is the decode-time
#       intervention still doing independent work?
#
#   If these still fail, depth-noise is necessary rather than papering over a data bug ---
#   which is the argument \S5.3 needs. If they do not fail, that is a surprising result worth
#   knowing before submission. Either outcome belongs in \S5.2/\S5.3 and MUST NOT be presented
#   as a replication of TRADES in \S5.1.
#
#   An earlier version of this script tried to build a controlled DDPM-100/DDIM-1 replication
#   pair on ckpt 0.681. That was dropped: the 0.681 on disk (dropout 0.1, lr 2.5e-4, 28 Jul)
#   is a DIFFERENT checkpoint from the one behind the documented replication (Jul-5 lineage,
#   not retained) --- they share only a rounded val_ema. No pre-fix checkpoint survives, so a
#   controlled pre-fix pair is not reconstructable. See analysis/results_51_replication_numbers.md.
#
# FLAG HANDLING (important): UNCLAMP_DEPTH/PRICE_REANCHOR are preprocessing+conditioning
# settings that must MATCH the checkpoint's training. The final-model candidates were trained
# WITH them; checkpoint 0.681 predates both and must run WITHOUT them. This script toggles the
# flag files per run and restores "both present" on exit, since that is the safe default for
# the main lineage. Do not run other sims concurrently on this machine --- they would see the
# wrong flag state.
#
# Usage:
#   bash scripts/long_horizon_and_replication.sh                # both priorities
#   bash scripts/long_horizon_and_replication.sh --only p1      # long-horizon only
#   bash scripts/long_horizon_and_replication.sh --only p2      # replication pair only
#   bash scripts/long_horizon_and_replication.sh --with-lever   # p1 also runs +book-balance
#
# Resumable: each cell writes a .done sentinel; re-running skips completed cells.
set -uo pipefail

TICKER="INTC"; SEED="30"
WINCFG="--depth-noise 0.3 --size-reshape --type-decode prior"
LEVER="--book-target-thick 2.0 --book-cancel-rate 0.5"
ONLY="both"; WITH_LEVER=0
OUT_DIR="longrun/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --only) ONLY="$2"; shift 2;;
  --with-lever) WITH_LEVER=1; shift;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs"
SUM="$OUT_DIR/summary.md"
: > "$SUM"
mkdir -p longrun; ln -sfn "$(basename "$OUT_DIR")" longrun/latest

# ---- flag state management -------------------------------------------------
# restore both flags on ANY exit path, so a crash can't leave the repo configured for a
# pre-fix checkpoint and silently corrupt a later run on the main lineage.
restore_flags () { touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG; }
trap restore_flags EXIT INT TERM

set_flags () {   # set_flags on|off
  if [[ "$1" == "on" ]]; then touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
  else rm -f UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG; fi
  local got; got=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
  local want; [[ "$1" == "on" ]] && want="True True" || want="False False"
  [[ "$got" == "$want" ]] || { echo "!! flag state wrong: wanted [$want] got [$got]"; exit 1; }
}

# ---- checkpoint lookup -----------------------------------------------------
# checkpoints have moved between directories across sessions; search all the likely homes
# rather than assuming one. Matches on the full distinguishing filename fragment, because
# e.g. 0.69_epoch=2 and 0.69_epoch=4 are different checkpoints from different lineages.
find_ckpt () {
  local frag="$1" hit
  for d in data/checkpoints/TRADES data/checkpoints/TRADES_ddpm_rollout_pretrain \
           data/checkpoints/TRADES_other_recovered data/checkpoints; do
    hit=$(ls "$d"/*"$frag"*.ckpt 2>/dev/null | head -1)
    [[ -n "$hit" ]] && { echo "$hit"; return 0; }
  done
  return 1
}

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }

ensure_real () {   # ensure_real <yyyymmdd> <st> <et>
  local D="$1" ST="$2" ET="$3"
  local rp="ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$D")_${ET//:/-}_${SEED}/processed_orders.csv"
  if [[ ! -f "$rp" ]]; then
    echo "   -- generating real replay $D $ST-$ET" >&2
    python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
      > "$OUT_DIR/logs/real_${D}_${ET//:/-}.txt" 2>&1
  fi
  echo "$rp"
}

# ---- one simulation cell ---------------------------------------------------
run_cell () {   # run_cell <tag> <ckpt> <date> <st> <et> <sampler> <nsteps> <flags on|off> <decode args...>
  local TAG="$1" CK="$2" D="$3" ST="$4" ET="$5" SAMP="$6" NST="$7" FL="$8"; shift 8
  local DEC="$*"
  local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "   SKIP $TAG (already done)"; return; }

  set_flags "$FL"
  local REALP; REALP=$(ensure_real "$D" "$ST" "$ET")

  echo "-- $TAG   [$SAMP-$NST, flags=$FL]"
  echo "     ckpt: $(basename "$CK")"
  local T0; T0=$(date +%s)
  if ! python -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
        -d True -m TRADES -type "$SAMP" -nsteps "$NST" -eta 0.0 \
        --ckpt-path "$CK" -seed "$SEED" $DEC \
        > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
    echo "   ERROR --- see logs/${TAG}.txt"
    { echo "## $TAG --- ERROR"; echo ""; } >> "$SUM"; return
  fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1)

  { echo "## $TAG"
    echo '```'
    echo "sampler:  $SAMP-$NST   flags: $FL"
    echo "ckpt:     $CK"
    echo "window:   $D $ST-$ET"
    echo "decode:   ${DEC:-<none, vanilla>}"
    echo "wallclock: ${SECS}s  ($((SECS/60))m $((SECS%60))s)"
    echo "csv:      ${CSV:-none}"
    echo "real:     $REALP"
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"
  echo "   done in $((SECS/60))m $((SECS%60))s"
}

# ---- preflight -------------------------------------------------------------
echo "=== preflight ==="
if pgrep -f "main.py" > /dev/null; then
  echo "!! training (main.py) is running --- GPU contention will inflate every timing below."
  echo "   Kill it first, or the wall-clock numbers are not usable as evidence."
  exit 1
fi
[[ -f data/quantile_targets/real_size_limit.npy ]] || {
  echo "-- building quantile targets (needed by --size-reshape)"
  python scripts/build_quantile_targets.py > "$OUT_DIR/logs/build_targets.txt" 2>&1; }
[[ -f data/quantile_targets/real_size_limit.npy ]] || { echo "!! quantile targets missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ============================ PRIORITY 1 ====================================
if [[ "$ONLY" == "both" || "$ONLY" == "p1" ]]; then
  echo "########## PRIORITY 1: two-hour runs, final-model candidates ##########"
  echo "# window matches the single-step replication collapse (~min 73)"
  echo ""
  for SPEC in "final_0724:0.724_epoch=0" "ss_e4_069:0.69_epoch=4"; do
    LBL="${SPEC%%:*}"; FRAG="${SPEC##*:}"
    if ! CK=$(find_ckpt "$FRAG"); then
      echo "!! checkpoint matching '$FRAG' not found --- skipping $LBL"
      echo "   (searched data/checkpoints/TRADES{,_ddpm_rollout_pretrain,_other_recovered})"
      continue
    fi
    run_cell "p1_${LBL}_2h" "$CK" 20150129 "10:00:00" "12:00:00" DDIM 10 on $WINCFG
    if [[ "$WITH_LEVER" == "1" ]]; then
      run_cell "p1_${LBL}_2h_lever" "$CK" 20150129 "10:00:00" "12:00:00" DDIM 10 on $WINCFG $LEVER
    fi
  done
  echo ""
fi

# ============================ PRIORITY 2 ====================================
if [[ "$ONLY" == "both" || "$ONLY" == "p2" ]]; then
  echo "########## PRIORITY 2: sampler ablation on the final-model checkpoint ##########"
  echo "# NOT a replication --- 0.724 carries PRICE_REANCHOR + UNCLAMP_DEPTH (our fixes)."
  echo "# Asks: do the data fixes alone rescue accelerated sampling, or is depth-noise still needed?"
  echo ""
  if CK=$(find_ckpt "0.724_epoch=0"); then
    # flags ON: 0.724 was trained with both fixes, so sim-time conditioning must match.
    # vanilla decode (no depth-noise / size-reshape / type-decode) isolates the sampler.
    run_cell "p2_abl_ddim1_vanilla"  "$CK" 20150129 "09:30:00" "10:00:00" DDIM 1  on
    run_cell "p2_abl_ddim10_vanilla" "$CK" 20150129 "09:30:00" "10:00:00" DDIM 10 on
    # reference point: same checkpoint/day/window WITH the decode-time fix, so the three
    # rows form a clean like-for-like ablation table.
    run_cell "p2_abl_ddim10_fixed"   "$CK" 20150129 "09:30:00" "10:00:00" DDIM 10 on $WINCFG
  else
    echo "!! checkpoint 0.724_epoch=0 not found on this machine."
    echo "   Copy it into data/checkpoints/TRADES/, then re-run with --only p2"
  fi
  echo ""
fi

restore_flags
echo "=========================================================="
echo " COMPLETE.  Summary: $SUM"
echo " Logs:      $OUT_DIR/logs/"
echo " Flags restored to: UNCLAMP_DEPTH + PRICE_REANCHOR present."
echo ""
echo " Next: pull the CSVs listed in the summary back to the Mac and run"
echo "   python3 evaluation/stylized_custom/paper_style_stylized_facts.py <real> <gen> <out.png>"
echo "   bash scripts/lob_bench_multiday.sh <dir> <label> <outdir>"
echo "=========================================================="
cat "$SUM"
