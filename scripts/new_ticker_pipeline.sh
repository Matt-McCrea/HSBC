#!/bin/bash
# new_ticker_pipeline.sh — reproduce the whole method on a different ticker and period.
#
# WHY THIS EXISTS. Two of the dissertation's methodological claims rest on INTC/January-2015 alone:
#   (1) a stable checkpoint can be found by BEHAVIOURAL search, because validation loss cannot
#       select one (0.681 froze under DDIM; 0.719 — the better loss — exploded);
#   (2) scheduled sampling improves LOB-Bench realism (0.468 -> 0.346 over 20 days).
# Neither is worth much if it is a property of that stock-month. This runs the same chain end to end
# on another Nasdaq symbol so the claims can be stated as properties of the method.
#
# STAGED AND RESUMABLE. Each phase writes .done_<phase> under the run directory and is skipped on a
# re-invocation, so a reclaimed GPU session resumes rather than restarts. Phases are invoked one at a
# time by design — several are 12h+ and chaining them unattended across a booked session has bitten
# this project before.
#
#   bash scripts/new_ticker_pipeline.sh --ticker MSFT --start 2015-03-02 --end 2015-03-31 --dry-run
#   bash scripts/new_ticker_pipeline.sh --ticker MSFT --start ... --end ... --phase setup
#   ... then train, triage, sigma, stability, ss-retrain, evaluate
#
# WHAT IS DELIBERATELY NOT ASSUMED FROM INTC:
#   * sigma. INTC's 0.3 was fitted to INTC's execution share and tick size. The `sigma` phase
#     re-brackets it. Skipping that phase invalidates everything downstream.
#   * the 1.5-2.5bp volatility band. That is INTC's real band; this prints the new ticker's own.
#   * which days are "hard". INTC's 0107/0129 were found empirically; there is no prior for a new
#     ticker, and raw event count does NOT predict it (see triage_days below).
set -uo pipefail

TICKER=""; START=""; END=""; PHASE=""; DRY=0
SEED="30"; ST="09:30:00"; ET="10:00:00"
TRIAGE_DAYS=6                 # spread-sample triage before spending a full period per ckpt
CAP_SECS=2400                 # 40 min/day — the established "unstable" criterion
LIMIT_TRAIN_BATCHES="0.15"    # short epochs => more early checkpoints (see run.py)
SIGMAS="0.15 0.3 0.45"        # first bracket; widen/narrow from the phase's own output
DEPTH_NOISE="0.3"             # used by `stability`; set from the sigma phase result
WITH_TRADES_DEFAULT=0
RUN_DIR=""
# Interpreter. The remote venv provides `python`; some systems only have `python3`. Override with
# PY=python3 to smoke-test the pipeline off the GPU box.
PY="${PY:-python}"

while [[ $# -gt 0 ]]; do case "$1" in
  --ticker) TICKER="$2"; shift 2;;
  --start) START="$2"; shift 2;;
  --end) END="$2"; shift 2;;
  --phase) PHASE="$2"; shift 2;;
  --run-dir) RUN_DIR="$2"; shift 2;;
  --triage-days) TRIAGE_DAYS="$2"; shift 2;;
  --cap-secs) CAP_SECS="$2"; shift 2;;
  --limit-train-batches) LIMIT_TRAIN_BATCHES="$2"; shift 2;;
  --sigmas) SIGMAS="$2"; shift 2;;
  --depth-noise) DEPTH_NOISE="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;
  --with-trades-default) WITH_TRADES_DEFAULT=1; shift;;
  --dry-run) DRY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

[[ -n "$TICKER" && -n "$START" && -n "$END" ]] || {
  echo "usage: $0 --ticker SYM --start YYYY-MM-DD --end YYYY-MM-DD [--phase P] [--dry-run]"; exit 1; }

# CANONICAL path, with no LOB-level suffix. ABIDES builds exactly this string
# (ABIDES/agent/WorldAgent.py:1153 and ABIDES/config/world_agent_sim.py:251) and calls os.listdir on
# it. Raw LOBSTER downloads arrive as {TICKER}_{start}_{end}_{levels}/ instead, so setup reconciles
# the two with a symlink — otherwise a new ticker trains for hours and only fails at simulation.
DATA_DIR="data/${TICKER}/${TICKER}_${START}_${END}"
[[ -n "$RUN_DIR" ]] || RUN_DIR="pipeline/${TICKER}_${START}_${END}"
mkdir -p "$RUN_DIR/logs"
SENT="$RUN_DIR/.done"

# TICKER/TRADING_* are read by constants.py and configuration.py; exporting here is what makes the
# whole chain — preprocessing, training, ABIDES — operate on this symbol.
export TICKER TRADING_START="$START" TRADING_END="$END"

say () { echo "[$(date +%T)] $*"; }
done_phase () { [[ -f "${SENT}_$1" ]]; }
mark_done  () { touch "${SENT}_$1"; }

# ---------------------------------------------------------------------------
# Trading-day discovery. Never hardcode a day list for a new ticker.
# ---------------------------------------------------------------------------
# Read-only resolution: the canonical dir if it exists, else the raw level-suffixed one. Lets
# --dry-run report days before `setup` has created the symlink.
resolved_dir () {
  if [[ -d "$DATA_DIR" ]]; then echo "$DATA_DIR"; return; fi
  ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1
}

day_list () {
  local d; d=$(resolved_dir); [[ -n "$d" ]] || return 0
  ls "$d" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u
}

# Triage subset: an EVENLY SPREAD sample across the period, always including the two held-out test
# days (the last two — SPLIT_RATES is .85/.05/.10 applied chronologically by day).
#
# Deliberately NOT ranked by "difficulty". Raw LOBSTER event count does not predict simulation
# difficulty: on INTC, 2015-01-06 has ~1.27M message lines against 2015-01-07's ~0.85M, yet it
# simulated in 414s against 1730s. Wall-clock tracks orders the MODEL generates, not events the real
# day contained. INTC's "hardest days" (0107, 0129) were identified empirically after the fact —
# which is precisely what triage discovers, so it cannot be an input to it. A spread is the honest
# choice for a ticker with no prior.
triage_days () {
  local n="${1:-$TRIAGE_DAYS}"
  # portable read loop rather than mapfile — macOS ships bash 3.2, which lacks it
  local -a all=(); local line
  while IFS= read -r line; do [[ -n "$line" ]] && all+=("$line"); done < <(day_list)
  local total=${#all[@]}
  [[ "$total" -gt 0 ]] || return 0
  if [[ "$total" -le "$n" ]]; then printf '%s\n' "${all[@]//-/}"; return; fi
  {
    printf '%s\n' "${all[$((total-2))]}" "${all[$((total-1))]}"   # the two held-out test days
    local step=$(( total / (n - 1) )); [[ "$step" -lt 1 ]] && step=1
    for ((i=0; i<total-2 && $(( i / step )) < n-2; i+=step)); do printf '%s\n' "${all[$i]}"; done
  } | tr -d '-' | sort -u | head -"$n"
}

# ---------------------------------------------------------------------------
if [[ "$DRY" == "1" ]]; then
  echo "=== plan: $TICKER  $START .. $END ==="
  echo "data dir : $DATA_DIR  ($([[ -d "$DATA_DIR" ]] && echo present || echo MISSING))"
  echo "run dir  : $RUN_DIR"
  N=$(day_list | wc -l | tr -d ' ')
  echo "days     : ${N:-0}"
  [[ "${N:-0}" -gt 0 ]] && echo "triage   : $(triage_days | tr "\n" " ")"
  cat <<EOF

phase           does                                                        est.
setup           validate layout, flag files, quantile targets               minutes
train           KEEP_EPOCH_CHECKPOINTS, LIMIT_TRAIN_BATCHES=$LIMIT_TRAIN_BATCHES               12-24 h
triage          adaptive_ckpt_search over ALL ckpts, $TRIAGE_DAYS-day spread          4-8 h
sigma           exec_bracket per survivor, sigmas: $SIGMAS            ~1.5 h
stability       adaptive_ckpt_search, full period, survivors only           ~7 h/ckpt
ss-retrain      scheduled sampling from best survivor (fallback: first)     12-24 h
evaluate        LOB-Bench + stylised facts + price path                     ~2 h
trades-default  vanilla DDPM-100 on the 2 held-out test days   $([[ "$WITH_TRADES_DEFAULT" == 1 ]] && echo "ENABLED" || echo "opt-in, off")

Estimates are INTC-calibrated. Wall-clock tracks event count, which varies ~4.5x across days
within a month and much more across tickers — re-derive from the smoke test.
EOF
  exit 0
fi

[[ -n "$PHASE" ]] || { echo "!! --phase required (or --dry-run). See --dry-run for the list."; exit 1; }

case "$PHASE" in

# ---------------------------------------------------------------------------
setup)
  done_phase setup && { say "setup already done"; exit 0; }
  say "resolving data layout for $TICKER"

  # ABIDES needs the un-suffixed directory. LOBSTER ships a level-suffixed one; link if that is what
  # is present. Unzip first if only an archive is there — the built-in auto-unzip in world_agent_sim
  # is guarded by `symbol == "INTC"` and will not fire for any other ticker.
  if [[ ! -d "$DATA_DIR" ]]; then
    RAW=$(ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1)
    if [[ -z "$RAW" ]]; then
      ZIP=$(ls "data/${TICKER}/${TICKER}_${START}_${END}"*.zip 2>/dev/null | head -1)
      if [[ -n "$ZIP" ]]; then
        say "unzipping $ZIP"
        unzip -q "$ZIP" -d "data/${TICKER}" || { echo "!! unzip failed"; exit 1; }
        RAW=$(ls -d "data/${TICKER}/${TICKER}_${START}_${END}"_* 2>/dev/null | grep -v '\.zip$' | head -1)
      fi
    fi
    if [[ -n "$RAW" ]]; then
      say "linking $(basename "$RAW") -> $(basename "$DATA_DIR")  (ABIDES expects no level suffix)"
      ln -sfn "$(basename "$RAW")" "$DATA_DIR"
    fi
  fi

  [[ -d "$DATA_DIR" ]] || { cat <<EOF
!! data not found: $DATA_DIR

ABIDES builds this exact path and calls os.listdir on it, so it must resolve. Provide any of:
  data/${TICKER}/${TICKER}_${START}_${END}/         (canonical — used directly)
  data/${TICKER}/${TICKER}_${START}_${END}_10/      (raw LOBSTER — this script symlinks it)
  data/${TICKER}/${TICKER}_${START}_${END}_10.zip   (this script unzips then symlinks)

containing one message+orderbook pair per trading day, e.g.
  ${TICKER}_${START}_34200000_57600000_message_10.csv
  ${TICKER}_${START}_34200000_57600000_orderbook_10.csv
EOF
  exit 1; }

  NM=$(ls "$DATA_DIR"/*message*.csv 2>/dev/null | wc -l | tr -d ' ')
  NO=$(ls "$DATA_DIR"/*orderbook*.csv 2>/dev/null | wc -l | tr -d ' ')
  say "message files: $NM   orderbook files: $NO"
  [[ "$NM" -gt 0 && "$NM" -eq "$NO" ]] || { echo "!! need equal, non-zero message/orderbook counts"; exit 1; }

  # Both data-pipeline corrections on, matching the adopted INTC configuration. File-gated: these
  # appear in no command line and no output path, so they must be set explicitly every session.
  touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
  PRE=$("$PY" -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
  [[ "$PRE" == "True True" ]] || { echo "!! flags not True True (got: $PRE)"; exit 1; }

  say "config resolves to:"
  "$PY" -c "
import constants as c, configuration
print('   stock :', configuration.Configuration().CHOSEN_STOCK)
print('   period:', c.DATE_TRADING_DAYS)" || exit 1

  say "building quantile targets (per-ticker size/depth marginals)"
  "$PY" scripts/build_quantile_targets.py --stock "$TICKER" --day-dir "$DATA_DIR" \
      --out-dir data/quantile_targets > "$RUN_DIR/logs/quantile_targets.txt" 2>&1 \
      || { echo "!! quantile targets failed — see $RUN_DIR/logs/quantile_targets.txt"; exit 1; }

  day_list > "$RUN_DIR/days.txt"
  triage_days > "$RUN_DIR/triage_days.txt"
  say "days: $(wc -l < "$RUN_DIR/days.txt" | tr -d ' ')   triage: $(tr '\n' ' ' < "$RUN_DIR/triage_days.txt")"
  mark_done setup; say "setup complete"
  ;;

# ---------------------------------------------------------------------------
train)
  done_phase train && { say "train already done"; exit 0; }
  done_phase setup || { echo "!! run --phase setup first"; exit 1; }
  pgrep -f "main.py" >/dev/null && { echo "!! main.py already running"; exit 1; }

  # One checkpoint per epoch, never deleted, so the search can trial them behaviourally rather than
  # trusting val loss. Short epochs concentrate checkpoints where the INTC search found the useful
  # ones (epochs 0-4).
  touch KEEP_EPOCH_CHECKPOINTS_FLAG
  rm -f SCHEDULED_SAMPLING_FLAG RESUME_TRAINING_FLAG   # a fresh lineage, not a continuation
  export LIMIT_TRAIN_BATCHES="$LIMIT_TRAIN_BATCHES"

  LOG="$RUN_DIR/logs/train.log"
  say "training $TICKER (LIMIT_TRAIN_BATCHES=$LIMIT_TRAIN_BATCHES) -> $LOG"
  nohup "$PY" -u main.py > "$LOG" 2>&1 &
  say "pid $! — watch: tail -f $LOG"
  say "when it finishes, mark it: touch ${SENT}_train"
  ;;

# ---------------------------------------------------------------------------
triage)
  done_phase triage && { say "triage already done"; exit 0; }
  DAYS=$(tr '\n' ' ' < "$RUN_DIR/triage_days.txt" 2>/dev/null)
  [[ -n "$DAYS" ]] || { echo "!! no triage days — run --phase setup"; exit 1; }
  say "triage over ALL checkpoints, $TRIAGE_DAYS-day spread: $DAYS"
  say "abandon-on-first-failure: a checkpoint that fails ANY day is not the generalising one"
  bash scripts/adaptive_ckpt_search.sh --ticker "$TICKER" --st "$ST" --et "$ET" --seed "$SEED" \
      --days "$DAYS" --cap-secs "$CAP_SECS" \
      --out-tag "${TICKER}_triage" 2>&1 | tee "$RUN_DIR/logs/triage.log"
  mark_done triage
  say "survivors -> ckpt_search/latest-${TICKER}_triage/progress.txt"
  ;;

# ---------------------------------------------------------------------------
sigma)
  done_phase sigma && { say "sigma already done"; exit 0; }
  HARD=$(head -1 "$RUN_DIR/triage_days.txt" 2>/dev/null)
  [[ -n "$HARD" ]] || { echo "!! no day available — run --phase setup"; exit 1; }
  say "bracketing depth-noise sigma on $TICKER $HARD; candidates: $SIGMAS"
  say "target is THIS ticker's real execution share, printed as the REAL row in each cell"
  bash scripts/exec_bracket.sh --ticker "$TICKER" --date "$HARD" --st "$ST" --et "$ET" \
      --seed "$SEED" --sigmas "$SIGMAS" --out-dir "$RUN_DIR/sigma" 2>&1 | tee "$RUN_DIR/logs/sigma.log"
  cat <<EOF

Read $RUN_DIR/sigma/summary.md and pick the sigma whose executed % is closest to the REAL row.
Then re-run stability with it:
  BASE="--depth-noise <sigma> --size-reshape --type-decode prior"
EOF
  mark_done sigma
  ;;

# ---------------------------------------------------------------------------
stability)
  done_phase stability && { say "stability already done"; exit 0; }
  DAYS=$(tr '\n' ' ' < "$RUN_DIR/days.txt" 2>/dev/null)
  [[ -n "$DAYS" ]] || { echo "!! no day list — run --phase setup"; exit 1; }
  say "full-period stability, $(wc -w <<< "$DAYS") days, survivors only"
  say "sigma: $DEPTH_NOISE  (set with --depth-noise from the sigma phase; 0.3 is INTC's, not portable)"
  bash scripts/adaptive_ckpt_search.sh --ticker "$TICKER" --st "$ST" --et "$ET" --seed "$SEED" \
      --depth-noise "$DEPTH_NOISE" --days "$DAYS" --cap-secs "$CAP_SECS" \
      --out-tag "${TICKER}_stability" --no-abandon 2>&1 | tee "$RUN_DIR/logs/stability.log"
  mark_done stability
  ;;

# ---------------------------------------------------------------------------
ss-retrain)
  done_phase ss-retrain && { say "ss-retrain already done"; exit 0; }
  pgrep -f "main.py" >/dev/null && { echo "!! main.py already running"; exit 1; }

  # Per the brief: retrain from the best survivor; if nothing cleared, fall back to the FIRST
  # checkpoint and re-enter stability afterwards. RESUME_TRAINING_FLAG makes Lightning resume from
  # the newest checkpoint in the dir (weights + optimizer + epoch), so the intended parent must be
  # the newest file there — archive others first if that is not already true.
  PROG="ckpt_search/latest-${TICKER}_stability/progress.txt"
  if [[ -f "$PROG" ]] && grep -q "WINNER" "$PROG"; then
    say "parent: stability winner — $(grep 'WINNER' "$PROG" | tail -1)"
  else
    say "no checkpoint cleared the period. FALLBACK: retraining from the first checkpoint,"
    say "then re-run --phase stability to re-test the SS lineage."
  fi

  touch SCHEDULED_SAMPLING_FLAG RESUME_TRAINING_FLAG KEEP_EPOCH_CHECKPOINTS_FLAG
  export LIMIT_TRAIN_BATCHES="$LIMIT_TRAIN_BATCHES"
  LOG="$RUN_DIR/logs/ss_retrain.log"
  say "scheduled-sampling retrain -> $LOG"
  nohup "$PY" -u main.py > "$LOG" 2>&1 &
  say "pid $! — when it finishes: touch ${SENT}_ss-retrain, then --phase stability, then --phase evaluate"
  ;;

# ---------------------------------------------------------------------------
evaluate)
  done_phase evaluate && { say "evaluate already done"; exit 0; }
  say "evaluation runs LOCALLY — LOB-Bench needs jax (py>=3.10), which the GPU box does not have."
  cat <<EOF

Pull the generated CSVs, then locally:

  1. Real references straight from the LOBSTER files (NOT the ABIDES replay — they differ slightly):
       python -m evaluation.stylized_custom.lobster_real_reference \\
              --stock $TICKER --day-dir $DATA_DIR --date <YYYYMMDD> --st $ST --et $ET

  2. LOB-Bench across the period:
       bash scripts/lob_bench_multiday.sh   # set the ticker/day list inside

  3. Stylised facts (minute bars for lag panels, 1s for the mid-price trace):
       python -m evaluation.stylized_custom.paper_style_stylized_facts <real.csv> <gen.csv> <out.png>

REPORT THE SPREAD DECOMPOSITION alongside any LOB-Bench gain. On INTC 72% of the scheduled-sampling
improvement was bid-ask spread alone (0.719 -> 0.192); excluding spread the other five metrics moved
0.418 -> 0.376. Whether that ratio holds on a second ticker is the more interesting result either
way, and a reader will compute it if you do not.

Quote five-metric and six-metric means separately — they are not comparable.
EOF
  mark_done evaluate
  ;;

# ---------------------------------------------------------------------------
trades-default)
  [[ "$WITH_TRADES_DEFAULT" == "1" ]] || {
    echo "!! opt-in phase. Re-run with --with-trades-default."; exit 1; }
  done_phase trades-default && { say "trades-default already done"; exit 0; }
  # The last two trading days are the held-out test set (SPLIT_RATES .85/.05/.10 by day).
  TEST_DAYS=$(tail -2 "$RUN_DIR/days.txt" | tr -d '-' | tr '\n' ' ')
  say "vanilla DDPM-100 on held-out test days: $TEST_DAYS"
  say "NO decode corrections — this is the published pipeline, not ours"
  for D in $TEST_DAYS; do
    say "-- $D"
    "$PY" -u ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ET" \
        -d True -m TRADES -type DDPM -nsteps 100 -eta 0.0 -seed "$SEED" \
        > "$RUN_DIR/logs/trades_default_${D}.txt" 2>&1 || say "   FAILED $D"
  done
  mark_done trades-default
  ;;

*) echo "!! unknown phase: $PHASE (see --dry-run)"; exit 1;;
esac
