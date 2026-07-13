#!/bin/bash
# unclamp_retrain.sh — retrain TRADES with SIGNED depth (marketable orders kept as depth<0).
#
# THE ROOT CAUSE we're attacking: training clamps depth≥0 (utils_data.py:324), so a marketable,
# spread-crossing order (depth<0) is NEVER a training target. The model can only emit aggression as
# sampling *noise* — which is exactly why few-step deterministic samplers (which kill that noise)
# freeze the mid-price. Unclamping lets the model LEARN to cross the spread, so aggression becomes a
# signal it can reproduce deterministically.
#
# LAST ATTEMPT FAILED SILENTLY: `export UNCLAMP_DEPTH=1` in an interactive shell, then `python main.py`
# ran in a process that never inherited it (background/job-scheduler/new-session env loss — a classic
# gotcha). LOBSTERDataBuilder unconditionally OVERWRITES normalization_stats.json every run (no
# caching) — so the only way the file stayed at the old clamped values (mean_depth≈1.38, std≈2.68) is
# that preprocessing ran WITHOUT UNCLAMP_DEPTH=1 in ITS OWN process environment. Training then ran for
# hours on ordinary clamped data.
#
# FIX: never `export` in one place and rely on it reaching another. The var is now INLINED directly on
# the same line as `python`, which guarantees it reaches that exact process regardless of shell/session/
# backgrounding. Also self-verifying: launches backgrounded (nohup, survives an SSH disconnect — but
# NOT past your booked walltime), and gives you a `--check` you can run ~1-2 min later (preprocessing
# finishes fast, well before the GPU training loop starts) to confirm it actually took before you walk
# away and lose another session to it.
#
# IMPORTANT: you must ALSO inline UNCLAMP_DEPTH=1 at SIMULATION/eval time later — same reasoning, same
# fix. Use `UNCLAMP_DEPTH=1 bash scripts/open_loop_sweep.sh ...` (not `export` then a separate call).
#
# Usage:
#   bash scripts/unclamp_retrain.sh            # backup clamped data, launch reprocess+train (background)
#   bash scripts/unclamp_retrain.sh --check    # run ~1-2 min after launch: did the unclamp take effect?
#   bash scripts/unclamp_retrain.sh --restore  # restore the clamped baseline data and exit

set -uo pipefail
STOCK="INTC"; DATA_DIR="data/${STOCK}"; BK="${DATA_DIR}/_clamped_backup"
STATS="${DATA_DIR}/normalization_stats.json"
LOG="unclamp_train.log"; PIDFILE="unclamp_train.pid"

if [[ "${1:-}" == "--restore" ]]; then
  [[ -d "$BK" ]] || { echo "no backup at $BK"; exit 1; }
  cp -v "$BK"/*.npy "$BK"/*.json "$DATA_DIR"/ 2>/dev/null
  echo "restored clamped baseline."; exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "── current normalization_stats.json (event depth) ──"
  if [[ -f "$STATS" ]]; then
    grep -A1 '"mean_depth"\|"std_depth"' "$STATS"
  else
    echo "STATS FILE NOT FOUND — preprocessing hasn't written it yet (still running, or crashed)."
  fi
  echo ""
  echo "OLD clamped baseline was:  mean_depth ≈ 1.38   std_depth ≈ 2.68"
  echo "EXPECT now:                mean_depth WELL BELOW 1.38, std_depth ABOVE 2.68"
  echo ""
  echo "If it still reads ~1.38 / ~2.68 → the unclamp did NOT take. Kill it now, don't burn the"
  echo "session on a wasted run:   kill \$(cat $PIDFILE) 2>/dev/null"
  echo ""
  echo "── training log tail ──"
  tail -20 "$LOG" 2>/dev/null || echo "(no log yet)"
  exit 0
fi

echo "UNCLAMP_DEPTH retrain — bulletproof relaunch"

# 1. Config sanity — must be False or preprocessing (and the unclamp) is skipped entirely, silently.
if grep -qE '^\s*self\.IS_DATA_PREPROCESSED\s*=\s*True' configuration.py; then
  echo "!! configuration.py has IS_DATA_PREPROCESSED = True — set it False so preprocessing re-runs"
  echo "   with unclamped depth, then re-run this script."; exit 1
fi

# 2. Show the OLD stats (if any) so --check has an obvious before/after.
if [[ -f "$STATS" ]]; then
  echo "current (pre-run) depth stats:"
  grep -A1 '"mean_depth"\|"std_depth"' "$STATS"
fi

# 3. One-time backup of the clamped baseline, so --restore always works.
if [[ ! -d "$BK" ]]; then
  mkdir -p "$BK"
  cp -v "$DATA_DIR"/*.npy "$BK"/ 2>/dev/null || true
  cp -v "$STATS" "$BK"/ 2>/dev/null || true
  echo "backed up clamped baseline → $BK"
else
  echo "backup already exists at $BK (not overwriting)"
fi

# 4. Launch — env var INLINED on the python invocation (not `export`), so it is guaranteed to reach
#    THIS process regardless of shell/session/job-scheduler boundaries. Backgrounded with nohup so an
#    accidental disconnect doesn't kill it (it will still die when your booked walltime itself ends).
rm -f "$LOG"
nohup env UNCLAMP_DEPTH=1 python main.py > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "launched: PID $PID   log: $LOG"
echo ""
echo "══════════════════════════════════════════"
echo "WAIT ~1-2 MIN (preprocessing finishes fast, before the GPU training loop starts), THEN RUN:"
echo "    bash scripts/unclamp_retrain.sh --check"
echo "Confirm mean_depth has dropped well below 1.38 BEFORE you walk away from this session."
echo "══════════════════════════════════════════"
