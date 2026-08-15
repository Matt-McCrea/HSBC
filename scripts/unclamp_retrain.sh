#!/bin/bash
# unclamp_retrain.sh — retrain TRADES with SIGNED depth (marketable orders kept as depth<0).
#
# THE ROOT CAUSE we're attacking: training clamps depth≥0 (utils_data.py:324), so a marketable,
# spread-crossing order (depth<0) is NEVER a training target. The model can only emit aggression as
# sampling *noise* — which is exactly why few-step deterministic samplers (which kill that noise)
# freeze the mid-price. Unclamping lets the model LEARN to cross the spread, so aggression becomes a
# signal it can reproduce deterministically.
#
# TWO ATTEMPTS ALREADY FAILED SILENTLY, both env-var-based:
#   1. `export UNCLAMP_DEPTH=1` in one shell, `python main.py` ran in a process that never saw it.
#   2. Inlined `env UNCLAMP_DEPTH=1 python main.py` (this file's previous version) — STILL didn't
#      take. Two different env-var mechanisms failing the same way points at the remote's job
#      launcher (booked-GPU-session wrapper / srun / sbatch, whatever it is) not propagating
#      arbitrary env vars through to the actual python subprocess at all.
#
# FIX: stop depending on env var propagation. constants.py now ALSO checks for a FILE named
# UNCLAMP_DEPTH_FLAG in the repo root — a file on disk survives any launcher, since it doesn't
# depend on process environment at all, only on cwd (which every script here already assumes).
# This script creates/removes that file for you, and — critically — self-tests BOTH the file flag
# AND the env var end-to-end BEFORE launching anything, so a broken mechanism is caught in under 2
# seconds instead of after hours of wasted training on the wrong data (again).
#
# The flag file also means eval-time no longer needs you to remember to set anything: once it
# exists, EVERY script run from this repo root picks it up automatically, until --restore removes it.
#
# Usage:
#   bash scripts/unclamp_retrain.sh            # pre-flight check, backup, launch (background)
#   bash scripts/unclamp_retrain.sh --check    # ~1-2 min after launch: did the unclamp take effect?
#   bash scripts/unclamp_retrain.sh --restore  # remove the flag + restore the clamped baseline data

set -uo pipefail
STOCK="INTC"; DATA_DIR="data/${STOCK}"; BK="${DATA_DIR}/_clamped_backup"
STATS="${DATA_DIR}/normalization_stats.json"
LOG="unclamp_train.log"; PIDFILE="unclamp_train.pid"; FLAG="UNCLAMP_DEPTH_FLAG"

if [[ "${1:-}" == "--restore" ]]; then
  rm -f "$FLAG"
  [[ -d "$BK" ]] || { echo "no backup at $BK (flag removed anyway)"; exit 1; }
  cp -v "$BK"/*.npy "$BK"/*.json "$DATA_DIR"/ 2>/dev/null
  echo "restored clamped baseline, removed $FLAG."; exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "── flag file: $([[ -f $FLAG ]] && echo "PRESENT ($FLAG)" || echo "MISSING — unclamp is OFF") ──"
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
  echo "── training log — the [main.py]/[LOBSTERDataBuilder] diagnostic lines (should be near the top) ──"
  grep -E "UNCLAMP_DEPTH|cst.UNCLAMP_DEPTH" "$LOG" 2>/dev/null || echo "(diagnostic lines not found in log yet)"
  echo ""
  echo "── training log tail ──"
  tail -20 "$LOG" 2>/dev/null || echo "(no log yet)"
  exit 0
fi

echo "UNCLAMP_DEPTH retrain — file-flag mechanism (env vars failed twice on this remote)"
echo ""

# 1. PRE-FLIGHT SELF-TEST — under 2 seconds, catches a broken mechanism before wasting a session.
echo "── pre-flight: creating $FLAG and verifying constants.py actually reads it ──"
touch "$FLAG"
PRECHECK=$(python3 -c "import constants as cst; print('UNCLAMP_DEPTH=' + str(cst.UNCLAMP_DEPTH))" 2>&1)
echo "  $PRECHECK"
if [[ "$PRECHECK" != *"UNCLAMP_DEPTH=True"* ]]; then
  echo ""
  echo "!! PRE-FLIGHT FAILED — cst.UNCLAMP_DEPTH did not come back True even with the flag file"
  echo "   present in this directory. Do NOT launch training — this would silently repeat the same"
  echo "   failure. Likely causes: not running this from the repo root (constants.py checks the file"
  echo "   relative to cwd), or the remote's python3 isn't picking up your repo's constants.py at all"
  echo "   (check 'which python3' / that you're in the right venv). Output above: $PRECHECK"
  rm -f "$FLAG"
  exit 1
fi
echo "  pre-flight OK — the file-flag mechanism works end-to-end."
echo ""

# 2. Config sanity — must be False or preprocessing (and the unclamp) is skipped entirely, silently.
if grep -qE '^\s*self\.IS_DATA_PREPROCESSED\s*=\s*True' configuration.py; then
  echo "!! configuration.py has IS_DATA_PREPROCESSED = True — set it False so preprocessing re-runs"
  echo "   with unclamped depth, then re-run this script."; rm -f "$FLAG"; exit 1
fi

# 3. Show the OLD stats (if any) so --check has an obvious before/after.
if [[ -f "$STATS" ]]; then
  echo "current (pre-run) depth stats:"
  grep -A1 '"mean_depth"\|"std_depth"' "$STATS"
fi

# 4. One-time backup of the clamped baseline, so --restore always works.
if [[ ! -d "$BK" ]]; then
  mkdir -p "$BK"
  cp -v "$DATA_DIR"/*.npy "$BK"/ 2>/dev/null || true
  cp -v "$STATS" "$BK"/ 2>/dev/null || true
  echo "backed up clamped baseline → $BK"
else
  echo "backup already exists at $BK (not overwriting)"
fi

# 5. Launch. The FLAG FILE (already created above) is the mechanism that matters; the env var is set
#    too, redundantly, in case it happens to work here — harmless either way. Backgrounded with nohup
#    so an accidental disconnect doesn't kill it (it will still die when your booked walltime ends).
rm -f "$LOG"
nohup env UNCLAMP_DEPTH=1 python main.py > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "launched: PID $PID   log: $LOG   flag file: $FLAG (persists until --restore)"
echo ""
echo "══════════════════════════════════════════"
echo "WAIT ~1-2 MIN (preprocessing finishes fast, before the GPU training loop starts), THEN RUN:"
echo "    bash scripts/unclamp_retrain.sh --check"
echo "Confirm mean_depth has dropped well below 1.38 BEFORE you walk away from this session."
echo "══════════════════════════════════════════"
