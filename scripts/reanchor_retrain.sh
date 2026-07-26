#!/bin/bash
# reanchor_retrain.sh — retrain TRADES with UNCLAMPED depth + DAY-ANCHORED prices.
#
# THE BOUNDARY we're removing: TRADES z-scores ABSOLUTE prices against a global training mean
# (mean_price≈3620=$36.21, σ≈67=$0.67), so the whole Jan-30 test day sits at −3.4..−3.9σ, and
# BOTH 75-min runs (fixed σ and σ-controlled) degenerated at the SAME threshold: the moment the
# genuinely-real intraday decline pushed the mid through $33.50 = z≈−4.0 — with a HEALTHY book in
# the controlled run, so it's the model breaking (time channel explodes, event rate −35x), not
# liquidity. PRICE_REANCHOR subtracts each day's opening mid from every price before
# normalization (training + sim conditioning, shared helper), turning the price channel into a
# bounded intraday deviation and removing the OOD cliff entirely.
#
# Keeps UNCLAMP_DEPTH on too — the new model is trained with BOTH corrections, and eval must run
# with BOTH flag files present (they persist until you remove them).
#
# WALLTIME WARNING: the last retrain took ~1.7h/epoch, ~22h to epoch 13 (where it got good).
# Checkpoints save every epoch, so a shorter session still yields usable intermediates (epoch
# 5-8 was already informative last time) — but plan for a long session or resume across sessions
# by just re-running main.py is NOT supported (no resume logic); one long session is best.
#
# Usage:
#   bash scripts/reanchor_retrain.sh            # pre-flight, backup, launch (background)
#   bash scripts/reanchor_retrain.sh --check    # ~2 min later: did BOTH corrections take?
#   bash scripts/reanchor_retrain.sh --restore  # remove reanchor flag + restore backed-up data

set -uo pipefail
STOCK="INTC"; DATA_DIR="data/${STOCK}"; BK="${DATA_DIR}/_preanchor_backup"
STATS="${DATA_DIR}/normalization_stats.json"
LOG="reanchor_train.log"; PIDFILE="reanchor_train.pid"

if [[ "${1:-}" == "--restore" ]]; then
  rm -f PRICE_REANCHOR_FLAG
  [[ -d "$BK" ]] || { echo "no backup at $BK (flag removed anyway)"; exit 1; }
  cp -v "$BK"/*.npy "$BK"/*.json "$DATA_DIR"/ 2>/dev/null
  echo "restored pre-anchor (unclamped) data, removed PRICE_REANCHOR_FLAG."
  echo "UNCLAMP_DEPTH_FLAG left as-is."; exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "── flags: UNCLAMP=$([[ -f UNCLAMP_DEPTH_FLAG ]] && echo ON || echo OFF)  "\
"REANCHOR=$([[ -f PRICE_REANCHOR_FLAG ]] && echo ON || echo OFF) ──"
  echo "── current normalization_stats.json (event price + depth) ──"
  grep -A1 '"mean_price"\|"mean_depth"' "$STATS" 2>/dev/null || echo "stats file missing"
  echo ""
  echo "OLD (absolute prices):  mean_price ≈ 3620   (=\$36.21, global level)"
  echo "EXPECT now:             |mean_price| < ~100  (intraday deviation from each day's open)"
  echo "AND mean_depth should stay ≈1.38 with the signed-depth negatives included (unclamp kept)."
  echo ""
  echo "If mean_price still reads ~3620 → the anchor did NOT take. Kill and investigate:"
  echo "    kill \$(cat $PIDFILE) 2>/dev/null"
  echo ""
  echo "── diagnostic lines from the training log ──"
  grep -E "UNCLAMP_DEPTH|PRICE_REANCHOR|day anchor" "$LOG" 2>/dev/null | head -8 || echo "(none yet)"
  echo "── log tail ──"
  tail -12 "$LOG" 2>/dev/null || echo "(no log yet)"
  exit 0
fi

echo "PRICE_REANCHOR + UNCLAMP_DEPTH retrain"
if pgrep -f "main.py" > /dev/null; then
  echo "!! a main.py process is already running — kill it first. Refusing."; exit 1
fi

# 1. Set BOTH flags and pre-flight the mechanism end-to-end (<2s, catches broken paths before
#    burning a 22h session — the lesson of the env-var nights).
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
echo "  pre-flight: UNCLAMP_DEPTH PRICE_REANCHOR = $PRECHECK"
if [[ "$PRECHECK" != "True True" ]]; then
  echo "!! PRE-FLIGHT FAILED — flags not read as True True. Do NOT launch. Output: $PRECHECK"
  exit 1
fi

# 2. Config sanity — preprocessing must run for the anchoring to reach the .npy + stats.
if grep -qE '^\s*self\.IS_DATA_PREPROCESSED\s*=\s*True' configuration.py; then
  echo "!! configuration.py has IS_DATA_PREPROCESSED = True — set False and re-run."; exit 1
fi

# 3. One-time backup of the current (unclamped, absolute-price) data.
if [[ ! -d "$BK" ]]; then
  mkdir -p "$BK"
  cp -v "$DATA_DIR"/*.npy "$BK"/ 2>/dev/null || true
  cp -v "$STATS" "$BK"/ 2>/dev/null || true
  echo "backed up pre-anchor data → $BK"
else
  echo "backup already exists at $BK (not overwriting)"
fi

# 4. Launch (env vars set redundantly; the FLAG FILES are the mechanism that matters).
rm -f "$LOG"
nohup env UNCLAMP_DEPTH=1 PRICE_REANCHOR=1 python main.py > "$LOG" 2>&1 &
PID=$!; echo "$PID" > "$PIDFILE"
echo "launched: PID $PID   log: $LOG   flags: UNCLAMP_DEPTH_FLAG + PRICE_REANCHOR_FLAG (persist)"
echo ""
echo "══════════════════════════════════════════"
echo "WAIT ~2 MIN, THEN:   bash scripts/reanchor_retrain.sh --check"
echo "Confirm |mean_price| dropped from ~3620 to <~100 BEFORE walking away."
echo "Eval later runs with BOTH flags still present — nothing extra to set. The winning config:"
echo "  python ABIDES/abides.py -c world_agent_sim -t INTC -date 20150130 -st 09:30:00 -et 11:00:00 \\"
echo "    -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -id <NEW_ID> \\"
echo "    --depth-noise 0.3 --size-reshape --type-decode prior --dn-target-exec 0.045"
echo "══════════════════════════════════════════"
