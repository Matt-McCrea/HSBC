#!/usr/bin/env bash
# Usage: bash make_plots.sh <log_dir>
#   <log_dir> = folder name under ABIDES/log/  (or a full path)
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash make_plots.sh <log_dir>"
  echo "  e.g. bash make_plots.sh world_agent_INTC_2012-06-21_12-00-00_30_DDIM_0.0_1_val_ema=-1.05"
  exit 1
fi

# repo root = dir containing ABIDES/  (assumes you run from repo root)
REPO="$(pwd)"
[ -d "$REPO/ABIDES" ] || { echo "ERROR: run this from the DeepMarket repo root (no ABIDES/ here)."; exit 1; }

# resolve log dir: accept a full path OR a bare folder name under ABIDES/log/
ARG="$1"
if [ -d "$ARG" ]; then
  LOG="$(cd "$ARG" && pwd)"
elif [ -d "$REPO/ABIDES/log/$ARG" ]; then
  LOG="$REPO/ABIDES/log/$ARG"
else
  echo "ERROR: log dir not found. Looked at:"; echo "  $ARG"; echo "  $REPO/ABIDES/log/$ARG"; exit 1
fi
echo ">> Log dir: $LOG"

# find the exchange order-stream bz2 (sim writes EXCHANGE_AGENT.bz2)
BZ2=""
for c in "EXCHANGE_AGENT.bz2" "ExchangeAgent.bz2"; do
  [ -f "$LOG/$c" ] && { BZ2="$LOG  [ -f "$LOG/$c" ne
  [ -f "$LOG/$c" ] && { BZ2="$LOG/*EXC  [ -f "$LOG/$c" ] && { BZ2="$LOG/*EXC  [ -f "$LOG/$c" ] && { -f  [ -f "$LOG/$c"cho "ERROR: no exchange .bz2 in $LOG"; ls -la "$LOG"; exit 1; }
echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "echo "ER>_<DATE>_...)
TICKER="$(basename "$LOG" | cut -d'_' -f3)"
echo ">> Ticker: $TICKER"

# 1) convert bz2 -> orders_<TICKER>_<YYYYMMDD>.pkl  (plot-scripts format)
PKL_DIR="$LOG/pkl"; mkdir -p "$PKL_DIR"
echo ">> Converting -> $PKL_DIR"
( cd "$REPO" && PYTHONPATH=ABIDES python3 ABIDES/util/formatting/convert_order_stream.py \
    "$BZ2" "$TICKER" 10 plot-scripts -o "$PKL_DIR" )
PKL="$(ls "$PKL_DIR"/orders_*.pkl 2>/dev/null | head -1 || true)"
[ -n "$PKL" ] && [ -s "$PKL" ] || { echo "ERROR: no non-empty orders_*.pkl produced"; ls -la "$PKL_DIR"; exit 1; }
echo ">> Built: $PKL ($(du -h "$PKL" | cut -f1))"

# 2) stylized-facts plots against that pkl dir  (-z bypasses any stale cache)
VIS="$REPO/visualizations"; mkdir -p "$VIS"
echo ">> Plotting -> $VIS"
( cd "$REPO/ABIDES" && PYTHONPATH=. python3 realism/order_flow_stylized_facts.py "$PKL_DIR" -o "$VIS" -z )

echo ">> Done. Output:"; ls -la "$VIS"
