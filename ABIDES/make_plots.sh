#!/usr/bin/env bash
# Usage: bash ABIDES/make_plots.sh <log_dir>
#   <log_dir> = folder name under ABIDES/log/  (or a full path)
# Generates BOTH order-flow and asset-return stylized-fact plots.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash ABIDES/make_plots.sh <log_dir>"
  echo "  e.g. bash ABIDES/make_plots.sh world_agent_INTC_2012-06-22_12-00-00_30_DDIM_0.0_1_val_ema=0.7_e"
  exit 1
fi

# repo root = dir containing ABIDES/  (run this from repo root)
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

# find the exchange order-stream bz2
BZ2=""
for c in "EXCHANGE_AGENT.bz2" "ExchangeAgent.bz2"; do
  [ -f "$LOG/$c" ] && { BZ2="$LOG/$c"; break; }
done
[ -n "$BZ2" ] || { echo "ERROR: no exchange .bz2 in $LOG"; ls -la "$LOG"; exit 1; }
echo ">> Found exchange log: $BZ2"

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

VIS="$REPO/visualizations"; mkdir -p "$VIS"
echo ">> Plotting -> $VIS"

# 2) order-flow stylized facts  (consumes the PKL DIR)
echo ">> [1/2] order-flow stylized facts..."
( cd "$REPO/ABIDES" && PYTHONPATH=. python3 realism/order_flow_stylized_facts.py "$PKL_DIR" -o "$VIS" -z )

# 3) asset-return stylized facts  (consumes the LOG DIR with the .bz2, NOT the pkl)
echo ">> [2/2] asset-return stylized facts..."
( cd "$REPO/ABIDES" && PYTHONPATH=. python3 realism/asset_returns_stylized_facts.py -s "$LOG" -o "$VIS" -z )

echo ">> Done. Output:"; ls -la "$VIS"
