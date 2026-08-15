#!/bin/bash
# measure_ddim1.sh — measure single-step DDIM's marketable-order rate (the number for the paper).
#
# RUN WHEN COMPUTE IS NON-VITAL: needs a free GPU for ~one 30-min-session sim. Deprioritise it
# behind training and real experiments — GPU contention turned a 45-min sim into 3.5h before, and
# it refuses to start while main.py is running.
#
# PURPOSE: replace the paragraph's theoretical ~24% with a MEASURED marketable fraction. Single-step
# DDIM's net output is ~1.7% model + ~98.3% Gaussian prior x_1 (cosine schedule: a_bar_1=0.9983,
# a_bar_0=0.9994; the directional term +0.025*eps cancels most of pred_x0's -0.042, leaving -0.017
# net). So the decoded depth channel stays ~N(0,1) and a fraction of orders decode marketable
# (depth<0) by chance. This logs that fraction from the depth_pre_drop diagnostic.
#
# Runs with the CURRENT checkpoint + normalization_stats.json + flag files exactly as they stand —
# the same internally-consistent config as your other sims (no train/sim mismatch). The ONLY change
# vs a normal run is `-type DDIM -nsteps 1`. Pass --id to pin a specific checkpoint (e.g. a clamped
# baseline as the paper's original-TRADES analog — but then ensure its matching
# normalization_stats.json is the one on disk, or the numbers are meaningless).

set -uo pipefail
ID=""; ST="09:30:00"; ET="10:00:00"; TICKER="INTC"; DATE="20150130"
OUT="measure_ddim1/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --start) ST="$2"; shift 2;; --end) ET="$2"; shift 2;;
  --out-dir) OUT="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT"; LOG="$OUT/ddim1.log"

if pgrep -f "main.py" >/dev/null; then
  echo "!! training (main.py) is running — this is a non-vital measurement, let it finish first"
  echo "   (GPU contention turned a 45-min sim into 3.5h before). Refusing."; exit 1
fi

echo "flags: UNCLAMP=$([[ -f UNCLAMP_DEPTH_FLAG ]] && echo ON || echo OFF)  "\
"REANCHOR=$([[ -f PRICE_REANCHOR_FLAG ]] && echo ON || echo OFF)   (using current state as-is)"
A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET"
   -d True -m TRADES -type DDIM -nsteps 1 -eta 0.0)
[[ -n "$ID" ]] && A+=(-id "$ID")
echo "running: ${A[*]}"
if ! "${A[@]}" > "$LOG" 2>&1; then echo "run FAILED — see $LOG"; tail -20 "$LOG"; exit 1; fi

echo ""
grep "checkpoint used" "$LOG" | tail -1
python3 - "$LOG" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'depth_pre_drop: neg=(\d+) 0=(\d+) 1-2=(\d+) 3-5=(\d+) 6\+=(\d+)', t)
d = re.search(r'decoded_pre_drop: limit=(\d+) cancel=(\d+) market=(\d+)', t)
b = re.search(r'B_crossing_limit=(\d+)', t)
print("\n=== SINGLE-STEP DDIM (nsteps=1) MEASURED RATES ===")
if m:
    neg, z, a, c, e = map(int, m.groups()); tot = neg + z + a + c + e
    print(f"  depth<0 (MARKETABLE) : {neg}/{tot} = {100*neg/max(tot,1):.2f}%"
          f"   <- the number for the paragraph  (theory upper bound if pure N(0,1): 24%)")
else:
    print("  depth_pre_drop line not found — did the run reach the diagnostics block?")
if d:
    lim, can, mkt = map(int, d.groups()); tt = lim + can + mkt
    print(f"  market-type decode  : {mkt}/{tt} = {100*mkt/max(tt,1):.2f}%   (real ~2.8%)")
if b:
    print(f"  B_crossing_limit    : {b.group(1)}   (executions via crossing limits)")
PY
echo ""
echo "full DIAG block:"
sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
