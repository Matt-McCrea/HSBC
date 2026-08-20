#!/bin/bash
# exec_bracket.sh — pin the execution rate to real (~7%) on the reanchored checkpoint.
#
# WHY: the sweep bracketed exec ABOVE real everywhere (sigma 0.2 -> 9.6%, 0.3 -> 12.2%). The
# marketable fraction scales with the decode-time depth-noise sigma, so lower sigma lowers exec.
# This runs three cells BELOW 0.2 to bracket and pin exec ~= real 7.0% on the 30-min window.
# Everything else is the sweep's headline config (DDIM10, --size-reshape, --type-decode prior).
#
# Runs on the SAME checkpoint the sweep auto-discovered (val_ema=0.627 by default; --id to pin).
# Both flag files (UNCLAMP + REANCHOR) must be ON — the reanchored ckpt was trained with both.
#
# TICKER-PORTABLE: --ticker/--date/--st/--et/--seed drive the run and the matched real replay path.
# The sigma that pins execution to real is a property of the STOCK (tick size, price level, typical
# depth), not of the method — INTC's 0.3 must not be carried across. Re-bracket per ticker.
#
# Usage:  bash scripts/exec_bracket.sh                 # auto-discover newest ckpt (INTC defaults)
#         bash scripts/exec_bracket.sh --id 0.627      # pin a checkpoint by val_ema
#         bash scripts/exec_bracket.sh --ticker MSFT --date 20150331 \
#              --ckpt-path data/checkpoints/TRADES/<file>.ckpt --sigmas "0.2 0.3 0.4"

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"; SEED="30"
CKPT_DIR="data/checkpoints/TRADES"; ID=""; CKPT_PATH=""
SIGMAS="0.10 0.125 0.15"                         # override with --sigmas "0.16 0.17 0.18"
REAL=""                                          # derived from ticker/date/window below
OUT_DIR="exec_bracket/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --sigmas) SIGMAS="$2"; shift 2;;
  --ticker) TICKER="$2"; shift 2;;               # any symbol; the sigma that pins exec to real is
  --date) DATE="$2"; shift 2;;                   #   ticker-specific, so never reuse INTC's 0.3
  --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;
  --ckpt-path) CKPT_PATH="$2"; shift 2;;         # exact file, bypasses -id val_ema matching
  --real) REAL="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# Real replay path follows market_replay_{TICKER}_{YYYY-MM-DD}_{ET with : -> -}_{SEED}.
if [[ -z "$REAL" ]]; then
  _D="${DATE:0:4}-${DATE:4:2}-${DATE:6:2}"
  REAL="ABIDES/log/market_replay_${TICKER}_${_D}_${ET//:/-}_${SEED}/processed_orders.csv"
fi
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

if pgrep -f "main.py" > /dev/null; then
  echo "!! training (main.py) running — kill it first (GPU contention). Refusing."; exit 1
fi

touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
echo "flags -> UNCLAMP_DEPTH PRICE_REANCHOR = $PRECHECK"
[[ "$PRECHECK" == "True True" ]] || { echo "!! flags not True True — refusing. Got: $PRECHECK"; exit 1; }

valof () { basename "$1" | sed -E 's/^[^=]*=([0-9.]+).*/\1/'; }
if [[ -n "$CKPT_PATH" ]]; then
  [[ -f "$CKPT_PATH" ]] || { echo "!! --ckpt-path not found: $CKPT_PATH"; exit 1; }
  echo "using exact checkpoint: $(basename "$CKPT_PATH")"
elif [[ -z "$ID" ]]; then
  NEWEST=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -1)
  [[ -n "$NEWEST" ]] || { echo "!! no .ckpt in $CKPT_DIR"; exit 1; }
  ID=$(valof "$NEWEST"); echo "auto-discovered: $(basename "$NEWEST") -> -id $ID"
fi
# val_ema matching cannot disambiguate two checkpoints sharing a rounded value — only relevant when
# selecting by -id, not when an exact path was given.
if [[ -z "$CKPT_PATH" ]]; then
  COLLIDE=$(for f in "$CKPT_DIR"/*.ckpt; do valof "$f"; done | grep -Fxc "$ID" || true)
  [[ "${COLLIDE:-0}" -le 1 ]] || { echo "!! $COLLIDE ckpts share val_ema=$ID — archive strays. Refusing."; exit 1; }
fi

CKPT_LABEL="${CKPT_PATH:-val_ema=$ID}"
echo "# Exec bracket — $(date '+%F %T')  $TICKER $DATE $ST-$ET  ckpt $(basename "$CKPT_LABEL")" > "$SUM"
echo "Target: this ticker's OWN real execution share on this window (read it off the REAL row below" >> "$SUM"
echo "in each cell). Do NOT reuse INTC's 7.0% or its sigma=0.3 — both are ticker-specific." >> "$SUM"

run () { # run <tag> <extra>
  local TAG="$1" EXTRA="$2"
  local LOG="$OUT_DIR/logs/${TAG}.txt"
  echo "-- $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET"
           -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 -seed "$SEED")
  if [[ -n "$CKPT_PATH" ]]; then A+=(--ckpt-path "$CKPT_PATH"); else A+=(-id "$ID"); fi
  # shellcheck disable=SC2206
  A+=($EXTRA)
  echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1; echo "csv: ${CSV:-none}"
    if [[ -n "$CSV" && -f "$REAL" ]]; then python -m evaluation.quantitative_eval.flow_mix --real "$REAL" --gen "$CSV" 2>&1
    elif [[ -n "$CSV" ]]; then python -m evaluation.quantitative_eval.flow_mix --gen "$CSV" 2>&1; fi
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo '```'; echo ""; } >> "$SUM"
  echo "  done ${SECS}s"
}

# One cell per sigma in $SIGMAS (default brackets below 0.2; pass --sigmas to re-target).
for s in $SIGMAS; do
  run "DDIM10_dn${s}_sr_prior" "--depth-noise ${s} --size-reshape --type-decode prior"
done

# mini table
python3 - "$SUM" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
rows = []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    tag = block.splitlines()[0].split('(')[0].strip()
    if 'ERROR' in block.splitlines()[0]: continue
    def g(pat, d='-'):
        m = re.search(pat, block); return m.group(1) if m else d
    # last (=generated) values from the flow_mix block
    execs = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block)
    lim   = re.findall(r'LIMIT_ORDER\s+([\d.]+)', block)
    can   = re.findall(r'ORDER_CANCELLED\s+([\d.]+)', block)
    mids  = re.findall(r'unique mid-prices: (\d+)', block)
    b = g(r'B_crossing_limit=(\d+)')
    bid = re.findall(r'bid_size_1:.*mean=(\d+)', block)
    ask = re.findall(r'ask_size_1:.*mean=(\d+)', block)
    rows.append((tag, lim[-1] if lim else '-', can[-1] if can else '-',
                 execs[-1] if execs else '-', mids[-1] if mids else '-',
                 b, bid[-1] if bid else '-', ask[-1] if ask else '-'))
hdr = f"{'cell':<26}{'Lim%':>6}{'Can%':>6}{'Exec%':>7}{'mids':>6}{'B':>7}{'bid1':>8}{'ask1':>8}"
line = "\n".join([hdr, '-'*len(hdr),
                  f"{'REAL':<26}{'49.2':>6}{'43.8':>6}{'7.0':>7}{'69':>6}{'-':>7}{'3899':>8}{'2117':>8}"] +
                 [f"{t:<26}{l:>6}{c:>6}{e:>7}{m:>6}{b:>7}{bd:>8}{ak:>8}" for t,l,c,e,m,b,bd,ak in rows])
print("\n==== EXEC BRACKET (target exec=7.0%, mids=69) ====\n"+line)
open(sys.argv[1],'w').write("# TABLE\n```\n"+line+"\n```\n\n"+text)
PY
echo ""; echo "Done. Summary: $SUM"
echo "READ: which sigma lands exec closest to 7.0% — that's your headline 30-min config."
