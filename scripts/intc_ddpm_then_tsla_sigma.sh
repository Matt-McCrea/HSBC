#!/bin/bash
# intc_ddpm_then_tsla_sigma.sh — two queued phases, run in order, resumable.
#
#   PHASE 1  INTC, DDPM-100, NO decode interventions, 2h, on INTC's two held-out test days.
#            The un-accelerated TRADES-default baseline for INTC.
#
#   PHASE 2  TSLA sigma sweep with the CORRECTED type prior, derived from TSLA's own real
#            marginals rather than INTC's hardcoded 0.49/0.48/0.03.
#
# WHY PHASE 2 NEEDS THE PRIOR. Every earlier TSLA bracket held execution share at 2.7-4.2% across
# sigma 0.15-3.0 (a 20x range) while real TSLA is 16.7%, because --type-decode prior was pinning the
# market-order class to Intel's 3%. Sigma was never the binding constraint. This sweep re-runs with
# TSLA's own prior so sigma can actually be read.
#
# ⚠️ FLAG FILES DIFFER BY PHASE, and this is the thing most likely to invalidate the output:
#   Phase 1 defaults to flags OFF  — a TRADES-default replication checkpoint is trained WITHOUT
#                                    UNCLAMP_DEPTH / PRICE_REANCHOR, and simulation must match
#                                    training. Override with --intc-flags on if yours was trained
#                                    with them.
#   Phase 2 uses flags ON          — the TSLA lineage was trained with both.
# The flags are file-gated, appear in no command line and no output path, and cannot be recovered
# from a finished run. Each phase sets them explicitly and prints what it set.
#
#   bash scripts/intc_ddpm_then_tsla_sigma.sh --dry-run
#   nohup bash scripts/intc_ddpm_then_tsla_sigma.sh \
#         --intc-ckpt data/checkpoints/INTC_archive/<0.667 file>.ckpt \
#         --tsla-ckpt data/checkpoints/TRADES/val_ema=0.809_epoch=11_TSLA_...ckpt \
#         > queue.log 2>&1 & disown
set -uo pipefail

INTC_CKPT=""; TSLA_CKPT=""
INTC_START="2015-01-02"; INTC_END="2015-01-30"
TSLA_START="2015-01-02"; TSLA_END="2015-01-30"
ST="10:00:00"; ET="12:00:00"          # 2h window for phase 1
SIG_ST="09:30:00"; SIG_ET="10:00:00"  # 30-min window for the sigma sweep
SIGMAS="0.15 0.3 0.6 1.0"
SEED="30"; CAP_SECS=14400
INTC_FLAGS="off"                      # TRADES-default replication => trained without the data fixes
PHASE=""; DRY=0
PY="${PY:-python}"
OUT_DIR="queue/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do case "$1" in
  --intc-ckpt) INTC_CKPT="$2"; shift 2;; --tsla-ckpt) TSLA_CKPT="$2"; shift 2;;
  --intc-flags) INTC_FLAGS="$2"; shift 2;;
  --sigmas) SIGMAS="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --st) ST="$2"; shift 2;; --et) ET="$2"; shift 2;;
  --cap-secs) CAP_SECS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --phase) PHASE="$2"; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/real"
say () { echo "[$(date +%T)] $*"; }

days_for () { # days_for <ticker> <start> <end>  -> the two held-out test days
  local t="$1" s="$2" e="$3" d="data/$1/$1_$2_$3"
  [[ -d "$d" ]] || d=$(ls -d "data/$1/$1_$2_$3"_* 2>/dev/null | grep -v '\.zip$' | head -1)
  ls "$d" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u | tail -2 | tr -d '-' | tr '\n' ' '
}
INTC_DAYS=$(days_for INTC "$INTC_START" "$INTC_END")
TSLA_DAY=$(days_for TSLA "$TSLA_START" "$TSLA_END" | awk '{print $1}')

if [[ "$DRY" == "1" ]]; then
  cat <<EOF
=== queued run ===
out: $OUT_DIR

PHASE 1 — INTC DDPM-100, no interventions, ${ST}-${ET}
  ckpt      : ${INTC_CKPT:-<REQUIRED: --intc-ckpt>}
  test days : ${INTC_DAYS:-<INTC data not found>}
  flag files: $INTC_FLAGS   $([[ "$INTC_FLAGS" == off ]] && echo "(TRADES default: trained without the data fixes)" || echo "(override)")
  cells     : $(wc -w <<< "$INTC_DAYS") x ~1-3 h

PHASE 2 — TSLA sigma sweep, corrected prior, ${SIG_ST}-${SIG_ET}
  ckpt      : ${TSLA_CKPT:-<REQUIRED: --tsla-ckpt>}
  day       : ${TSLA_DAY:-<TSLA data not found>}
  sigmas    : $SIGMAS
  prior     : derived from TSLA real data (INTC's 0.49,0.48,0.03 is what broke the last sweep)
  flag files: on (the TSLA lineage was trained with both)
  cells     : $(wc -w <<< "$SIGMAS") x ~2-10 min
EOF
  exit 0
fi

pgrep -f "main.py" >/dev/null && { echo "!! training running — kill it first"; exit 1; }

# ---------------------------------------------------------------- phase 1
if [[ -z "$PHASE" || "$PHASE" == "1" ]]; then
  if [[ -f "$OUT_DIR/logs/.done_phase1" ]]; then say "phase 1 already done"; else
  [[ -n "$INTC_CKPT" && -f "$INTC_CKPT" ]] || { echo "!! --intc-ckpt required and must exist"; exit 1; }
  [[ -n "$INTC_DAYS" ]] || { echo "!! no INTC days found"; exit 1; }

  export TICKER=INTC TRADING_START="$INTC_START" TRADING_END="$INTC_END"
  if [[ "$INTC_FLAGS" == "off" ]]; then
    rm -f UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
    say "phase 1 flag files: OFF (TRADES-default replication)"
  else
    touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
    say "phase 1 flag files: ON (override)"
  fi
  "$PY" -c "import constants as c; print('   UNCLAMP_DEPTH',c.UNCLAMP_DEPTH,' PRICE_REANCHOR',c.PRICE_REANCHOR)"

  for D in $INTC_DAYS; do
    TAG="INTC_ddpm100_vanilla_${D}"
    [[ -f "$OUT_DIR/logs/.done_${TAG}" ]] && { say "SKIP $TAG"; continue; }
    say "-- $TAG  [DDPM-100 $D $ST-$ET, no decode flags]"
    T0=$(date +%s)
    if timeout -k 30 "$CAP_SECS" "$PY" -u ABIDES/abides.py -c world_agent_sim -t INTC -date "$D" \
         -st "$ST" -et "$ET" -d True -m TRADES -type DDPM -nsteps 100 -eta 0.0 \
         --ckpt-path "$INTC_CKPT" -seed "$SEED" > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
      S=$(( $(date +%s) - T0 )); touch "$OUT_DIR/logs/.done_${TAG}"
      grep -oE '/[^ ]+processed_orders\.csv' "$OUT_DIR/logs/${TAG}.txt" | tail -1 > "$OUT_DIR/logs/.csv_${TAG}"
      say "   done $((S/60))m $((S%60))s"
    else
      RC=$?; S=$(( $(date +%s) - T0 ))
      say "   $([[ $RC -eq 124 || $RC -eq 137 ]] && echo TIMEOUT || echo "ERROR rc=$RC") after $((S/60))m"
    fi
  done
  touch "$OUT_DIR/logs/.done_phase1"; say "PHASE 1 COMPLETE"
  fi
fi

# ---------------------------------------------------------------- phase 2
if [[ -z "$PHASE" || "$PHASE" == "2" ]]; then
  if [[ -f "$OUT_DIR/logs/.done_phase2" ]]; then say "phase 2 already done"; else
  [[ -n "$TSLA_CKPT" && -f "$TSLA_CKPT" ]] || { echo "!! --tsla-ckpt required and must exist"; exit 1; }
  [[ -n "$TSLA_DAY" ]] || { echo "!! no TSLA days found"; exit 1; }

  export TICKER=TSLA TRADING_START="$TSLA_START" TRADING_END="$TSLA_END"
  touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
  say "phase 2 flag files: ON (TSLA lineage was trained with both)"

  REAL="$OUT_DIR/real/real_TSLA_${TSLA_DAY}.csv"
  [[ -f "$REAL" ]] || "$PY" -m evaluation.stylized_custom.lobster_real_reference \
      --ticker TSLA --date "$TSLA_DAY" --st "$SIG_ST" --et "$SIG_ET" --out "$REAL" \
      > "$OUT_DIR/logs/real_TSLA.txt" 2>&1 || { echo "!! real reference failed"; cat "$OUT_DIR/logs/real_TSLA.txt"; exit 1; }

  PRIOR=$("$PY" - "$REAL" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); m = d.TYPE.value_counts(normalize=True)
l, c, e = (float(m.get(k, 0.0)) for k in ("LIMIT_ORDER", "ORDER_CANCELLED", "ORDER_EXECUTED"))
t = l + c + e
print(f"{l/t:.4f},{c/t:.4f},{e/t:.4f}")
PY
)
  say "TSLA prior derived from real $TSLA_DAY: $PRIOR   (INTC's is 0.49,0.48,0.03)"

  bash scripts/exec_bracket.sh --ticker TSLA --date "$TSLA_DAY" --st "$SIG_ST" --et "$SIG_ET" \
      --seed "$SEED" --sigmas "$SIGMAS" --real "$REAL" --ckpt-path "$TSLA_CKPT" \
      --extra "--type-prior $PRIOR" --out-dir "$OUT_DIR/tsla_sigma" 2>&1 | tee "$OUT_DIR/logs/tsla_sigma.log"

  touch "$OUT_DIR/logs/.done_phase2"; say "PHASE 2 COMPLETE — table in $OUT_DIR/tsla_sigma/summary.md"
  fi
fi

say "ALL DONE — $OUT_DIR"
