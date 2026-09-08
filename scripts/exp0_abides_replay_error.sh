#!/bin/bash
# exp0_abides_replay_error.sh — Experiment 0 (instability paper). Feeds ONLY real, ground-truth
# order flow (no diffusion model, no GPU) through ABIDES's real-replay path, for every INTC day
# x a range of session lengths, and measures whether ABIDES's own book-state error (vs the raw
# LOBSTER file replayed independently by rl_execution/book_state_error.py) stays flat/bounded or
# keeps climbing. This gates everything else in the instability paper plan — must run first.
#
# Resumable via .done sentinels (same pattern as scripts/checkpoint_stability.sh). CPU-only —
# no GPU contention, no training/main.py conflict.
#
# MEMORY WARNING (learned the hard way, locally): ABIDES's real-replay keeps the full order
# stream + LOB snapshots in memory for the whole session (log_orders=True, wide_book=True, no
# bounded logging) -- cost scales badly with session length. A single 390-minute (full-day) real
# replay hit 23GB RSS and OOM'd an 8GB Mac. Every run below is wrapped in `ulimit -v` (default
# 16GB, override with --max-mem-gb) so a single run gets killed cleanly instead of taking the
# whole box down -- raise it once you know this box's actual headroom, and remember this box may
# ALSO be running GPU training/sampling that wants its own RAM headroom at the same time.
# (`ulimit -v` is a no-op on macOS -- Darwin doesn't enforce RLIMIT_AS -- prints a harmless
# warning and runs uncapped there; it works as intended on Linux, i.e. the remote box.)
#
# Usage:  bash scripts/exp0_abides_replay_error.sh
#         bash scripts/exp0_abides_replay_error.sh --days "20150130 20150107" --lengths "30 60"
#         bash scripts/exp0_abides_replay_error.sh --max-mem-gb 48   # bigger box, full 390min sweep
set -uo pipefail
TICKER="INTC"; ST="09:30:00"; SEED="30"; MAX_MEM_GB=16
LOB_DIR="data/INTC/INTC_2015-01-02_2015-01-30"
DAYS="20150102 20150105 20150106 20150107 20150108 20150109 20150112 20150113 20150114 20150115 \
20150116 20150120 20150121 20150122 20150123 20150126 20150127 20150128 20150129 20150130"
# minutes from 09:30 -> end-of-window clock time; 390min ~= 16:00 (full continuous session)
LENGTHS="30 60 90 120 180 240 390"
OUT_DIR="exp0_results/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --days) DAYS="$2"; shift 2;; --lengths) LENGTHS="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
  --lob-dir) LOB_DIR="$2"; shift 2;; --max-mem-gb) MAX_MEM_GB="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
echo "memory cap per run: ${MAX_MEM_GB}GB (ulimit -v) -- raise with --max-mem-gb if this box has more headroom"
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
echo "# Experiment 0 — ABIDES real-flow book-state error — $(date '+%F %T')" > "$SUM"
echo "days: $(echo $DAYS | wc -w | tr -d ' ')  lengths(min): [$LENGTHS]  out: $OUT_DIR"

ymd_dash () { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
etdash () { echo "${1//:/-}"; }
et_for_length () {  # minutes from 09:30 -> HH:MM:SS clock time
  python3 -c "print((__import__('datetime').datetime(2000,1,1,9,30)+__import__('datetime').timedelta(minutes=$1)).strftime('%H:%M:%S'))"
}
message_csv_for () { echo "$LOB_DIR/${TICKER}_$(ymd_dash "$1")_34140000_57660000_message_10.csv"; }

real_dir_for () { echo "ABIDES/log/market_replay_${TICKER}_$(ymd_dash "$1")_$(etdash "$2")_${SEED}"; }
ensure_real_run () {  # ensure_real_run <day> <et> -> prints the run's log dir
  local D="$1" ETv="$2" DIR; DIR=$(real_dir_for "$D" "$ETv")
  if [[ ! -f "$DIR/EXCHANGE_AGENT.bz2" ]]; then
    echo "  -- real replay $D -> $ETv" >&2
    ( ulimit -v $((MAX_MEM_GB * 1024 * 1024))
      exec python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$D" -st "$ST" -et "$ETv" -seed "$SEED" \
    ) > "$OUT_DIR/logs/real_${D}_$(etdash "$ETv").txt" 2>&1
    local RC=$?
    [[ "$RC" -ne 0 ]] && echo "    (exit $RC -- check for OOM-kill if this run's length is large; see logs/real_${D}_$(etdash "$ETv").txt)" >&2
  fi
  echo "$DIR"
}

for D in $DAYS; do
  MSG_CSV=$(message_csv_for "$D")
  if [[ ! -f "$MSG_CSV" ]]; then echo "!! no message CSV for $D ($MSG_CSV) — skipping day"; continue; fi
  for L in $LENGTHS; do
    TAG="${D}__${L}min"
    DONE="$OUT_DIR/logs/.done_${TAG}"
    [[ -f "$DONE" ]] && { echo "SKIP $TAG"; continue; }
    ET=$(et_for_length "$L")
    RUN_DIR=$(ensure_real_run "$D" "$ET")
    if [[ ! -f "$RUN_DIR/EXCHANGE_AGENT.bz2" ]]; then
      echo "ERROR $TAG — no EXCHANGE_AGENT.bz2 at $RUN_DIR (see logs/real_${D}_$(etdash "$ET").txt)"
      echo "## $TAG — ERROR (real replay failed)" >> "$SUM"
      continue
    fi
    CSV_OUT="$OUT_DIR/csv/${D}/${L}min.csv"
    echo "-- $TAG"
    if python -m rl_execution.book_state_error --message-csv "$MSG_CSV" \
        --exchange-bz2 "$RUN_DIR/EXCHANGE_AGENT.bz2" --out "$CSV_OUT" \
        > "$OUT_DIR/logs/${TAG}.txt" 2>&1; then
      { echo "## $TAG"; echo '```'; tail -4 "$OUT_DIR/logs/${TAG}.txt"; echo '```'; echo ""; } >> "$SUM"
      touch "$DONE"; echo "  done"
    else
      echo "  ERROR — see logs/${TAG}.txt"
      echo "## $TAG — ERROR (book_state_error.py failed)" >> "$SUM"
    fi
  done
done

# Master rollup: one row per day x length, plus the single most important number this
# experiment produces -- did a one-sided (unquoted) book ever occur on REAL flow, anywhere.
python3 - "$OUT_DIR" "$SUM" <<'PY'
import glob, os, sys
import pandas as pd

out_dir, sum_path = sys.argv[1], sys.argv[2]
rows = []
any_one_sided_ever = False
for csv in sorted(glob.glob(os.path.join(out_dir, "csv", "*", "*min.csv"))):
    day = os.path.basename(os.path.dirname(csv))
    length = os.path.basename(csv).replace("min.csv", "")
    df = pd.read_csv(csv)
    early = df.iloc[min(2, len(df) - 1)]
    final = df.iloc[-1]
    growing = abs(final.shares_missing) > 1.5 * max(abs(early.shares_missing), 1)
    one_sided = bool(df.one_sided.any())
    any_one_sided_ever = any_one_sided_ever or one_sided
    rows.append((day, length, int(final.shares_missing), int(final.depth_error),
                 "GROWING" if growing else "flat", one_sided))

hdr = f"{'day':<12}{'len_min':>8}{'final_shares_missing':>22}{'final_depth_error':>19}{'trend':>10}{'one_sided':>11}"
tab = "\n".join([hdr, "-" * len(hdr)] +
                [f"{d:<12}{l:>8}{sm:>22}{de:>19}{tr:>10}{str(os):>11}" for d, l, sm, de, tr, os in rows])
headline = (
    "ABIDES's own book-state error on REAL order flow "
    + ("DID reproduce a one-sided (unquoted) book at least once -- the pathology is not "
       "exclusive to the generative model; every later experiment needs an ABIDES-baseline "
       "control."
       if any_one_sided_ever else
       "NEVER produced a one-sided (unquoted) book, at any day or session length tested -- "
       "consistent with that pathology belonging to the model, not the engine.")
)
print("\n==== EXPERIMENT 0 — MASTER TABLE ====\n" + tab)
print("\nHEADLINE: " + headline)
with open(sum_path, "a") as f:
    f.write("\n# MASTER TABLE\n```\n" + tab + "\n```\n\n## Headline\n" + headline + "\n")
PY

echo ""; echo "Done. Summary: $SUM"
