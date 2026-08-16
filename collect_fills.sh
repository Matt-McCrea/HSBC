#!/usr/bin/env bash
#
# collect_fills.sh — gather the twelve `fills`-phase runs (sweep_final.sh)
# into one clean folder: each run's sweep log plus its full ABIDES output
# directory (processed_orders.csv, plots, everything ABIDES wrote). Run from
# the repo root, after `./sweep_final.sh fills` has completed.
#
#   ./collect_fills.sh              collect, then tar it up
#   ./collect_fills.sh --no-tar     collect only, leave as a directory
#
# WHY NOT JUST GLOB ABIDES/log: the sim() name ("typedecode_l1", "sigma0.30",
# ...) is not the ABIDES output directory name -- that is built from date,
# sampler, eta, nsteps, the checkpoint truncated to 13 characters, and decode
# flag suffixes, and ABIDES/log holds months of unrelated runs besides. The
# reliable way to find the right directory is to read it back out of each
# run's own "Orderbook constructed and saved in:" log line.
#
# WHY SEARCH ACROSS ALL sweep_*/ DIRECTORIES: sweep_final.sh sets its output
# directory once per invocation ($OUT="sweep_$(date ...)"). If `fills` was
# ever re-run after a partial failure, its logs may be split across more than
# one sweep_* directory. For each run name we take the most recent log that
# exists, rather than assuming a single directory holds all twelve.
#
# No `set -e`: one missing/failed run must not stop the rest from collecting.
set -u

FILLS_RUNS=(
  typedecode_l1
  ddim1_vanilla_20150129
  ddpm100_vanilla_20150129
  ddim1_vanilla_20150130
  ddpm100_vanilla_20150130
  repl_ddpm100_vanilla_0129
  eta1_vanilla
  vanilla
  ddim100_corr
  sigma0.15
  sigma0.30
  sigma0.50
)

DEST="fills_collected_$(date +%Y%m%d_%H%M)"
DO_TAR=1
[ "${1:-}" = "--no-tar" ] && DO_TAR=0

mkdir -p "$DEST"
echo "collecting into $DEST"
echo

# Most recent log for a given run name, searched across every sweep_*/ dir.
find_log_for() {
  local name="$1"
  for d in $(ls -dt sweep_*/ 2>/dev/null); do
    if [ -f "${d}${name}.log" ]; then
      echo "${d}${name}.log"
      return 0
    fi
  done
  return 1
}

# The exact ABIDES output directory this run wrote to, read out of its own log.
csv_dir_from_log() {
  grep "Orderbook constructed and saved in:" "$1" 2>/dev/null \
    | tail -1 \
    | sed 's/^Orderbook constructed and saved in:[[:space:]]*//' \
    | tr -d '\r'
}

n_ok=0 n_missing_log=0 n_missing_dir=0

for name in "${FILLS_RUNS[@]}"; do
  log="$(find_log_for "$name")"
  if [ -z "$log" ]; then
    echo "MISSING  $name  -- no log found in any sweep_*/ directory"
    n_missing_log=$((n_missing_log+1))
    continue
  fi

  mkdir -p "$DEST/$name"
  cp "$log" "$DEST/$name/run.log"

  csvdir="$(csv_dir_from_log "$log")"
  if [ -z "$csvdir" ] || [ ! -d "$csvdir" ]; then
    echo "NO CSV   $name  -- run.log copied, but no output directory found"
    echo "         (likely timed out or failed -- check $DEST/$name/run.log)"
    n_missing_dir=$((n_missing_dir+1))
    continue
  fi

  cp -r "$csvdir" "$DEST/$name/simdir"
  echo "OK       $name  <-  $csvdir"
  n_ok=$((n_ok+1))
done

# Carry the sweep's own progress log along for reference, if findable.
for d in $(ls -dt sweep_*/ 2>/dev/null); do
  [ -f "${d}PROGRESS.txt" ] && cp "${d}PROGRESS.txt" "$DEST/PROGRESS_${d%/}.txt"
done

echo
echo "================ SUMMARY ================"
echo "ok: $n_ok / ${#FILLS_RUNS[@]}   missing log: $n_missing_log   missing csv dir: $n_missing_dir"
du -sh "$DEST"

if [ "$DO_TAR" -eq 1 ]; then
  tar czf "${DEST}.tgz" "$DEST"
  echo
  echo "tarred -> ${DEST}.tgz"
  du -sh "${DEST}.tgz"
  echo
  echo "pull with, e.g.:"
  echo "  rsync -avz <user>@<host>:$(pwd)/${DEST}.tgz ~/Desktop/"
fi
