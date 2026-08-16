#!/bin/bash
# overnight_unclamp.sh — evaluate the UNCLAMPED-depth checkpoint(s), correctly flagged, one command.
#
# Companion to overnight.sh (the CLAMPED-baseline night) — do NOT use overnight.sh for this
# checkpoint. It explicitly unsets UNCLAMP_DEPTH and would silently evaluate with the WRONG
# (clamped) conditioning, invalidating the whole test — exactly the train/sim mismatch this
# whole investigation has been careful to avoid.
#
# THE POINT of tonight: does plain DDIM10 (eta=0, no churn, no hybrid) now show a genuine
# negative-depth bucket and move the mid-price on its own, now that the model was actually
# trained on signed depth? That single number is what this whole retrain was for.
#
# Prerequisites:
#   1. Kill the training process first — frees the GPU AND finalizes the latest checkpoint file
#      to disk (a still-running process may not have flushed it yet):
#        kill $(cat unclamp_train.pid)
#   2. Then run this script. It refuses to start if it still finds a training process alive —
#      today's DDPM_100 run took 3.5h instead of ~45min from exactly this kind of contention.
#
# Usage:
#   bash scripts/overnight_unclamp.sh                       # auto-discovers checkpoints from this run
#   bash scripts/overnight_unclamp.sh --ids "0.638 0.644"   # pin specific checkpoints instead
#   bash scripts/overnight_unclamp.sh --n-ckpts 5            # widen auto-discovery bracket (default 3)

set -uo pipefail
REAL="ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv"
IDS=""; N_CKPTS=3
while [[ $# -gt 0 ]]; do case "$1" in
  --ids) IDS="$2"; shift 2;;
  --n-ckpts) N_CKPTS="$2"; shift 2;;
  --real) REAL="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="overnight_unclamp/${STAMP}"; mkdir -p "$DIR"
LOG="$DIR/master.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 1. Safety: training must be dead first — GPU needs to be free, and the checkpoint needs to be
#    fully flushed to disk before we try to load it.
if pgrep -f "main.py" > /dev/null; then
  echo "!! a main.py process is still running (pgrep found it). Kill it first:"
  echo "     kill \$(cat unclamp_train.pid)"
  echo "   then re-run this script. Refusing to start."
  exit 1
fi

# 2. Set the flag — the whole point of tonight requires it. A FILE, not an env var: exports have
#    failed silently on this remote twice already (didn't propagate to the actual python process).
touch UNCLAMP_DEPTH_FLAG
say "UNCLAMP_DEPTH_FLAG set: $(ls -la UNCLAMP_DEPTH_FLAG)"

# 3. Auto-discover checkpoints from THIS run, unless --ids was given explicitly. "Newer than
#    unclamp_train.pid" works because that file is written exactly once, at launch, and never
#    touched again (unlike unclamp_train.log, which keeps getting appended to and so keeps
#    updating its own mtime — not usable as a launch-time marker).
if [[ -z "$IDS" ]]; then
  REF="unclamp_train.pid"
  if [[ ! -f "$REF" ]]; then
    echo "!! $REF not found — can't auto-detect which checkpoints are from the unclamp run."
    echo "   Pass --ids \"<val_ema1> <val_ema2> ...\" manually (see ls data/checkpoints/TRADES/)."
    exit 1
  fi
  IDS=$(python3 - "$REF" "$N_CKPTS" <<'PY'
import sys, os, glob, re
ref_mtime = os.path.getmtime(sys.argv[1])
n = int(sys.argv[2])
ckpts = glob.glob("data/checkpoints/TRADES/*.ckpt")
new = []
for f in ckpts:
    if os.path.getmtime(f) <= ref_mtime:
        continue
    m = re.search(r"val_ema=([\d.]+)_", os.path.basename(f))
    if m:
        new.append((float(m.group(1)), m.group(1)))   # sort by float, print the ORIGINAL string
new.sort(key=lambda x: x[0])
print(" ".join(s for _, s in new[:n]))
PY
)
  if [[ -z "$IDS" ]]; then
    echo "!! no checkpoints found newer than $REF — has training actually saved any yet?"
    echo "   ls -la data/checkpoints/TRADES/ to check, or pass --ids manually."
    exit 1
  fi
fi
say "evaluating unclamped checkpoints: $IDS"

say "──── STAGE 1/2: open-loop sensitivity matrix (cheap) ────"
bash scripts/open_loop_sweep.sh --ids "$IDS" 2>&1 | tee -a "$LOG"
say "stage 1 done"

say "──── STAGE 2/2: closed-loop battery ────"
bash scripts/eval_new_checkpoint.sh --real "$REAL" --ids "$IDS" 2>&1 | tee -a "$LOG"
say "stage 2 done"

say "=== OVERNIGHT (UNCLAMP) COMPLETE ==="
say "THE decisive line: DDIM10_eta0's DIAG depth_pre_drop negative bucket, and unique mid count,"
say "in each ckpt_*/summary.md under eval_new_ckpt/. If DDIM10 alone (no churn, no hybrid) now"
say "moves the mid where it froze before, the depth clamp was the root cause -- confirmed."
say "Also check DIAG execution_channels: does Channel B (crossing limit) activate now, matching"
say "the ~35% share measured on the clamped DDPM baseline?"
