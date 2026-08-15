#!/bin/bash
# sweep_reanchored.sh — exploratory decode-time sweep on the FRESHLY-RETRAINED reanchored checkpoint,
# plus the decisive 75-minute stability test the PRICE_REANCHOR retrain was actually FOR.
#
# WHY THIS SHELL EXISTS ─────────────────────────────────────────────────────────────────────────
#   The winning config was tuned on ckpt 0.635. Established lesson: deterministic samplers are
#   HYPERSENSITIVE to checkpoint calibration, so the decode-time controls (depth-noise sigma, the
#   execution-rate target) are NOT universal constants — they must be re-tuned per checkpoint. This
#   sweep re-tunes them against the new reanchored, fully-converged model, and then runs the
#   long-horizon test: does the mid now GLIDE through $33.50 (z~-4) instead of hitting the OOD cliff?
#
# PRECONDITIONS ─────────────────────────────────────────────────────────────────────────────────
#   * reanchor_retrain.sh has finished; a NEW checkpoint is the newest .ckpt in data/checkpoints/TRADES.
#   * the OLD epoch-11 checkpoint has been archived (the -id loader matches by val_ema FLOAT value,
#     so two files with the same val_ema collide — this shell REFUSES if it detects that).
#   * reanchored .npy data is on disk and BOTH flag files belong ON (the new ckpt trained with both).
#   * no main.py running (GPU contention turned a 45-min sim into 3.5h before).
#
# Usage:  bash scripts/sweep_reanchored.sh                 # auto-discover newest ckpt, full sweep
#         bash scripts/sweep_reanchored.sh --id 0.628      # pin a specific val_ema
#         bash scripts/sweep_reanchored.sh --out-dir sweep_reanchored/rerun   # resume (.done sentinels)

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
CKPT_DIR="data/checkpoints/TRADES"
ID=""                                            # empty → auto-discover newest
REAL="ABIDES/log/market_replay_${TICKER}_2015-01-30_10-00-00_30/processed_orders.csv"
REAL_LONG="ABIDES/log/market_replay_${TICKER}_2015-01-30_11-00-00_30/processed_orders.csv"
OUT_DIR="sweep_reanchored/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --real) REAL="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"

# ── Stage 0: preflight ──────────────────────────────────────────────────────────────────────────
if pgrep -f "main.py" > /dev/null; then
  echo "!! training (main.py) still running — kill it first. Refusing (GPU contention)."; exit 1
fi

# BOTH flags belong ON for the reanchored ckpt (conditioning must match training). EXIT trap keeps
# them ON if the script dies. File flags, NOT env vars (env silently failed twice on this remote).
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG' EXIT
PRECHECK=$(python3 -c "import constants as cst; print(cst.UNCLAMP_DEPTH, cst.PRICE_REANCHOR)" 2>&1)
echo "flags → UNCLAMP_DEPTH PRICE_REANCHOR = $PRECHECK"
[[ "$PRECHECK" == "True True" ]] || { echo "!! flags not read True True — do NOT sweep. Got: $PRECHECK"; exit 1; }

# Discover the checkpoint. The loader (world_agent_sim.py) selects by the FLOAT value of val_ema=,
# so the chosen val must be UNIQUE in the dir or the match is nondeterministic.
valof () { basename "$1" | sed -E 's/^[^=]*=([0-9.]+).*/\1/'; }
if [[ -z "$ID" ]]; then
  NEWEST=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -1)
  [[ -n "$NEWEST" ]] || { echo "!! no .ckpt in $CKPT_DIR — did the retrain finish?"; exit 1; }
  ID=$(valof "$NEWEST")
  echo "auto-discovered newest checkpoint: $(basename "$NEWEST")  → -id $ID"
fi
# Uniqueness guard (covers both auto and --id): count files whose val_ema float equals ID.
COLLIDE=$(for f in "$CKPT_DIR"/*.ckpt; do valof "$f"; done | grep -Fxc "$ID" || true)
if [[ "${COLLIDE:-0}" -gt 1 ]]; then
  echo "!! $COLLIDE checkpoints share val_ema=$ID — the -id loader would pick nondeterministically."
  echo "   Archive the stale one(s) out of $CKPT_DIR (e.g. mv … _archive/) and re-run. Refusing."
  ls -t "$CKPT_DIR"/*.ckpt | while read -r f; do [[ "$(valof "$f")" == "$ID" ]] && echo "     $f"; done
  exit 1
fi
echo "# Reanchored sweep — $(date '+%F %T')  ckpt val_ema=$ID" > "$SUM"; echo "" >> "$SUM"

# Real-data quantile targets for --size-reshape (CPU-only; hard-verify the artifact, not the exit code).
if [[ ! -f data/quantile_targets/real_size_limit.npy ]]; then
  echo "── building real-data quantile targets ──"
  python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt"
fi
if [[ ! -f data/quantile_targets/real_size_limit.npy ]]; then
  echo "!! quantile targets still missing after build — every --size-reshape cell would crash."
  echo "   See $OUT_DIR/logs/build_targets.txt. Refusing."; exit 1
fi

# Real market-replay CSVs for flow_mix (both windows; pure-replay, no -d flag → CPU).
regen_replay () { # <et> <expected_csv> <log_name>
  local _ET="$1" _CSV="$2" _LOG="$3"
  [[ -f "$_CSV" ]] && return
  echo "── regenerating real market-replay CSV ($_ET window) ──"
  python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$_ET" \
    > "$OUT_DIR/logs/${_LOG}.txt" 2>&1
  [[ -f "$_CSV" ]] || echo "WARNING: replay done but $_CSV missing — flow_mix runs gen-only for that window"
}
regen_replay "$ET"      "$REAL"      "market_replay_1000"
regen_replay "$ET_LONG" "$REAL_LONG" "market_replay_1100"

LAST_CSV=""
run () { # run <tag> <type> <nsteps> <extra> [et_override] [seed]
  local TAG="$1" TYPE="$2" NS="$3" EXTRA="$4"
  local ET_RUN="${5:-$ET}" SEED_RUN="${6:-}"
  local REAL_RUN="$REAL"; [[ "$ET_RUN" == "$ET_LONG" ]] && REAL_RUN="$REAL_LONG"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  LAST_CSV=""
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; LAST_CSV=$(cat "${DONE}.csv" 2>/dev/null); return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET_RUN"
           -d True -m TRADES -type "$TYPE" -nsteps "$NS" -eta 0.0 -id "$ID")
  [[ -n "$SEED_RUN" ]] && A+=(-seed "$SEED_RUN")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA" ]] && A+=($EXTRA)
  echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  LAST_CSV="$CSV"
  { echo "## $TAG  (${SECS}s)"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1
    echo "csv: ${CSV:-none}"
    if [[ -n "$CSV" && -f "$REAL_RUN" ]]; then python -m evaluation.quantitative_eval.flow_mix --real "$REAL_RUN" --gen "$CSV" 2>&1
    elif [[ -n "$CSV" ]]; then python -m evaluation.quantitative_eval.flow_mix --gen "$CSV" 2>&1; fi
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo '```'; echo ""; } >> "$SUM"
  echo "$CSV" > "${DONE}.csv"; touch "$DONE"; echo "  done ${SECS}s"
}

drift () { # drift <tag> <csv>  — time-bucketed profile for a long-horizon cell (does it SURVIVE?)
  local TAG="$1" CSV="$2"
  [[ -n "$CSV" && -f "$CSV" ]] || { echo "  (no CSV for drift $TAG)"; return; }
  echo "── drift_profile $TAG"
  { echo "### drift_profile — $TAG"; echo '```'
    if [[ -f "$REAL_LONG" ]]; then python evaluation/diagnostics/drift_profile.py "$CSV" --real "$REAL_LONG" 2>&1
    else python evaluation/diagnostics/drift_profile.py "$CSV" 2>&1; fi
    echo '```'; echo ""; } >> "$SUM"
}

echo "════ STAGE 1: where does the fresh converged checkpoint sit? ════"
# A well-trained model STILL freezes under deterministic sampling without variance control — this
# cell demonstrates the thesis point on the NEW ckpt. Then the DDPM anchor: the fidelity this ckpt
# can actually reach (the target the fast sampler is chasing).
run "DDIM10_prior_raw"  DDIM 10  "--type-decode prior"
run "DDPM100_prior"     DDPM 100 "--type-decode prior"

echo "════ STAGE 2: depth-noise dose-response — RE-TUNE sigma on THIS checkpoint ════"
# On ckpt 0.635, sigma 0.3 was the sweet spot; the converged reanchored ckpt has different
# conditional variance, so the correct sigma may shift. Full fast-sampler candidate (sr+prior).
run "DDIM10_dn0.2_sr_prior" DDIM 10 "--depth-noise 0.2 --size-reshape --type-decode prior"
run "DDIM10_dn0.3_sr_prior" DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior"
run "DDIM10_dn0.4_sr_prior" DDIM 10 "--depth-noise 0.4 --size-reshape --type-decode prior"

echo "════ STAGE 3: controller-target sweep — hold realized exec at the real rate ════"
# The controller scales sigma to a target exec share; it interacts with the base sigma, so sweep
# the target at fixed sigma=0.3. On the 30-min window the controller was counterproductive before
# (throttled sigma) — this checks whether the converged ckpt changes that.
run "DDIM10_dn0.3_sr_prior_te0.03"  DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior --dn-target-exec 0.03"
run "DDIM10_dn0.3_sr_prior_te0.045" DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior --dn-target-exec 0.045"
run "DDIM10_dn0.3_sr_prior_te0.06"  DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior --dn-target-exec 0.06"

echo "════ STAGE 4: THE DECISIVE TEST — 75-min horizon on the reanchored ckpt ════"
# 09:30–11:00 = 75 min of generation; the real mid declines through \$33.50 (z~-4) here. Pre-reanchor
# BOTH long runs degenerated at that threshold. Success = mid glides through, no min-50 cliff, event
# rate steady. Controller ON (needed for long-horizon liquidity) + a fixed-sigma control.
run "DDIM10_dn0.3_sr_prior_te0.045_ET1100" DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior --dn-target-exec 0.045" "$ET_LONG"
drift "te0.045_ET1100" "$LAST_CSV"
run "DDIM10_dn0.3_sr_prior_ET1100"         DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior" "$ET_LONG"
drift "fixed_ET1100" "$LAST_CSV"

# ════ STAGE 5: seed robustness (GATED on the 30-min headline actually moving) ════
getB () { grep -o 'B_crossing_limit=[0-9]*' "$OUT_DIR/logs/$1.txt" 2>/dev/null | head -1 | cut -d= -f2; }
HEADLINE_B=$(getB DDIM10_dn0.3_sr_prior); HEADLINE_B="${HEADLINE_B:-0}"
if [[ "$HEADLINE_B" -gt 0 ]]; then
  echo "════ STAGE 5: seed robustness (headline B=$HEADLINE_B > 0) ════"
  run "DDIM10_dn0.3_sr_prior_s31" DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior" "" 31
  run "DDIM10_dn0.3_sr_prior_s32" DDIM 10 "--depth-noise 0.3 --size-reshape --type-decode prior" "" 32
else
  echo "════ STAGE 5 SKIPPED (headline DDIM10_dn0.3_sr_prior B=0 — nothing moved) ════"
  { echo "## Stage 5 — SKIPPED (headline B_crossing_limit=0)"; echo ""; } >> "$SUM"
fi

# ════ MORNING TABLE + LOB-BENCH MANIFEST ══════════════════════════════════════════════════════
python3 - "$SUM" <<'PY'
import re, sys
sum_path = sys.argv[1]
text = open(sum_path).read()
rows, manifest = [], []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    head = block.splitlines()[0]
    tag = head.split('(')[0].strip()
    if 'SKIPPED' in tag or 'ERROR' in head:
        continue
    def g(pat, default='-'):
        m = re.search(pat, block); return m.group(1) if m else default
    b = g(r'B_crossing_limit=(\d+)')
    a = g(r'A_market_order=(\d+)')
    mids = re.findall(r'unique mid-prices: (\d+)', block)
    mid = mids[-1] if mids else '-'                     # last = GENERATED
    execs = re.findall(r'ORDER_EXECUTED\s+([\d.]+)', block)
    ex = execs[-1] if execs else '-'
    zc = re.search(r'cond_z\[depth\]: min=([-\d.]+) mean=[-\d.]+ max=([-\d.]+)', block)
    zrange = f"{zc.group(1)}..{zc.group(2)}" if zc else '-'
    rows.append((tag, b, a, mid, ex, zrange))
    csvm = re.search(r'^csv: (.+)$', block, flags=re.M)
    if csvm and csvm.group(1) != 'none':
        manifest.append(f"  --gen {tag}={csvm.group(1)} \\")
hdr = f"{'cell':<40}{'B_cross':>8}{'A_mkt':>7}{'mids':>6}{'exec%':>7}  cond_z[depth]"
lines = [hdr, '-'*len(hdr)] + [f"{t:<40}{b:>8}{a:>7}{m:>6}{e:>7}  {z}" for t,b,a,m,e,z in rows]
table = "\n".join(lines)
print("\n════ MORNING TABLE  (real: mids=69, exec~7%; frozen: B=0, mids<20) ════\n" + table)
if manifest:
    print("\n════ LOB-Bench manifest (run locally on the Mac after copying CSVs) ════")
    print("external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \\")
    print("  --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-30_34140000_57660000_orderbook_10.csv \\")
    print("\n".join(manifest))
    print("  --out-dir lob_bench_reanchored --window 09:45")
body = open(sum_path).read()
open(sum_path, 'w').write("# MORNING TABLE\n```\n" + table + "\n```\n\n" + body)
PY

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary (table at top): $SUM"
echo "══════════════════════════════════════════"
echo "READ: Stage 2 → which sigma hits exec~7% / mids~69 on the NEW ckpt (re-tuned optimum);"
echo "  Stage 3 → whether the controller helps or throttles on 30-min now;"
echo "  Stage 4 → the drift_profile blocks: does the mid GLIDE through \$33.50 with no min-50 cliff?"
echo "  Stage 5 → seed robustness of the headline config."
