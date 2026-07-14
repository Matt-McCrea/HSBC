#!/bin/bash
# overnight_reshape.sh — decode-time distribution repair: can we unfreeze the mid-price at
# 10-step cost WITHOUT retraining and WITHOUT touching the sampler?  (full-night edition, ~9h)
#
# ── WHERE EVERY PRIOR LEVER LANDED ────────────────────────────────────────────────────────────
#   CFG (guidance)        ✗  reshapes the conditional MEAN; adds no variance
#   --depth-temp (linear) ✗  cliff — linearly sliding a collapsed spike crosses all-at-once
#   HYBRID / CHURN        ✗  cliff + numerically unstable on the unclamped ckpt (cond_z ±1400)
#   unclamp retrain       ◐  fixed the SIGN axis (neg-depth decodes x11; DDPM B-share 35→40%)
#                            but NOT the MAGNITUDE axis: DDIM10 B_crossing_limit stayed 0 —
#                            deterministic sampling still collapses depth's variance, so the
#                            negative excursions are too small to exceed the spread.
#
# ── TONIGHT'S TWO NEW MECHANISMS (both decode-time only; sampler & weights untouched) ────────
#   --depth-reshape   Quantile match: rank each raw z within the model's OWN recent outputs
#                     (rolling buffer, midrank ties), read that quantile off the REAL signed-depth
#                     marginal. Nonlinear per-sample map — uses the continuous intra-spike
#                     variation as rank, so the bottom ~0.9% of ranks land on genuine -1..-10-tick
#                     crossing depths with REAL magnitudes. Fail-safe: a fully-degenerate source
#                     maps to the target's MEDIAN, not an extreme (unlike depth-temp's cliff).
#   --depth-noise σ   The dumb comparator: per-sample N(0,σ) on z_depth at decode, LIMIT only.
#   --size-reshape    Same quantile mechanism for size, per decoded type. Bonus: kills the
#                     30-40% negative-size decode waste (42% of DDPM batches were resampled).
#
# ── NIGHT STRUCTURE (info-per-minute order; .done sentinels make walltime truncation graceful) ─
#   Stage 0    preflight: refuse-if-training, flag, targets, BOTH replay CSVs (10:00 + 11:00)
#   Stage 1    headline: DDIM10 +dr / +dr+sr / +dr+sr+prior (the full fast-sampler candidate)
#   Stage 2    dumb-variance comparators: dn0.15 / dn0.3
#   Stage 3    generality: DPMpp10+dr+sr, sr-only isolation
#   Stage 4    ADAPTIVE GATE: parse headline B_crossing_limit → gates stages 8-9
#   Stage 5    NFE ladder: DDIM5 / DDIM20 full candidate (speed/fidelity frontier)
#   Stage 6    clamped-ckpt isolation: DDIM10+dr on 0.656, flag OFF for that cell (if reshape
#              unfreezes even the clamp-trained model, the fix is fully decode-side)
#   Stage 7    dn0.15+sr combo, then the slow DDPM100+dr+sr control
#   Stage 8    seed robustness ×2 (GATED on headline B>0)
#   Stage 9    extended horizon 09:30-11:00 (GATED; run LAST, ~2h — does the unfrozen market SURVIVE?)
#   End        morning one-table summary + LOB-Bench --gen manifest (score locally on the Mac)
#
# WHAT DECIDES SUCCESS per cell:  B_crossing_limit > 0  (nothing else has achieved this on DDIM),
# unique mids → tens (real=69; frozen 6-17; diverging 100+), sane cond_z[depth], reshape counts >0.
#
# Usage:  bash scripts/overnight_reshape.sh          # everything, ~9h
#         bash scripts/overnight_reshape.sh --id 0.635 --real <csv>

set -uo pipefail
TICKER="INTC"; DATE="20150130"; ST="09:30:00"; ET="10:00:00"; ET_LONG="11:00:00"
ID="0.635"                                   # the unclamped-retrain checkpoint
CLAMPED_ID="0.656"                           # the old clamped baseline (for the isolation cell)
REAL="ABIDES/log/market_replay_${TICKER}_2015-01-30_10-00-00_30/processed_orders.csv"
REAL_LONG="ABIDES/log/market_replay_${TICKER}_2015-01-30_11-00-00_30/processed_orders.csv"
OUT_DIR="overnight_reshape/$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do case "$1" in
  --id) ID="$2"; shift 2;; --real) REAL="$2"; shift 2;;
  --out-dir) OUT_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
mkdir -p "$OUT_DIR/logs"; SUM="$OUT_DIR/summary.md"
echo "# Reshape night — $(date '+%F %T')  ckpt=$ID" > "$SUM"; echo "" >> "$SUM"

# ── Stage 0: preflight ────────────────────────────────────────────────────────────────────────
if pgrep -f "main.py" > /dev/null; then
  echo "!! training (main.py) still running — kill it first (GPU contention turned a 45min run"
  echo "   into 3.5h last time). Refusing to start."; exit 1
fi

# Evaluating the UNCLAMPED checkpoint → conditioning must match training. File flag, not env
# (env vars silently failed twice on this remote). The EXIT trap guarantees the flag is never
# left OFF if the script dies inside the clamped-isolation cell (stage 6).
touch UNCLAMP_DEPTH_FLAG
trap 'touch UNCLAMP_DEPTH_FLAG' EXIT
echo "UNCLAMP_DEPTH_FLAG set (ckpt $ID is unclamp-trained; conditioning must match)"

# Real quantile targets (CPU-only, ~minutes). Built from the SAME preprocess_data pipeline as
# training — includes the pre-event-snapshot indexing fix, so signed depths are genuine.
if [[ ! -f data/quantile_targets/real_depth_limit.npy ]]; then
  echo "── building real-data quantile targets ──"
  python scripts/build_quantile_targets.py 2>&1 | tee "$OUT_DIR/logs/build_targets.txt" \
    || { echo "target build FAILED — reshape cells cannot run"; exit 1; }
fi

# Real market-replay CSVs for flow_mix (missing file crashed every flow_mix last night).
# Regenerated by the same config in pure-replay mode (no -d flag → no model, CPU).
regen_replay () { # regen_replay <et> <expected_csv> <log_name>
  local _ET="$1" _CSV="$2" _LOG="$3"
  [[ -f "$_CSV" ]] && return
  echo "── regenerating real market-replay CSV ($_ET window) ──"
  python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$_ET" \
    > "$OUT_DIR/logs/${_LOG}.txt" 2>&1
  [[ -f "$_CSV" ]] || echo "WARNING: replay finished but $_CSV still missing — flow_mix will run gen-only"
}
regen_replay "$ET"      "$REAL"      "market_replay_1000"
regen_replay "$ET_LONG" "$REAL_LONG" "market_replay_1100"   # for the stage-9 extended-horizon cell

run () { # run <tag> <type> <nsteps> <extra> [et_override] [seed] [id_override]
  local TAG="$1" TYPE="$2" NS="$3" EXTRA="$4"
  local ET_RUN="${5:-$ET}" SEED_RUN="${6:-}" ID_RUN="${7:-$ID}"
  local REAL_RUN="$REAL"; [[ "$ET_RUN" == "$ET_LONG" ]] && REAL_RUN="$REAL_LONG"
  local LOG="$OUT_DIR/logs/${TAG}.txt"; local DONE="$OUT_DIR/logs/.done_${TAG}"
  [[ -f "$DONE" ]] && { echo "  SKIP $TAG"; return; }
  echo "── $TAG"
  local S; S=$(mktemp); touch "$S"; local T0; T0=$(date +%s)
  local A=(python ABIDES/abides.py -c world_agent_sim -t "$TICKER" -date "$DATE" -st "$ST" -et "$ET_RUN"
           -d True -m TRADES -type "$TYPE" -nsteps "$NS" -eta 0.0 -id "$ID_RUN")
  [[ -n "$SEED_RUN" ]] && A+=(-seed "$SEED_RUN")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA" ]] && A+=($EXTRA)
  echo "   ${A[*]}"
  if ! "${A[@]}" > "$LOG" 2>&1; then echo "  ERROR — see $LOG"; echo "## $TAG — ERROR" >> "$SUM"; rm -f "$S"; return; fi
  local SECS=$(( $(date +%s) - T0 ))
  local CSV; CSV=$(find ABIDES/log -name processed_orders.csv -newer "$S" ! -path "*/paper/*" ! -path "*market_replay*" 2>/dev/null | sort | tail -1); rm -f "$S"
  { echo "## $TAG  (${SECS}s)"; echo '```'
    grep "checkpoint used" "$LOG" | tail -1
    echo "csv: ${CSV:-none}"
    if [[ -n "$CSV" && -f "$REAL_RUN" ]]; then python -m evaluation.quantitative_eval.flow_mix --real "$REAL_RUN" --gen "$CSV" 2>&1
    elif [[ -n "$CSV" ]]; then python -m evaluation.quantitative_eval.flow_mix --gen "$CSV" 2>&1; fi
    echo ""; sed -n '/=== WORLDAGENT DIAGNOSTICS ===/,/=== END DIAGNOSTICS ===/p' "$LOG"
    echo '```'; echo ""; } >> "$SUM"
  touch "$DONE"; echo "  done ${SECS}s"
}

echo "════ STAGE 1: the headline — quantile depth reshape on the frozen sampler ════"
run "DDIM10_dr"          DDIM 10 "--depth-reshape"
run "DDIM10_dr_sr"       DDIM 10 "--depth-reshape --size-reshape"
# Channel A regressed on ckpt 0.635 (market decode 0.78% vs real 2.8%); prior decode restores it.
# This cell = the full fast-sampler candidate: both channels repaired at 10-step cost.
run "DDIM10_dr_sr_prior" DDIM 10 "--depth-reshape --size-reshape --type-decode prior"

echo "════ STAGE 2: the dumb-variance comparator — is smart reshaping even needed? ════"
run "DDIM10_dn0.15"      DDIM 10 "--depth-noise 0.15"
run "DDIM10_dn0.3"       DDIM 10 "--depth-noise 0.3"

echo "════ STAGE 3: generality — other backbone, size-axis isolation ════"
run "DPMpp10_dr_sr"      DPM_SOLVER_PP 10 "--depth-reshape --size-reshape"
run "DDIM10_sr"          DDIM 10 "--size-reshape"                     # size axis alone

# ════ STAGE 4: adaptive gate — spend the night's tail based on the headline result ════
# Success-only cells (seeds, extended horizon) are wasted if the headline froze; diagnostic
# cells (ladder, isolation, combo, DDPM control) carry information either way and run regardless.
HEADLINE_B=$(grep -o 'B_crossing_limit=[0-9]*' "$OUT_DIR/logs/DDIM10_dr_sr_prior.txt" 2>/dev/null | head -1 | cut -d= -f2)
[[ -z "${HEADLINE_B:-}" ]] && HEADLINE_B=$(grep -o 'B_crossing_limit=[0-9]*' "$OUT_DIR/logs/DDIM10_dr.txt" 2>/dev/null | head -1 | cut -d= -f2)
HEADLINE_B="${HEADLINE_B:-0}"
echo "════ GATE: headline B_crossing_limit=$HEADLINE_B → success-only stages $([[ "$HEADLINE_B" -gt 0 ]] && echo ENABLED || echo SKIPPED) ════"
echo "**GATE: headline B_crossing_limit=$HEADLINE_B**" >> "$SUM"; echo "" >> "$SUM"

echo "════ STAGE 5: NFE ladder — does decode-time repair hold as acceleration deepens? ════"
run "DDIM5_dr_sr_prior"   DDIM 5  "--depth-reshape --size-reshape --type-decode prior"
run "DDIM20_dr_sr_prior"  DDIM 20 "--depth-reshape --size-reshape --type-decode prior"

echo "════ STAGE 6: clamped-ckpt isolation — is the fix decode-side and checkpoint-independent? ════"
# 0.656 was CLAMP-trained → conditioning must be clamped for this one cell. (Caveat, noted not
# fixed: normalization_stats.json on disk is the unclamped version — mean_depth 1.3790 vs the
# clamped 1.3847, a 0.4% difference; negligible, not worth the risk of file-swapping mid-script.)
rm -f UNCLAMP_DEPTH_FLAG
echo "  [flag OFF for clamped cell]"
run "DDIM10_dr_CLAMPED${CLAMPED_ID}" DDIM 10 "--depth-reshape" "" "" "$CLAMPED_ID"
touch UNCLAMP_DEPTH_FLAG
echo "  [flag restored ON]"

echo "════ STAGE 7: cheap-complete combo + the slow healthy-sampler control ════"
run "DDIM10_dn0.15_sr"   DDIM 10 "--depth-noise 0.15 --size-reshape"
# Control: reshape on the WORKING sampler. Tests (a) no harm, (b) whether it corrects DDPM's
# known depth-too-deep bias (LOB-Bench: median 4 ticks vs real 1). Slowest non-gated cell.
run "DDPM100_dr_sr"      DDPM 100 "--depth-reshape --size-reshape"

if [[ "$HEADLINE_B" -gt 0 ]]; then
  echo "════ STAGE 8: seed robustness — rule out a lucky draw on the headline claim ════"
  run "DDIM10_dr_sr_prior_s31" DDIM 10 "--depth-reshape --size-reshape --type-decode prior" "" 31
  run "DDIM10_dr_sr_prior_s32" DDIM 10 "--depth-reshape --size-reshape --type-decode prior" "" 32

  echo "════ STAGE 9: extended horizon — does the unfrozen market SURVIVE 75 min of generation? ════"
  run "DDIM10_dr_sr_prior_ET1100" DDIM 10 "--depth-reshape --size-reshape --type-decode prior" "$ET_LONG"
else
  echo "════ STAGES 8-9 SKIPPED (gate: headline B=0) ════"
  { echo "## Stages 8-9 — SKIPPED (gate: headline B_crossing_limit=0)"; echo ""; } >> "$SUM"
fi

# ════ MORNING TABLE + LOB-BENCH MANIFEST ══════════════════════════════════════════════════════
python3 - "$SUM" <<'PY'
import re, sys
sum_path = sys.argv[1]
text = open(sum_path).read()
rows, manifest = [], []
for block in re.split(r'^## ', text, flags=re.M)[1:]:
    tag = block.splitlines()[0].split('(')[0].strip()
    if 'SKIPPED' in tag or 'ERROR' in block.splitlines()[0]:
        continue
    def g(pat, default='-'):
        m = re.search(pat, block)
        return m.group(1) if m else default
    b   = g(r'B_crossing_limit=(\d+)')
    a   = g(r'A_market_order=(\d+)')
    mids = re.findall(r'unique mid-prices: (\d+)', block)
    mid = mids[-1] if mids else '-'          # last = GENERATED (first is REAL when present)
    m = re.search(r'depth_pre_drop: neg=(\d+) 0=(\d+) 1-2=(\d+) 3-5=(\d+) 6\+=(\d+)', block)
    negpct = f"{100*int(m.group(1))/max(1,sum(int(m.group(i)) for i in range(1,6))):.1f}%" if m else '-'
    resh = g(r'reshape: depth_applied=(\d+)')
    zc = re.search(r'cond_z\[depth\]: min=([-\d.]+) mean=[-\d.]+ max=([-\d.]+)', block)
    zrange = f"{zc.group(1)}..{zc.group(2)}" if zc else '-'
    rows.append((tag, b, a, mid, negpct, resh, zrange))
    csvm = re.search(r'^csv: (.+)$', block, flags=re.M)
    if csvm and csvm.group(1) != 'none':
        manifest.append(f"  --gen {tag}={csvm.group(1)} \\")
hdr = f"{'cell':<28}{'B_cross':>8}{'A_mkt':>7}{'mids':>6}{'neg%':>7}{'reshaped':>9}  cond_z[depth]"
lines = [hdr, '-'*len(hdr)] + [f"{t:<28}{b:>8}{a:>7}{m:>6}{n:>7}{r:>9}  {z}" for t,b,a,m,n,r,z in rows]
table = "\n".join(lines)
print("\n════ MORNING TABLE  (real: mids=69, exec~7%; frozen: B=0, mids<20) ════\n" + table)
if manifest:
    print("\n════ LOB-Bench manifest (run locally on the Mac after copying CSVs) ════")
    print("external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \\")
    print("  --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-30_34140000_57660000_orderbook_10.csv \\")
    print("\n".join(manifest))
    print("  --out-dir lob_bench_reshape --window 09:45")
with open(sum_path) as f: body = f.read()
with open(sum_path, 'w') as f:
    f.write("# MORNING TABLE\n```\n" + table + "\n```\n\n" + body)
PY

echo ""; echo "══════════════════════════════════════════"
echo "  Done. Summary (table at top): $SUM"
echo "══════════════════════════════════════════"
echo "READ: B_crossing_limit and mids per cell; the DDIM5→10→20 ladder trend; whether the"
echo "  CLAMPED-0.656 isolation cell also unfroze (fix = decode-side, checkpoint-independent);"
echo "  dr vs dn cells (smart map vs plain noise); and the ET1100 cell for survival over 75 min."
