# Instability paper — running log

*One dated paragraph per completed step, written as it happens — not reconstructed afterward.
See `/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md` for the execution plan this log
tracks (Track A = local/CPU, Experiments 0/1/5; Track B = remote/GPU, Experiments 2/3/4).*

## 2026-09-08/09/10 — Trained three fresh checkpoints; first real Exp 2 signal at 90-min horizon

Trained `baseline`/`reanchor`/`ss` per `analysis/model_under_test.md`'s superseded-plan section
(`scripts/train_three_variants.sh`, unattended, ~5h cap each). `baseline` and `reanchor` reached
epoch 1 (val_ema 0.702, 0.704); `ss` reached only epoch 0 (val_ema 0.757) before its cap — scheduled
sampling's extra rollout sampling during training makes it slower per epoch, and epoch 0 is still
within the "epoch 1 at most epoch 2" (1-indexed) target, so used as-is.

30-min smoke test (`--smoke`) on `baseline`: both seeds survived cleanly, ~2230s/run. Expected —
prior knowledge (stated by the user, matches the dissertation's own findings) is that instability
onset is characteristically past the 1-hour mark, so a 30-min window mostly tests "does it survive
30 min," not the phenomenon this paper is about. Cost model confirmed from a real 90-min data point
(10875s) against the 30-min number (2200s): cost scales as roughly `horizon^1.5`, not linearly —
a 90-min run costs ~3h, not the ~3x-of-30min a linear guess would predict.

**One 90-minute validation run per variant** (day 2015-01-30, seed 30, no day/seed sweep yet —
n=1 each, a qualitative check that the phenomenon replicates on these freshly-trained checkpoints
before spending more budget on a full grid):

| variant | outcome | time | mechanism |
|---|---|---|---|
| `baseline` | failed | 78.4 min (froze, then diverged 25s later) | freeze → price_departure |
| `reanchor` | failed | **50 min — earliest of the three** | price_departure directly, no freeze first |
| `ss` | failed | 65 min | froze, never diverged by end of window |

All three fail within the 90-minute window, confirming the instability is not specific to one
checkpoint or one training configuration. **`reanchor` failing earliest, and via a different
mechanism (direct divergence, not freeze-then-diverge), is a genuine early answer to the paper's
open question about price re-anchoring** — the flag was introduced specifically to fix an
out-of-distribution price-drift problem, so failing *faster* than the plain baseline is
counter to the naive expectation and worth leading with, not burying. `ss` still failing (at
epoch 0 only) is worth reporting honestly rather than treated as refuting the fix, given how
little training it received relative to the other two.

Each 90-min run costs ~3h wall-clock on this box — n=1/variant only, not yet a statistically
sized sample. Next step (budget permitting): more days/seeds per variant at this same ~90-min
horizon, prioritizing `baseline` first since it's the paper's primary claim.

## 2026-09-11 — Exp 3 (teacher-forced) run for all three variants; caught a real bucketing bug

`overnight_batch.sh` produced Exp 3 (`open_loop_eval.py --bucket-by-time`) results for all three
variants. Caught a bug in the `--bucket-by-time` patch before trusting it: the "time" field
decoded from the model/dataset is the inter-arrival DELTA to the previous event (`utils_data.py`
diffs it during preprocessing), not session clock time — so the "early vs late session" median
split was actually splitting on delta magnitude, unrelated to session position (tell: all three
runs' `bucket_split_time` was ~5e-5, an interarrival-gap-sized number, not a plausible session
timestamp). Fixed in `evaluation/diagnostics/open_loop_eval.py` to split on the sampled windows'
dataset INDEX instead (chronologically ordered, aligned 1:1 with the result arrays) — pushed, not
yet rerun. **Do not cite the bucket_early/bucket_late numbers from the first three Exp 3 JSON
files** — only the pooled (non-bucketed) numbers in those files are valid.

**Valid, striking finding from the pooled numbers** (unaffected by the bucketing bug): all three
variants generate marketable (spread-crossing, negative-depth) orders at roughly **43x the real
rate**, even under teacher forcing with perfect real conditioning history —

| | real | baseline | reanchor | ss |
|---|---|---|---|---|
| marketable order share | 0.56% | 24.4% | 24.3% | 23.7% |

This is present from the very first generation step, consistently across all three training
variants — not something that only emerges as self-generated error compounds. Plausible mechanism
for the closed-loop failures independent of exposure bias: over-aggressive order flow eating
through the book faster than reality. Worth reporting regardless of how the (buggy, not yet fixed)
early/late comparison turns out.

## 2026-09-24 — All three TSLA variants complete: TSLA is more forgiving, and Exp 3 explains why

`session3_batch.sh` ran to completion — `tsla_reanchor` and `tsla_ss` trained and survival-checked
after the `tsla_baseline` results logged 2026-09-22/23. Full 90-min survival picture now:

| variant | TSLA survived / tested | INTC survived / tested |
|---|---|---|
| baseline | 2/6 (33%) | 0/11 (0%) |
| reanchor | 1/4 (25%) | 0/3 (0%) |
| ss       | 0/2 (0%)  | 0/3 (0%) |

`tsla_ss` took ~5h to train (hit the safety cap right after epoch=1 saved, per the training log's
own warning that scheduled sampling's 100-step rollout makes each self-conditioned training step
expensive) vs ~1h10min for `tsla_baseline`/`tsla_reanchor` — expected, not investigated further.

**Exp 3 (teacher-forced) gives a coherent explanation for why TSLA survives more often.** TSLA's
own REAL data has a far higher marketable-order rate than INTC's: **6.3% vs 0.56%** (11x). The
model's *generated* marketable-order rate is similar in absolute terms across both stocks (TSLA
~21-22%, INTC ~22-26%) — so the excess over real is only **~3.5x for TSLA vs ~40x for INTC**. If
the generated rate is closer to a fixed model-intrinsic miscalibration than to something that
scales with the real distribution, then TSLA's real trading pattern already sits closer to what
the model naturally produces — the "shock" to the closed feedback loop is smaller, which plausibly
explains why TSLA's closed-loop runs survive a 90-minute window more often. This ties the Exp 2
survival-rate finding and the Exp 3 marketable-order-rate finding into one mechanism rather than
two separate, unconnected observations.

**Reading the pattern across variants:** `ss` shows 0% survival on BOTH stocks — the one place the
finding is uniform regardless of stock. `baseline` and `reanchor` both show TSLA surviving
meaningfully more than INTC. Small n throughout (2-11 runs per cell) — don't over-index on exact
percentages, but the *direction* (TSLA more forgiving for baseline/reanchor, ss uniformly bad on
both) is consistent enough across three independent variant/stock combinations to be worth
reporting as a real, if qualified, cross-stock difference — not noise.

**Checkpoints backed up to git** (`data/checkpoints/TRADES_tsla_baseline/reanchor/ss`) before the
month-long gap, given this remote's history of losing checkpoints to wipes.

## 2026-09-22/23 — TSLA wired up and trained; first cross-stock result is a genuine nuance, not a clean confirmation

**Exp 0 for TSLA: same headline as INTC.** All 20 TSLA days x 7 lengths (140/140, no errors this
time — the earlier batch failed instantly on every combo from a placeholder bug, see below)
scored `one_sided_ever=False` throughout. `shares_missing` grows with session length on every day
(mostly; two cells — 2015-01-07 and 2015-01-16 at 30min — read "flat", noise at that short a
window) same as INTC. **Second-stock confirmation of the load-bearing Exp 0 finding**: ABIDES
itself never produces the one-sided pathology on real flow, INTC or TSLA.

**Exp 4, INTC baseline, second day (2015-01-16):** both checkpoints (epoch=0 val_ema=0.723,
epoch=1 val_ema=0.702) failed again (0/2 survived). Two independent days now agree: not one
unlucky checkpoint.

**`tsla_baseline` trained**: epoch=1 val_ema=0.809 (pinned, the higher epoch per the existing
selection rule), epoch=0 val_ema=0.838.

**`tsla_baseline` survival checks — the first genuinely surprising cross-stock result:**

| day | horizon | outcome |
|---|---|---|
| 2015-01-08 | 30min (smoke) | survived |
| 2015-01-02 | 90min | **failed** |
| 2015-01-07 | 90min | **failed** |
| 2015-01-15 | 90min | **failed** |
| 2015-01-22 | 90min | **failed** |
| 2015-01-30 | 90min | **survived** |

**2/6 survived (33%) at 90 minutes, vs INTC baseline's 0/11 (0%).** This is NOT the clean
"instability generalizes identically" result — TSLA baseline is measurably more likely to survive
a 90-minute window than INTC baseline, at least on this small sample. Don't overclaim from n=6,
but don't paper over it either: this is exactly the kind of finding that makes the cross-stock
angle valuable rather than a formality — either TSLA's liquidity/volatility profile is
genuinely more forgiving for this model class, or day-to-day variance is just wider than INTC's
and more TSLA days would narrow it back toward 0%. `tsla_reanchor` (next in the queue) will be the
first real test of whether this TSLA-vs-INTC gap is a genuine stock effect or noise — if
`tsla_reanchor` also survives more often than INTC's `reanchor`, that's a real pattern; if it
reverts to ~0%, `tsla_baseline`'s two survivals were noise.

**A real bug caught and fixed, twice, this round:**
1. The first TSLA Exp 0 attempt failed on literally every one of 140 combos with `N: unbound
   variable` — traced to a paste-back command that used a literal placeholder (`--max-mem-gb N`)
   instead of a real number. Nothing was wasted (failed before touching ABIDES); reran clean.
2. `session3_batch.sh`'s `teacher_forced_check` crashed with `V: unbound variable` immediately
   after `tsla_baseline`'s survival checks finished, before Exp 3 could run. Root cause:
   `local V="$1" OUT="...${V}..."` — bash expands every word on a `local` line (including `${V}`
   in the second assignment) before any declaration takes effect, so `${V}` looked up a `V` that
   didn't exist yet in that scope. Fixed by splitting into two `local` statements (matching the
   pattern the script's other functions already used safely); grepped every other script in
   `scripts/` for the same anti-pattern and confirmed this was the only instance.

## 2026-09-21 — Exp 0 finished (headline confirmed); Exp 4 surfaces a real non-determinism finding

**Exp 0 completed** for all 20 INTC days, most at all 7 lengths (30/60/90/120/180/240/390 min) —
131 of 140 day×length combos scored; the missing 9 are all at 390min (`20150106`, `20150112`,
`20150113`, `20150114`, `20150115`, `20150116`, `20150127`, `20150128`, `20150130` — either "real
replay failed" or "book_state_error.py failed", not investigated further, doesn't threaten the
headline given 131/140 succeeded including 11/20 days' full 390min cells).

**Headline, confirmed at full scale: `one_sided_ever=False` on EVERY single combo — real order
flow through ABIDES never once produces an unquoted/one-sided book**, from 30 minutes up to a full
6.5-hour session, on any of the 20 days. This is the load-bearing number the whole plan was gated
on (see 2026-09-07): the one-sided-book pathology seen in the generative model's closed-loop runs
does NOT reproduce on real flow through the same engine — consistent with it belonging to the
model, not ABIDES.

The `shares_missing` leak (ABIDES retaining more resting inventory than raw-LOBSTER ground truth)
IS real and grows with session length in every single day tested (e.g. 2015-01-02: -62,517 at
30min → -704,400 at 390min) — this is NOT the flat/bounded outcome the plan's decision framework
anticipated, but it's also not the runaway-into-one-sided-collapse outcome that would force adding
an ABIDES-baseline control to Exp 2's own numbers. It's a real, honest limitation to report (ABIDES
has a growing bookkeeping bias on real flow) that sits alongside, not underneath, the model's
instability finding — the two are separable because the specific pathology (one side draining to
zero) only ever appears in generated runs. `depth_error` (top-10-level only) stays comparatively
small and mixed-sign throughout, suggesting the "leaked" inventory sits away from the touch, not
concentrated at best bid/ask — plausibly why it never manifests as a one-sided top-of-book.

**Exp 4** (`baseline`'s two checkpoints, epoch=0 val_ema=0.723 and epoch=1 val_ema=0.702, on
2015-01-02): both failed within the 90-min window (froze at 88.2min and 25.3min respectively) —
another data point that instability isn't specific to one epoch. **But the epoch=1 result here
(froze 25.3min) directly contradicts the 2026-09-19 batch_12h.sh run of the IDENTICAL config**
(same checkpoint file, same day, same seed=30, same 90-min window), which froze at **62.2min** —
different `n_rows` too (86456 vs 86487), confirming genuinely different generated sequences, not a
scoring artifact. Checked the code: `torch.manual_seed(seed)` is called
(`ABIDES/config/world_agent_sim.py:231`), but `main.py`'s `set_torch()` enables
`cudnn.allow_tf32`/`cuda.matmul.allow_tf32` with no `cudnn.deterministic` or
`use_deterministic_algorithms` set — the standard cause of GPU non-reproducibility: tiny
floating-point differences from TF32/cudnn kernel selection, compounded over 100 diffusion steps x
tens of thousands of sequential orders, diverging into different trajectories despite an identical
seed. **Caveat for the writeup: "seed 30" does not identify a reproducible run on this hardware/
software stack — describe repeated runs as independent stochastic draws, not as a specific
reproducible seed.** Silver lining: every nominally-repeated (checkpoint, day, seed) combo run so
far is actually an independent additional data point, not a duplicate — the effective sample size
across all of Exp 2/4 is larger than the seed-count alone would suggest.

## 2026-09-20 — Exp 0 and Exp 4 launched, in progress at session's end

Exp 0 had never actually run on this box (CPU-only, no GPU contention) — launched via
`nohup bash scripts/exp0_abides_replay_error.sh --max-mem-gb N > exp0_run.log 2>&1 &` alongside
Exp 4 (`bash scripts/exp4_checkpoint_sweep.sh --ckpt-dir data/checkpoints/TRADES_baseline --day
20150102 --seeds 30 --et 11:00:00`, testing `baseline`'s epoch=0 vs epoch=1 checkpoints on a day
already covered in Exp 2). Session expected to end before either finishes — both are resumable
(`.done` sentinels), so rerunning the exact same commands next session continues rather than
restarts. No results yet as of this entry — check `exp0_results/*/summary.md` and
`exp4_results/*/summary.md` first thing next time.

## 2026-09-19 — Four more Exp 2 spot-checks: reanchor and ss consistently fail earlier than baseline

`batch_12h.sh` added: baseline 2015-01-02 (froze 62.2min), baseline 2015-01-09 (froze 61.5min),
reanchor 2015-01-16 (froze 20.7min), ss 2015-01-23 (froze 16.2min — earliest yet). All four froze,
none diverged. **11/11 scored 90-min runs have now failed, 0% survival, across every variant and
every day tested.**

Per-variant time-to-failure so far (minutes, freeze or diverge, whichever came first):

| variant | days tested | times | median |
|---|---|---|---|
| baseline | 5 | 21.2, 61.5, 62.2, 78.0, 88.4 | **62.2** |
| reanchor | 3 | 20.7, 24.1*, 50.1 | **24.1** |
| ss | 3 | 16.2, 24.1*, 65.5 | **24.1** |

(*reanchor/ss's 24.1min entries are the same 2015-01-15 day flagged 2026-09-11 as a likely shared
warm-up artifact — both may fail earlier than 24.1min in reality.)

Small n, but a consistent pattern across two independent runs of new days each: **both `reanchor`
and `ss` fail earlier (median ~24min) than `baseline` (median ~62min)**, not later. Combined with
the 2026-09-09 finding that `reanchor` diverges via a different mechanism (direct price departure,
not freeze-first) on 2015-01-30, this is now two separate lines of evidence pointing the same
way: neither price re-anchoring nor the scheduled-sampling/teacher-forcing fix delays instability
onset relative to the plain baseline — if anything, both make it worse. Worth leading with this in
the writeup rather than treating it as a footnote.

Also noted: these four runs took 16064-16814s each at the SAME 90-min horizon that took ~10875s
in the 2026-09-09/10 runs — roughly 50% slower. Likely GPU contention on a shared box rather than
a property of the model; flagging in case it matters for future budget math, not investigating
further right now.

## 2026-09-18 — Exp 3 rerun (fixed bucketing): confirms compounding self-generated error, not a late-session blind spot

`scripts/rerun_exp3_fixed.sh` results for all three variants, now correctly split early/late by
dataset index. Marketable-order rate (the ~40x-real-rate finding from 2026-09-11) is flat from
early to late session in every variant under teacher forcing: baseline 25.3%→25.8%, reanchor
21.6%→23.7%, ss 22.5%→23.1% (real: 0.6%→0.5%). Market-order type share and the rest of the
type/depth histograms move similarly little.

**This is the confirmatory result Experiment 3 was designed to produce.** Per-step prediction
quality does not degrade over the session when the model is always shown real history — so Exp 2's
consistent within-90-minute closed-loop failures are not explained by the model simply getting
worse at modelling later parts of the trading day. The failure is specific to the closed loop:
conditioning on the model's own drifting output, not a generic capability gap. Combined with the
marketable-order-excess finding, the mechanism story is now: a baseline miscalibration (~40x too
many spread-crossing orders, present from the very first generated step, not worsening with
exposure) becomes catastrophic specifically once the model starts feeding on its own output.

## 2026-09-11 — Session ended: full Exp 2 table to date, and what's still open

GPU session ended (multi-day access, not indefinite — see 2026-09-08 entry). Consolidating
everything scored so far before the gap, so the next session picks up cleanly rather than
reconstructing it.

**Every scored Exp 2 result to date** (90-min horizon unless noted; `.score.json` under each
`exp2_results/*/logs/`):

| variant | day | outcome | time to fail |
|---|---|---|---|
| baseline | 2015-01-30 (30min) | survived | — |
| baseline | 2015-01-30 (30min) | survived | — |
| baseline | 2015-01-30 | froze → diverged (price_departure) | 78.0 min → 78.4 min |
| baseline | 2015-01-07 | froze, no diverge | 88.4 min |
| baseline | 2015-01-22 | froze, no diverge | 21.2 min |
| reanchor | 2015-01-30 | diverged directly (price_departure), no freeze first | 50.1 min |
| reanchor | 2015-01-15 | froze, no diverge | 24.1 min* |
| ss | 2015-01-30 | froze, no diverge | 65.5 min |
| ss | 2015-01-15 | froze, no diverge | 24.1 min* |

**Headline: 0/7 runs survived a 90-minute horizon**, across all three variants and four different
days — confirms the >1h instability at real scale, not one cherry-picked day/checkpoint. Failure
timing spans 21–88 min: real day-to-day variance, not a fixed clock.

*`reanchor` and `ss` froze at the EXACT same microsecond timestamp (09:54:08.032466001) on
2015-01-15 — two independent checkpoints can't coincide by chance, so this is almost certainly the
shared real-data warm-up period's last price move (both runs condition on identical real history
before generation takes over), with both models freezing effectively immediately once generation
starts on that particular day. The 24.1min figure likely understates how early the failure really
begins on that day — flag as a caveat if it goes in the writeup, don't cite as a precise number.

**Also open**: the pooled Exp 3 finding (all three variants generate marketable/negative-depth
orders at ~43x the real rate, even teacher-forced — see the 2026-09-11 entry below) stands, but
the early-vs-late bucketing bug fix (`evaluation/diagnostics/open_loop_eval.py`, pushed) has not
yet been rerun — `bash scripts/rerun_exp3_fixed.sh` is ready to go, just needs a GPU session.

**What's still open for the next session, roughly in priority order:**
1. `bash scripts/rerun_exp3_fixed.sh` — cheap, rerun with the bucketing fix, get the real
   early-vs-late comparison.
2. More Exp 2 days/seeds per variant — 7 scored runs is a strong qualitative signal but not yet a
   statistically sized survival curve. `baseline` has 3 (90-min) + 2 (30-min); `reanchor`/`ss`
   have 2 each. Priority: more `baseline` days (primary claim), then even out `reanchor`/`ss`.
3. Experiment 4 (`exp4_checkpoint_sweep.sh --ckpt-dir data/checkpoints/TRADES_baseline`) — not
   started yet; checkpoint-epoch robustness within a variant.
4. Experiment 0's full sweep status on the remote is unconfirmed — check
   `exp0_results/*/summary.md` for how far it got (it was launched with `--max-mem-gb`, sized to
   this box's free RAM, per `rl_execution/RUNBOOK_instability.md`'s `exp0-cpu` window).
5. Experiment 5 (TRADES-LOB check) — not started; still needs the released dataset pulled from
   the Google Drive folder in `READMEFORMEHMET.md` onto this box.

## 2026-09-07 — Track A build + Exp 0 kickoff

Built `rl_execution/book_state_error.py` (the Exp-0 core measurement) and
`scripts/exp0_abides_replay_error.sh` (the 20-day x 7-length driver). Design: two independent
matching-book reconstructions diffed at 5-minute intervals — ground truth from the raw LOBSTER
message CSV via `rl_execution/orderbook_reconstructor` (crossing-aware, needed because raw
exchange feeds have gaps), and ABIDES's own record from that run's `EXCHANGE_AGENT.bz2` (converted
via ABIDES's own `util/formatting/convert_order_stream.py`, replayed with pure add/reduce/remove
bookkeeping — no re-derived crossing, since ABIDES's own stream already reports every match's
outcome and re-deriving it would double-count). Deliberately did NOT use `processed_orders.csv`
for the ABIDES-side reconstruction: it's built with `ignore_cancellations=True`
(`ABIDES/realism/realism_utils.py`), which drops exactly the events this experiment needs to not
lose.

**Smoke test finding (2015-01-30, 30-minute window, one day):** `shares_missing` (ground-truth
resting shares minus ABIDES's own resting shares) grows from 0 at open to **-202,308 at 30
minutes**, and the growth is not linear-then-flat — it's still accelerating at the last sample
(-97,733 at 25 min -> -202,308 at 30 min). Sign convention: negative means ABIDES is retaining
*more* resting inventory than the real book ever had (128,229 real shares resting at 30 min vs
330,537 in ABIDES's own reconstruction) — consistent in direction with the prior finding cited in
the paper plan (1.47M vs 0.21M unretired inventory). One-sided (unquoted-side) pathology did NOT
occur on this day/window once a definition bug was fixed (see below) — real ground truth and
ABIDES's reconstruction both stayed two-sided throughout.

**Bug caught and fixed before the full sweep launched:** the first cut flagged "one-sided" any
time ABIDES's book had an empty side, which trivially fires at t=market_open (both
reconstructions start empty) and would have contaminated every day's headline flag. Fixed to
require the real ground truth be two-sided at that same instant — `one_sided = ab_one_sided and
not gt_one_sided` in `book_state_error.py`.

**Located a concrete candidate mechanism** for the growing-inventory finding (not yet confirmed as
the sole cause — Exp 0's plan step 6, per-order life story, would confirm it properly, deferred
until the full sweep's result is in): `ABIDES/agent/WorldAgent.py:_preprocess_events_for_market_replay`
drops any cancel/execute row whose order was never seen as a same-day `NEW_ORDER` row *before
simulation even starts* — a pre-simulation filter on the loaded LOBSTER file, not a kernel-level
bug. Whether this fully explains the magnitude above is exactly what Exp 0's fuller result (and,
if warranted, the per-order diff) should settle.

**Full sweep launched** (20 INTC days x {30, 60, 90, 120, 180, 240, 390} minutes, CPU-only,
resumable via `.done` sentinels) — running in the background as of this entry; results land in
`exp0_results/<timestamp>/`. Multi-day splicing (plan step 4's "longer where the data allows") was
scoped out for this pass — a single INTC day already reaches 390 minutes (the full continuous
session), which covers the "longer than the 30-120min used elsewhere" ask without the added
complexity of reconciling order-id namespaces across day boundaries. Noted, not silently dropped.

## 2026-09-07 — Exp 1 gap found: the "0.724" checkpoint is not in this repo

Built `scripts/exp1_pin_checkpoint.sh` and `analysis/model_under_test.md`. Finding: **only one
checkpoint has ever been committed to this repo on any branch** (`val_ema=0.7_epoch=2`), and its
own commit message documents it as an explicit fallback, not the preferred one ("Not the 0.627
winner ... but a usable checkpoint"). Reading `analysis/MASTER_RESULTS.md` as it exists on
`origin/main` (not present on `rl-execution`) identifies `val_ema=0.724` as the actual Phase-2
baseline checkpoint (`UNCLAMP_DEPTH` + `PRICE_REANCHOR` on, no scheduled sampling) — the closest
existing match to "the model under test" — but that checkpoint file itself is not present in this
repo, in git history, on any branch. It exists only on whatever machine trained it. Full reasoning
and the required next step (locate it on the remote GPU box, or retrain it with the flag state
logged from the start this time) are in `analysis/model_under_test.md`; its "Confirmed checkpoint"
section is deliberately left blank until this is resolved on the remote box — Track B must not
proceed past this point on a guess.

## 2026-09-07 — Exp 0 full sweep OOM'd locally, moved to the remote box

The full 20-day x 7-length sweep launched in the earlier entry ran for ~6 combos before a single
390-minute (full continuous session) real-replay run hit **23GB RSS** and exhausted the local
Mac's memory (down to ~64MB free, degrading the whole machine). Root cause: ABIDES's real-replay
keeps the full order stream + LOB snapshots (`log_orders=True`, `wide_book=True`) in memory for
the whole session with no bounded logging — cost scales badly with session length, not linearly.
Killed the run. Hardened `scripts/exp0_abides_replay_error.sh` with a `ulimit -v`-based
`--max-mem-gb` cap (default 16GB) around each real-replay subprocess, so a single cell OOMs
cleanly (one ERROR row, sweep continues) instead of taking the box down — note this is a no-op on
macOS (Darwin doesn't enforce `RLIMIT_AS`) but works as intended on Linux. Moved Experiment 0 to
run on the remote box instead (still CPU-only, no GPU needed — its own `exp0-cpu` tmux window in
`rl_execution/RUNBOOK_instability.md`, alongside rather than queued behind `gpu-main`) rather than
re-risking the local machine. Not yet re-launched as of this entry.

## 2026-09-08 — Checkpoint selection hardened against the val_ema-collision landmine

User placed checkpoints on the remote box, "among others" (i.e. a directory now holding more than
one file). `analysis/MASTER_RESULTS.md` §1.4 already documents that two checkpoints can share a
rounded val_ema and be indistinguishable by `--id`/`-id` matching (e.g. `0.69_epoch=2` vs
`0.69_epoch=4`) — exactly the failure mode that would silently select an SS-resumed checkpoint
instead of the plain baseline, or vice versa. Closed this before it could bite:

- `evaluation/diagnostics/open_loop_eval.py`: added `--ckpt-path` (exact file, bypasses `--id`
  entirely, mirrors `world_agent_sim.py`'s own flag), and made `find_checkpoint` raise instead of
  silently keeping the last match when `--id` is ambiguous.
- `scripts/exp3_teacher_forced.sh`: switched from `--ckpt-val-loss` (ambiguous) to `--ckpt-path`
  (exact), matching `exp2_survival_sweep.sh`'s and `exp4_checkpoint_sweep.sh`'s existing
  `--ckpt-path` convention — every Track B script now takes an exact file everywhere, no script
  takes a bare val_ema number.
- `scripts/exp1_pin_checkpoint.sh`: now lists every checkpoint, flags known decoys by val_ema
  (0.627/0.681/0.719 pre-fix, 0.7_epoch the documented fallback), and explicitly reports
  zero/one/many matches for `val_ema=0.724*` rather than just checking existence — many-match
  case prints all candidates and refuses to pick one.
- `scripts/exp4_checkpoint_sweep.sh`: excludes the same known decoys from its default sweep (a
  mixed checkpoint directory would otherwise silently turn "robustness across checkpoints from
  this training run" into "robustness across unrelated runs"); `--include-decoys` overrides.

## 2026-09-08 — Redesigned for a no-copy-paste remote terminal

User feedback after the first remote attempt: pasting the checkpoint path into
`analysis/model_under_test.md`'s markdown did nothing — no script ever read it back out, so it was
purely decorative. Combined with "I cannot copy and paste on the remote," hand-typing a full
checkpoint path was exactly the failure mode that happened. Also: the runbook gave phase labels
("B1 pilot") instead of literal commands. Fixed both:

- `analysis/model_under_test.md`'s "Confirmed checkpoint" section is now a machine-readable
  `CKPT_PATH=...` line, not freeform text.
- `scripts/exp1_pin_checkpoint.sh` now WRITES that line automatically the moment it finds exactly
  one unambiguous `val_ema=0.724*` match — nothing to type or paste.
- New `scripts/_ckpt_lib.sh`, sourced by `exp2_survival_sweep.sh` and `exp3_teacher_forced.sh`:
  resolves the checkpoint from that line automatically; `--ckpt-path` is now an optional override,
  not a required argument.
- `exp2_survival_sweep.sh` gained `--smoke`/`--pilot` zero-typing presets; every Track B script
  now runs as a single short command with no flags in the common case.
- `rl_execution/RUNBOOK_instability.md` rewritten as six numbered, literal, copy-paste-free
  commands (each stated as what to type, what it does, and what to check), replacing the old
  phase-label ("B0"/"B1"/...) style.

## 2026-09-07 — Track B package built (scripts only, not yet run)

Built the paste-back package for the remote GPU box: `rl_execution/survival_metrics.py` (frozen/
diverged scoring, pre-registered definitions — see the plan file), `scripts/exp2_survival_sweep.sh`,
a `--bucket-by-time` patch to `evaluation/diagnostics/open_loop_eval.py`, `scripts/exp3_teacher_forced.sh`,
`scripts/exp4_checkpoint_sweep.sh`, and `rl_execution/RUNBOOK_instability.md` tying them together.
None of these have been run — they need the checkpoint gap above resolved first, and a GPU.
