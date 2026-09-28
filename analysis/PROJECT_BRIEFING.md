# Instability paper — complete project briefing

*Written 2026-09-28, for two audiences at once: Matthew consulting this after a month away, and
any other AI session (Claude or otherwise) picking this project up cold. Nothing is summarized
away — where a number exists, it's here. The chronological version of this same material, with
the reasoning behind each decision as it happened, is `analysis/instability_paper_log.md`; this
document is the same content reorganized for reference rather than narrative. If the two ever
disagree, the log is closer to ground truth (it was written live); this file was assembled from
it plus direct verification against the repo, so treat any conflict as a bug in this file.*

---

## 1. The claim this paper is testing

**Not** "we made TRADES fast" or "we made TRADES stable." The claim: **the published TRADES
model — the full 100-step DDPM sampler, the configuration the authors actually recommend — breaks
down over realistic run lengths, and this is a property of the model, not an artifact of this
reproduction or of the ABIDES exchange simulator it runs inside.**

Explicitly out of scope: the one-step sampler (its failure is already explained — it discards all
sampling noise by construction) and the ten-step "repaired" scheduled-sampling version (that's
*our* fix, not the thing being diagnosed). The original plan document is
`/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md`.

**Headline result, in one line:** on INTC, every single 90-minute run across three independently
trained variants of the model failed (0/11). On TSLA, some survive (2/11), and Experiment 3
supplies a mechanistic reason why. Neither of the model's own stability interventions — price
re-anchoring or scheduled sampling — delays failure relative to the plain baseline; price
re-anchoring, if anything, fails faster.

---

## 2. Scope — exactly what "the model under test" means

Full detail in `analysis/model_under_test.md`. Summary:

**Training-time flags (baked into the checkpoint, unrecoverable from the file afterward —
`configuration.py` never persists them into the saved `hyper_parameters`):**

| flag | wanted state | what it does |
|---|---|---|
| `UNCLAMP_DEPTH` | **True** | keeps signed (marketable) depth targets instead of clamping at 0 — the "remove the clamp" data-pipeline fix |
| depth-index fix | **True** (unconditional on this branch, no flag) | `utils/utils_data.py` uses the pre-event orderbook snapshot (`index = j - 1`) for every event type, not just cancels — fixes a self-referential bug that made `UNCLAMP_DEPTH` a no-op on the original code |
| `SCHEDULED_SAMPLING` | **False** for `baseline`/`reanchor`, **True** for `ss` | this is "our fix" (10-step repaired model) — explicitly the *subject* of one variant, not baked into the other two |
| `PRICE_REANCHOR` | **False** for `baseline`/`ss`, **True** for `reanchor` | anchors prices to each day's opening mid instead of a global z-score — the open question this paper resolves empirically rather than assuming |

**Simulate-time flags (CLI, independent of the checkpoint):** `-type DDPM -nsteps 100`, and
explicitly **no** decode-time stability interventions (`--type-decode`, `--depth-noise`,
`--size-reshape`, `--depth-reshape`, `--book-target-thick`/`--book-cancel-rate`, `--cond-clip`,
`--flow-balance`, `--depth-drift`, `--cancel-boost`, `--dn-target-exec`, non-default
`--guidance-scale`/`--churn-*`). All of these are the *other* things this project has built to fix
the instability — this paper is specifically about the model *without* any of them.

**Why fresh checkpoints, not the paper-authors' or this project's earlier ones:** the obvious
candidate (`val_ema=0.724`, referenced in `analysis/MASTER_RESULTS.md` on `origin/main`) was never
actually present with verifiable training-flag provenance on any box this project used —
`configuration.py` doesn't record flag state into checkpoints, and pre-existing checkpoints found
on the GPU box (`0.69`, `0.7`, etc.) had exactly the same problem. Six checkpoints were trained
fresh instead, with flag state logged at the moment of training (§5 below has every checkpoint's
exact path and flags).

---

## 3. Repository architecture

### 3.1 Directory map (this project's additions, within the wider TRADES/ABIDES fork)

```
analysis/
  instability_paper_log.md      running log, chronological, the ground-truth narrative
  model_under_test.md           flag definitions + the six CKPT_PATH_<VARIANT> pointers
  PROJECT_BRIEFING.md           this file
  MASTER_RESULTS.md             pre-existing, origin/main only (not this branch) — earlier
                                 project's own consolidated numbers, cited for checkpoint history

rl_execution/
  book_state_error.py           Exp 0's core measurement (see §6.1)
  survival_metrics.py           Exp 2/4's freeze/diverge scoring (pre-registered definitions)
  orderbook_reconstructor.py    pre-existing, reused unmodified — ground-truth LOBSTER book builder
  RUNBOOK_instability.md        the remote GPU-box runbook, numbered literal commands

evaluation/
  diagnostics/open_loop_eval.py Exp 3's teacher-forced sampler, patched with --bucket-by-time
                                 and --ckpt-path
  lob_bench/tradeslob_warning_signs.py   Exp 5's script (built, never run — needs the dataset)

scripts/
  exp0_abides_replay_error.sh   Exp 0 driver (20 days x 7 lengths x {INTC,TSLA})
  exp1_pin_checkpoint.sh        legacy single-checkpoint pinning (superseded by variant training)
  exp2_survival_sweep.sh        Exp 2 driver — the main survival test
  exp3_teacher_forced.sh        Exp 3 driver
  exp4_checkpoint_sweep.sh      Exp 4 driver — checkpoint/epoch robustness
  train_variant.sh              trains ONE variant (baseline/reanchor/ss, optionally tsla_ prefixed)
  train_three_variants.sh       chains baseline+reanchor+ss for one stock, unattended
  pin_trained_variant.sh        writes CKPT_PATH_<VARIANT> after training (see the §5 caveat)
  session3_batch.sh             the full third-session plan, chained: INTC Exp4 + all 3 TSLA variants
  batch_12h.sh, overnight_batch.sh, rerun_exp3_fixed.sh   smaller one-off chained batches
  rescore.sh                    rescans generated runs for a missing .score.json, no GPU needed
  backup_before_gap.sh          the checkpoint/result-data git backup script (see §7 bug list)
  _ckpt_lib.sh                  shared bash functions: resolve_ckpt_path_for_variant,
                                 reanchor_for_variant, ticker_for_variant

configuration.py                +MAX_EPOCHS_OVERRIDE, +STOCK_OVERRIDE (file/env, additive)
constants.py                    pre-existing UNCLAMP_DEPTH/PRICE_REANCHOR/SCHEDULED_SAMPLING flags
```

### 3.2 The variant system

A "variant" is a string like `baseline`, `reanchor`, `ss`, `tsla_baseline`, `tsla_reanchor`,
`tsla_ss` — the `tsla_` prefix is the *only* thing that switches stock; everything else about a
script's behavior derives from the base name. `_ckpt_lib.sh` provides:

- `resolve_ckpt_path_for_variant <variant>` — reads `CKPT_PATH_<VARIANT>=` (uppercased) from
  `analysis/model_under_test.md`.
- `reanchor_for_variant <variant>` — `on` if the name contains `reanchor`, else `off`. Forces the
  simulate-time `PRICE_REANCHOR_FLAG` to match how the checkpoint was *trained* — this is not an
  independent per-run choice, since re-anchoring changes training conditioning too, not just
  simulate-time behavior.
- `ticker_for_variant <variant>` — `TSLA` if the name starts with `tsla_`, else `INTC`.

`train_variant.sh <variant>` sets the training-time flags (`UNCLAMP_DEPTH_FLAG` always on,
`PRICE_REANCHOR_FLAG`/`SCHEDULED_SAMPLING_FLAG` per the table in §2, `STOCK_OVERRIDE` for TSLA
variants), caps training at 2 epochs via `MAX_EPOCHS_OVERRIDE`, keeps every epoch's checkpoint
(`KEEP_EPOCH_CHECKPOINTS_FLAG`), and moves everything it produces into its own
`data/checkpoints/TRADES_<variant>/` directory — necessary because all variants share identical
seed/hyperparameters and would otherwise collide on filename in the shared
`data/checkpoints/TRADES/`. `pin_trained_variant.sh <variant>` then picks the highest epoch
produced and is *meant* to write `CKPT_PATH_<VARIANT>=` into `model_under_test.md` — but see the
§7 bug entry: on this project, that step's output was never actually committed to git.

### 3.3 The remote workflow

GPU work happens on a remote box this session has no direct access to (no SSH tool, no shared
filesystem) — everything is **paste-back**: scripts are written and pushed from a local session,
the user runs them on the remote and pastes output back. Consequences baked into every script:

- Every long-running script is written to be launched with
  `nohup bash scripts/X.sh > X.log 2>&1 & disown` and checked with `tail -f X.log` — survives a
  dropped connection.
- Everything is resumable via fixed `--out-dir` + `.done` sentinel files (`exp2_survival_sweep.sh`,
  `exp4_checkpoint_sweep.sh`) or an explicit "already pinned/already has output" skip check
  (training, Exp 3) — rerunning the exact same command after an interruption continues rather than
  restarts.
- The remote terminal has **no copy/paste** — every command handed to the user is a short, exact,
  zero-or-near-zero-argument string. Nothing is ever a template with a placeholder to fill in (one
  instance of this rule being broken — `--max-mem-gb N` sent as literal text — cost an entire failed
  Exp 0 run; see §7).
- `data/` (and therefore `data/checkpoints/`) is gitignored wholesale, deliberately, because it
  also covers licensed LOBSTER market data that must never be committed. Anything under
  `data/checkpoints/` needs `git add -f`, always — a plain `git add` fails silently with only a
  warning. See §7 for the incident this caused.

---

## 4. GPU cost model (for budgeting future sessions)

Measured directly, not estimated:

| horizon | measured wall-clock | notes |
|---|---|---|
| 30 min | ~2200–2230s (~37 min) | the `--smoke` preset |
| 90 min | ~10,875–16,900s (~3–4.7h) | the standard Exp 2/4 horizon; later runs ran ~50% slower than earlier ones, likely GPU contention on a shared box, not investigated further |
| 2h | >18,000s (>5h) | the originally-planned pilot horizon — this single data point is why the plan was resized down to 90 min |

Cost scales roughly as `horizon^1.5`, not linearly — a 90-minute run is not "3x a 30-minute run,"
it's closer to 5x. Training one variant (2 epochs) takes ~1h10min for `baseline`/`reanchor`, but
~5h for `ss` (scheduled sampling's rollout sampling during training is expensive — its own
training log prints a warning about this).

**Important, separate finding — do not treat "seed" as reproducible on this hardware.** An
identical (checkpoint, day, seed) configuration produced two different failure times (25.3 min and
62.2 min) across two separate runs. `torch.manual_seed(seed)` *is* called
(`ABIDES/config/world_agent_sim.py:231`), but `main.py`'s `set_torch()` enables
`cudnn.allow_tf32`/`cuda.matmul.allow_tf32` with no `cudnn.deterministic` or
`use_deterministic_algorithms` set — the standard cause of GPU non-reproducibility, compounded
over 100 diffusion steps × tens of thousands of sequential orders into materially different
trajectories. Practical upshot: every nominally-repeated run in this project's data is actually an
independent stochastic draw, not a duplicate — the effective sample size is larger than the
distinct-seed count suggests, but no specific "seed 30 result" should be described as
reproducible.

---

## 5. Trained checkpoints — the complete inventory

All ten checkpoints (six variants × up to two epochs each) are on `origin/rl-execution`, real
content verified (not LFS pointer stubs) as of 2026-09-27. `analysis/model_under_test.md`'s
`CKPT_PATH_*` lines (reconstructed 2026-09-28, see that file's own note) point at the ones actually
used throughout every experiment below.

| variant | stock | epoch | val_ema | file | used as CKPT_PATH? |
|---|---|---|---|---|---|
| baseline | INTC | 1 | 0.702 | `data/checkpoints/TRADES_baseline/val_ema=0.702_epoch=1_INTC_se_256_au_64_CD_8_seed_30.ckpt` | **yes** |
| baseline | INTC | 0 | 0.723 | `data/checkpoints/TRADES_baseline/val_ema=0.723_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt` | no (used in Exp 4 robustness check only) |
| reanchor | INTC | 1 | 0.704 | `data/checkpoints/TRADES_reanchor/val_ema=0.704_epoch=1_INTC_se_256_au_64_CD_8_seed_30.ckpt` | **yes** (only epoch produced) |
| ss | INTC | 0 | 0.757 | `data/checkpoints/TRADES_ss/val_ema=0.757_epoch=0_INTC_se_256_au_64_CD_8_seed_30.ckpt` | **yes** (only epoch produced — training hit its epoch-2 target but this repo's convention is "epoch 1 at most epoch 2," 1-indexed, so epoch 0 alone still satisfies it) |
| tsla_baseline | TSLA | 1 | 0.809 | `data/checkpoints/TRADES_tsla_baseline/val_ema=0.809_epoch=1_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | **yes** |
| tsla_baseline | TSLA | 0 | 0.838 | `data/checkpoints/TRADES_tsla_baseline/val_ema=0.838_epoch=0_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | no |
| tsla_reanchor | TSLA | 1 | 0.807 | `data/checkpoints/TRADES_tsla_reanchor/val_ema=0.807_epoch=1_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | **yes** |
| tsla_reanchor | TSLA | 0 | 0.837 | `data/checkpoints/TRADES_tsla_reanchor/val_ema=0.837_epoch=0_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | no |
| tsla_ss | TSLA | 1 | 0.824 | `data/checkpoints/TRADES_tsla_ss/val_ema=0.824_epoch=1_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | **yes** |
| tsla_ss | TSLA | 0 | 0.843 | `data/checkpoints/TRADES_tsla_ss/val_ema=0.843_epoch=0_TSLA_se_256_au_64_CD_8_seed_30.ckpt` | no |

There is also one pre-existing, unrelated checkpoint, `data/checkpoints/TRADES/val_ema=0.7_epoch=2_INTC_se_256_au_64_CD_8_seed_30.ckpt` — the "documented fallback" from before this paper's work started (commit `b0f449c`). Not used anywhere in this paper's results; mentioned only so it isn't confused for one of the six variants above.

---

## 6. Complete experiment results

### 6.1 Experiment 0 — is the failure the model, or the ABIDES engine underneath it?

**Method:** replay real, historical order flow (no generative model at all) through ABIDES via
`world_agent_sim` with the diffusion flag off, at 7 session lengths (30/60/90/120/180/240/390 min)
per day, for every available trading day, both stocks. Measured every 5 minutes: shares missing
between an independent ground-truth book reconstruction (`orderbook_reconstructor.py`, built from
the raw LOBSTER message file) and ABIDES's own record (its `EXCHANGE_AGENT.bz2` stream, replayed
with pure add/reduce/remove bookkeeping, deliberately *not* using `processed_orders.csv` since that
file is built with `ignore_cancellations=True` and drops exactly the events this check needs).
CPU-only, no GPU, runs independently of everything else.

**INTC:** 131 of 140 day×length cells scored (9 errors, all at the 390-min/full-session length —
either "real replay failed" or "book_state_error.py failed," not investigated further, doesn't
threaten the result since 131/140 succeeded, including 11/20 days' full 390-min cells).

**TSLA:** 140 of 140 day×length cells scored, no errors (the *first* TSLA attempt failed instantly
on all 140 from a paste-back placeholder bug — see §7 — the rerun was clean).

**Headline, both stocks: `one_sided_ever = False` on every single one of 271 scored cells.** Real
order flow through ABIDES never once produces the "one side of the book drains to nothing"
pathology, from 30 minutes up to a full 6.5-hour session. This is the number the whole plan was
gated on — it means the pathology studied in Experiments 2–4 is attributable to the model, not to
ABIDES.

**A separate, real finding, not folded into the above:** `shares_missing` (ABIDES's own resting
inventory minus ground truth) is consistently negative and grows with session length on every
single day tested, both stocks — e.g. INTC 2015-01-02: −62,517 at 30 min → −704,400 at 390 min.
ABIDES retains more resting inventory than the real record ever had, and the gap does not
plateau. `depth_error` (top-10-level-only) stays comparatively small and mixed-sign throughout,
suggesting the excess inventory sits away from the touch — plausibly why it never manifests as a
one-sided top-of-book. A candidate mechanism was located but not confirmed:
`ABIDES/agent/WorldAgent.py:_preprocess_events_for_market_replay` drops any cancel/execute row
whose order was never seen as a same-day `NEW_ORDER` row *before simulation starts* — a
preprocessing filter, not a live kernel bug.

Full per-day, per-length numbers: `exp0_results/*/summary.md` and the per-cell CSVs under
`exp0_results/*/csv/<day>/<length>min.csv`.

### 6.2 Experiment 1 — checkpoint provenance

Not a numeric result — see §2 and §5 above. Summary of what happened: the obvious pre-existing
checkpoint had no verifiable training-flag history, so six were trained fresh with flag state
logged at training time instead of trying to forensically reconstruct an old file's history.

### 6.3 Experiment 2 — the main survival test

**Method:** each variant run for 90 minutes (past the ~1-hour point where instability is known to
set in) on multiple independent trading days, scored against freeze/diverge definitions
pre-registered *before* any result was seen (`rl_execution/survival_metrics.py`):

- **Frozen**: no new unique mid-price level visited within a trailing 10-minute window.
- **Diverged**: either (a) one side's top-of-book size is 0 for a continuous ≥60s span, or
  (b) mid price departs from the run's own arrival mid by more than 3× the real day's own
  high–low range.
- A run hitting neither by the end of its horizon counts as **survived** — a real data point, not
  a discard.

**Every scored 90-minute run, INTC:**

| variant | day | outcome | time to fail (min) |
|---|---|---|---|
| baseline | 2015-01-30 | froze, then diverged 25s later | 78.0 → 78.4 |
| baseline | 2015-01-07 | froze, no diverge | 88.4 |
| baseline | 2015-01-22 | froze, no diverge | 21.2 |
| baseline | 2015-01-02 | froze, no diverge | 62.2 |
| baseline | 2015-01-09 | froze, no diverge | 61.5 |
| reanchor | 2015-01-30 | diverged directly (price departure), no freeze first | 50.1 |
| reanchor | 2015-01-15 | froze, no diverge (⚠ see caveat below) | 24.1 |
| reanchor | 2015-01-16 | froze, no diverge | 20.7 |
| ss | 2015-01-30 | froze, no diverge | 65.5 |
| ss | 2015-01-15 | froze, no diverge (⚠ see caveat below) | 24.1 |
| ss | 2015-01-23 | froze, no diverge | 16.2 |

**INTC totals: baseline 0/5, reanchor 0/3, ss 0/3 — 0/11 overall (0%).**

⚠ **Caveat on the two `24.1 min` entries:** `reanchor` and `ss` froze at the *exact same
microsecond timestamp* (09:54:08.032466001) on 2015-01-15. Two independent checkpoints cannot
coincide by chance — this is almost certainly the shared real-data warm-up period's last price
move (both runs condition on identical real history before generation takes over), with both
models freezing effectively immediately once generation starts that day. The 24.1 min figure
likely understates how early the failure really begins on that specific day — don't cite it as a
precise per-model number, the *direction* (both fail, early) still holds.

**Every scored 90-minute run, TSLA:**

| variant | day | outcome |
|---|---|---|
| baseline | 2015-01-02 | failed |
| baseline | 2015-01-07 | failed |
| baseline | 2015-01-15 | failed |
| baseline | 2015-01-22 | failed |
| baseline | 2015-01-30 | **survived** |
| reanchor | 2015-01-05 | failed |
| reanchor | 2015-01-09 | failed |
| reanchor | 2015-01-16 | **survived** |
| reanchor | 2015-01-23 | failed |
| ss | 2015-01-06 | failed |
| ss | 2015-01-13 | failed |

**TSLA totals: baseline 1/5, reanchor 1/4, ss 0/2 — 2/11 overall (18%).**

**30-minute smoke checks** (sanity-only, not part of the 90-min comparison above): every variant,
both stocks, survived cleanly at 30 minutes — 2/2 seeds for INTC baseline, 1/1 for every other
variant/stock combination tested. Confirms the checkpoints load and behave normally; the failure
is specifically a >1-hour phenomenon, consistent with prior knowledge going into this paper.

**Reading the pattern:**
1. **INTC fails 100% of the time at 90 minutes, across all three variants.** No exceptions.
2. **`reanchor` fails earliest on INTC** (median ~24 min vs baseline's ~62 min), and via a
   *different* mechanism on its one non-warm-up-confounded data point (direct divergence, not
   freeze-then-diverge). Price re-anchoring was introduced specifically to fix an
   out-of-distribution price-drift problem — failing *faster* than the plain baseline is the
   opposite of the naive expectation.
3. **TSLA survives meaningfully more often** for `baseline` and `reanchor` (both ~20-25%), while
   `ss` stays at 0% on both stocks — the one place the finding is stock-independent.
4. Sample sizes are small (2–5 days per cell) — the *direction* of every pattern above is
   consistent across independent variant/stock combinations, but exact percentages should be read
   as indicative pending more days, not final.

Full data: `exp2_results/*/summary.md` (human-readable) and `exp2_results/*/logs/*.score.json`
(machine-readable, one per run — `time_to_freeze`, `time_to_diverge`, `diverge_reason`,
`survived_to_end`, `real_day_range`, `n_rows`, `last_timestamp`).

### 6.4 Experiment 3 — what's actually driving the collapse

**Method:** `evaluation/diagnostics/open_loop_eval.py` — sample the model on real conditioning
windows with real history fed back at every step (never its own output), no ABIDES loop at all.
Compares generated vs. real next-event distributions, pooled and split early-vs-late session by
dataset index.

**⚠ One bug, caught and fixed mid-project:** the original `--bucket-by-time` patch split
early/late on the model's own `"time"` field — which, per `utils/utils_data.py`'s preprocessing,
is the *inter-arrival delta* to the previous event, not session clock time. The split was
therefore separating "fast gaps" from "slow gaps," not early from late session (tell: the reported
split point was ~5×10⁻⁵, an interarrival-sized number, not a plausible session timestamp). Fixed
to split on the sampled windows' dataset index (chronologically ordered) instead. **Any Exp 3 JSON
file with `bucket_split_time` in it (not `bucket_split_index`) has invalid early/late numbers —
only its pooled numbers are trustworthy.** Post-fix files are under `exp3_results/session3_*` and
`exp3_results/fixed_*`.

**Pooled marketable-order (spread-crossing / negative-depth) rate, real vs. generated:**

| | real INTC | generated INTC | real TSLA | generated TSLA |
|---|---|---|---|---|
| marketable order share | 0.56% | ≈ 22.8–25.6% (prior decode: 22.8/22.6/24.4/24.3/23.7% across the five post-fix runs) | 6.3% | ≈ 21.4–22.1% |
| excess over real | — | **≈ 40×** | — | **≈ 3.5×** |

**Early vs. late session (post-fix, correctly bucketed by dataset index), INTC, all three
variants:** flat. `baseline` marketable rate 25.3%→25.8%, `reanchor` 21.6%→23.7%, `ss` 22.5%→23.1%
(real: 0.6%→0.5%). Type histograms and the rest move similarly little. TSLA's post-fix early/late
numbers (from the `session3_*` runs) show the same pattern — roughly flat, no systematic
late-session degradation.

**What this establishes:** per-step prediction quality, given real history throughout, does not
degrade over the course of a session on any variant, either stock. Closed-loop simulation still
collapses reliably. That isolates the mechanism to the closed loop specifically — the model
conditioning on its *own* drifting output — rather than a generic inability to model later parts
of the trading day.

**Why this also explains TSLA's better survival rate:** the model's *generated* marketable-order
rate is similar in absolute terms across both stocks (~21–26%), while the *real* rate differs by
11× between stocks (0.56% vs 6.3%). If the generated rate reflects a roughly fixed model-intrinsic
miscalibration rather than something that scales with the real distribution, TSLA's real trading
pattern already sits closer to what the model naturally produces — a smaller "shock" to the
closed feedback loop, plausibly explaining the higher TSLA survival rate found in Experiment 2.
This ties the Exp 2 and Exp 3 findings into one mechanism.

Full data: `exp3_results/*/open_loop_*.json` (structured) and `.txt` (human-readable, includes
sampling progress).

### 6.5 Experiment 4 — is this one unlucky checkpoint?

**Method:** every saved epoch of a variant, run independently, scored the same way as Exp 2.

**INTC baseline, both epochs, two independent days:**

| checkpoint | day | outcome |
|---|---|---|
| epoch=1 (val_ema=0.702) | 2015-01-02 | failed (froze 25.3 min) |
| epoch=0 (val_ema=0.723) | 2015-01-02 | failed (froze 88.2 min) |
| epoch=1 (val_ema=0.702) | 2015-01-16 | failed |
| epoch=0 (val_ema=0.723) | 2015-01-16 | failed |

**4/4 failed.** Not one unlucky snapshot — both epochs, on two separate days, all fail. (The
epoch=1/2015-01-02 result here is also the pair used to discover the GPU non-determinism finding
in §4 — the same config produced a different failure time, 25.3 min, than an earlier independent
run of it, 62.2 min.)

TSLA and the other two INTC variants were not run through Exp 4 (time-budget priority went to
widening Exp 2's TSLA coverage instead — see §8).

Full data: `exp4_results/*/summary.md`.

### 6.6 Experiment 5 — not run

Checks the TRADES authors' own released `TRADES-LOB` benchmark file for the same early-warning
trend (growing imbalance, shrinking marketable-order share, slowing price-level movement) within
their own published 30-minute evaluation window. The script
(`evaluation/lob_bench/tradeslob_warning_signs.py`) is built and was smoke-tested against a
stand-in file — never run against the real dataset, which needs manually pulling from the Google
Drive folder referenced in `READMEFORMEHMET.md` (same folder the original TRADES checkpoints came
from) and placing at `data/TRADES-LOB/`.

---

## 7. Bugs encountered and fixed (chronological, full detail in the log)

Kept in full because several of these are the kind of thing that silently corrupts a result if
not caught — worth knowing about even after they're fixed, in case a symptom recurs.

1. **Exp 0 OOM'd the local Mac.** ABIDES's real-replay keeps the whole order stream + LOB
   snapshots in memory with no bounded logging; a single 390-min replay hit 23GB RSS. Fixed with a
   `ulimit -v`-based `--max-mem-gb` cap (no-op on macOS, works on Linux) and moved Exp 0 to the
   remote box entirely.
2. **The `--bucket-by-time` inter-arrival-delta bug** — see §6.4. Caught by an implausible split
   value before being trusted, not by inspection of the code.
3. **Checkpoint val_ema collision risk.** Two checkpoints can round to the same `val_ema` and be
   indistinguishable by `--id`/`-id` matching (documented precedent in `MASTER_RESULTS.md` §1.4).
   Every script now takes an exact `--ckpt-path`, never a bare val_ema number.
4. **A markdown-insertion regex bug** in the first version of the checkpoint-pinning scripts used
   `re.DOTALL`/"replace to end of file" logic that would have silently deleted unrelated content
   appended later in `model_under_test.md`. Caught by testing locally with placeholder checkpoints
   before shipping; rewritten as scoped, line-level edits.
5. **A literal placeholder sent as a real argument.** `--max-mem-gb N` was meant as "replace N with
   a number" but got run literally, failing all 140 Exp 0 TSLA cells instantly with
   `N: unbound variable`. Nothing was wasted (failed before touching ABIDES); reinforced the
   "never send a template, always send the exact command" rule for this remote.
6. **A bash `local` cross-reference bug**: `local V="$1" OUT="...${V}..."` fails under `set -u`
   because bash expands every word on a `local` line — including `${V}` in the second assignment —
   before any declaration takes effect, so `${V}` looks up a `V` that doesn't exist yet in that
   scope. Crashed `session3_batch.sh` mid-run. Fixed by splitting into two `local` statements;
   grepped every script in `scripts/` and confirmed this was the only instance.
7. **The big one — checkpoints silently never backed up.** `data/` is gitignored wholesale
   (deliberately, it also covers licensed LOBSTER data that must never be committed). A plain
   `git add data/checkpoints/...` in `backup_before_gap.sh` silently skipped everything with only a
   warning, not an error — the backup commit "succeeded" and contained zero checkpoints. Not
   discovered until after the remote session was already gone, at which point GitHub push auth was
   *also* broken on that box (a stale VS Code credential-relay socket). Recovered by routing around
   both problems: `git bundle` for commit history + a manual `tar` of `.git/lfs/objects` for the
   actual LFS content (bundles don't carry LFS blobs), both transferred off the box by hand and
   merged from a different machine. Took several rounds (one merge attempt left orphaned partial
   files from an earlier failed LFS smudge; one file transfer silently returned a stale cached copy
   twice). All ten checkpoints recovered and verified. `backup_before_gap.sh` now uses `git add -f`
   and refuses to proceed if on-disk and staged checkpoint counts disagree.
8. **`model_under_test.md`'s `CKPT_PATH_*` lines were never committed at all**, discovered
   2026-09-28 while writing this document — `pin_trained_variant.sh` only ever wrote the local
   file, never committed it, and `backup_before_gap.sh` never staged that file either. Reconstructed
   from the checkpoint filenames actually present in git plus every paste-back in the log (§5 above
   has the result); fixed in the same commit as this file.

---

## 8. What's still open

1. **Sample sizes.** Most cells are 2–5 days; `tsla_ss` is only 2. Enough for a clear qualitative
   pattern, not a statistically sized survival curve. Priority for more runs: even out `tsla_ss`
   and `tsla_reanchor` first (thinnest cells), then widen everything for a real curve with error
   bars.
2. **Experiment 5** not run — needs the TRADES-LOB dataset pulled in manually (no GPU needed once
   it's there).
3. **Exp 4** only covers INTC `baseline`. `reanchor`/`ss` and all of TSLA are untested for
   checkpoint-epoch robustness.
4. **ABIDES's own inventory leak** (§6.1) — the mechanism was hypothesized but never confirmed via
   the plan's own suggested per-order life-story diff. Not blocking, but a real loose end if time
   allows.
5. **No figures built yet.** Everything above is tables and prose; the actual survival-curve and
   depth-histogram charts for the write-up don't exist as images/plots yet — only as an SVG bar
   chart in the artifact memo already sent to the supervisor
   (`https://claude.ai/code/artifact/9ce01b63-ac26-4913-b6ff-8ea761855391`).
6. **Stretch experiments from the original plan**, lowest priority, not started: a matched
   real-vs-generated-flow comparison through ABIDES (partially answered already by combining §6.1
   and §6.3), and a sensitivity/nudge test (do two runs differing by one tiny early perturbation
   diverge from each other, quantifying "small errors compound" directly).

---

## 9. How to resume

```
git pull
cat analysis/instability_paper_log.md        # the narrative, if context is needed
cat analysis/PROJECT_BRIEFING.md              # this file
```

On the remote GPU box (paste-back, no copy/paste — every command below is meant to be short and
typed exactly): `rl_execution/RUNBOOK_instability.md` has the full numbered sequence. In short,
for more coverage of an existing variant:

```
bash scripts/exp2_survival_sweep.sh --days <one new day, e.g. 20150113> --seeds 30 --et 11:00:00 --variant <variant>
```

For Exp 4 on an untested variant:

```
bash scripts/exp4_checkpoint_sweep.sh --ckpt-dir data/checkpoints/TRADES_<variant> --day 20150102 --seeds 30 --et 11:00:00
```

For Exp 5, once the dataset is placed at `data/TRADES-LOB/`:

```
python -m evaluation.lob_bench.tradeslob_warning_signs data/TRADES-LOB/<file>.csv --bucket-min 5
```

**Before ending any future remote session again:** `bash scripts/backup_before_gap.sh` — it now
correctly force-adds checkpoints and will refuse to proceed silently if the same failure mode from
§7 bug 7 ever recurs.
