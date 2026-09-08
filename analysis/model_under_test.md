# Experiment 1 — the exact model under test

*Written 2026-09-07, as part of the instability-paper execution plan
(`/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md`). Every later experiment (2, 3, 4)
must cite this file verbatim rather than re-deciding any of it.*

## What "the model under test" means for this paper

Per the paper's own scope: DDPM, 100 steps, trained with the two data-pipeline corrections
**only** — no decode-time corrections, no scheduled sampling. Two things determine this:

**Training-time (baked into the checkpoint, not recoverable from the file afterward —
`configuration.py` does not persist these flags into the saved `hyper_parameters.config`):**

| flag | wanted | source |
|---|---|---|
| `UNCLAMP_DEPTH` | **True** (file `UNCLAMP_DEPTH_FLAG` present at train time) | `constants.py:17` — the "remove the clamp" data fix |
| `DEPTH_INDEX_FIX` | **True** (default) | `constants.py`, added in `f9edaa3` — **not merged into this branch** (`git branch --contains f9edaa3` excludes `rl-execution`); on this branch the fix is simply unconditional (`utils/utils_data.py` uses `index = j - 1` always, no flag at all). Either way the fix is active here; the toggle only matters on whichever branch runs the intentional "no-fixes baseline" comparison. |
| `SCHEDULED_SAMPLING` | **False** (default) | `constants.py` — this is "our fix" (the repaired/10-step version), explicitly out of scope for this paper |
| `PRICE_REANCHOR` | **both** (open question, run both ways — see below) | `constants.py:28` |

**Simulate-time (CLI flags, independent of the checkpoint — safe to change per-run):**
`-type DDPM -nsteps 100`, and **no** `--type-decode`, `--depth-noise`, `--size-reshape`,
`--depth-reshape`, `--book-target-thick`/`--book-cancel-rate`, `--cond-clip`, `--flow-balance`,
`--depth-drift`, `--cancel-boost`, `--dn-target-exec`, `--guidance-scale` (default 1.0),
`--churn-steps`/`--churn-strength` (defaults are no-ops at DDPM/eta irrelevant). All of these are
the decode-time stability interventions the paper explicitly excludes.

## Price re-anchoring — resolved as "run both"

Per the plan's own suggestion (cheap to toggle, the answer is itself data): Exp 2 and Exp 3's
grids run **both** `PRICE_REANCHOR` on and off as two named variants of "the model under test."
Report whether it changes the failure rate rather than picking one before running anything.

## Checkpoint provenance — the actual gap, stated plainly

**Only one checkpoint has ever been committed to this repo, on any branch:**
`data/checkpoints/TRADES/val_ema=0.7_epoch=2_INTC_se_256_au_64_CD_8_seed_30.ckpt` (present
locally now). Its own commit message (`b0f449c`, 2026-07-27) says explicitly:

> "Not the 0.627 winner — that needs a remote backup or a re-run — but a usable checkpoint to
> test stability / resume from."

So the local checkpoint is a **documented fallback**, not the project's preferred one. Reading
`analysis/MASTER_RESULTS.md` as it exists on `origin/main` (not on this branch — read it with
`git show origin/main:analysis/MASTER_RESULTS.md`, don't merge branches to get it) gives the real
picture:

- **`0.724`** is "Phase 2 — Stability"'s baseline checkpoint: trained with `UNCLAMP_DEPTH` +
  `PRICE_REANCHOR` file flags set (confirmed explicitly in that doc's §2.1), no scheduled
  sampling. This is the closest existing match to "the model under test," modulo needing a
  **vanilla** (no decode-flags) DDPM-100 run, which that document's own run matrix (§ "Part 3",
  rows **a**-**d**) shows was never actually done — a `0.724, vanilla, DDPM-100` cell does not
  exist in any table there. That's expected: it's what Exp 2 exists to produce for the first
  time, not something to find already computed.
- **`SS epoch 4`** resumes from `0.724` with scheduled sampling on — this is explicitly "our fix"
  per the paper's scope and must NOT be used for the model-under-test checkpoint (it's the right
  checkpoint for a *different*, future paper about the fix, not this one).
- **`0.627`, `0.681`, `0.719`** are earlier "Phase 1 — Diagnosis" checkpoints, described as
  **pre-fix** in that document's own phase table (§1.3) — do not use.
- **The `0.724` checkpoint file itself is not present in this repo, on any branch, in git
  history.** Checkpoints are gitignored and never committed except the one recovery exception
  above. It exists only on whatever machine trained it (presumably the remote GPU box's local
  disk, possibly lost the same way `0.627` was — the recovery commit implies checkpoints have
  been lost to remote wipes before).

## 2026-09-08 — superseded: train fresh instead of hunting for pre-existing checkpoints

Decided live, on the GPU box, after finding the `data/checkpoints/TRADES/` directory holds
several pre-existing checkpoints (`0.69`, `0.7`, plus whatever else) with exactly the provenance
problem this whole document warned about — none of them verifiably clean. Rather than forensically
resolve an old file's history, **train three fresh checkpoints**, one per axis this paper needs to
report on, each with its flag state logged automatically the moment it's produced:

| variant | `UNCLAMP_DEPTH` | `PRICE_REANCHOR` | `SCHEDULED_SAMPLING` |
|---|---|---|---|
| `baseline` | True | False | False |
| `reanchor` | True | **True** | False |
| `ss`       | True | False | **True** |

Training cost is small for this model at this batch size — the original TRADES paper itself
trained lightly, so stopping after **epoch 1, at most epoch 2** (~1.5h/epoch) is in keeping with
that, not a shortcut. `scripts/train_variant.sh <variant>` enforces exactly this (via
`MAX_EPOCHS_OVERRIDE=2`, `configuration.py`, so training stops cleanly at epoch 2 rather than
being cut off by a wall-clock guess) and keeps every epoch's checkpoint
(`KEEP_EPOCH_CHECKPOINTS_FLAG`) rather than only the best-by-val-loss one, since either epoch may
be worth testing. `scripts/train_three_variants.sh` runs all three back to back, unattended
(~5h safety cap each, ~15h total) — see `rl_execution/RUNBOOK_instability.md`.

**Collision note:** all three variants share identical seed/hyperparameters, so an epoch-1
checkpoint from two variants could round to the *same filename* in the shared
`data/checkpoints/TRADES/` directory. `train_variant.sh` moves every checkpoint it produces into
its own `data/checkpoints/TRADES_<variant>/` the moment training stops, before the next variant
can start and risk overwriting it.

**Price re-anchoring is no longer an independent per-run choice** (the "run both ways" idea from
2026-09-07 below assumed one checkpoint being evaluated both ways at simulate time only — but
`PRICE_REANCHOR` also changes *training* conditioning, so a checkpoint trained without it and
simulated with it would be out-of-distribution, not a real "both ways" test). The `reanchor`
variant is trained with it on; `exp2_survival_sweep.sh --variant reanchor` and
`exp3_teacher_forced.sh --variant reanchor` force the matching simulate-time state automatically.

## Confirmed checkpoints

**Written automatically by `bash scripts/pin_trained_variant.sh <variant>`**, called by
`train_three_variants.sh` right after each training run — one `CKPT_PATH_<VARIANT>=` line per
variant, below. `exp2`/`exp3` read the right one via `--variant baseline|reanchor|ss`. Do not
hand-edit these lines — rerun `pin_trained_variant.sh` instead.

*(empty until training finishes on the box that's actually running it)*

---

## Superseded: the original pre-existing-checkpoint hunt (2026-09-07)

*Kept for the record, not acted on further — see the 2026-09-08 section above for what actually
happened. `scripts/exp1_pin_checkpoint.sh` and its `CKPT_PATH=` (no suffix) convention still work
as a fallback path if a specific pre-existing checkpoint's provenance is ever pinned down by other
means, but the plan going forward is the three trained variants above.*

**Only one checkpoint has ever been committed to this repo, on any branch:**
`data/checkpoints/TRADES/val_ema=0.7_epoch=2_INTC_se_256_au_64_CD_8_seed_30.ckpt`. Its own commit
message (`b0f449c`, 2026-07-27) says explicitly: "Not the 0.627 winner ... but a usable checkpoint
to test stability / resume from." `analysis/MASTER_RESULTS.md` (on `origin/main`, not this branch)
names `0.724` as the Phase-2 baseline and `0.627`/`0.681`/`0.719` as pre-fix — none of these were
ever confirmed present with verifiable provenance on the actual GPU box.
