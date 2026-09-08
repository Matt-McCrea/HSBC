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

## What Track B must do before Exp 2 can start (this is Track B's real "B(-1)")

1. On the remote GPU box: run `bash scripts/exp1_pin_checkpoint.sh`. It lists every checkpoint in
   `data/checkpoints/TRADES/`, flags known decoys by name (0.627/0.681/0.719 pre-fix, 0.7_epoch
   the documented fallback), and reports whether exactly one, zero, or more than one file matches
   `val_ema=0.724*`. If more than one file could plausibly be the SS-resumed checkpoint sharing a
   rounded val_ema with the plain baseline (the exact landmine in §1.4 below), do not guess —
   resolve it (training log, mtime, whoever ran it) before filling in the section below. If
   exactly one match: treat it as the model-under-test checkpoint (pending the vanilla DDPM-100
   smoke test in B0) and skip to B0. **Every Track B script takes the exact file via
   `--ckpt-path`, never a val_ema number via `--id`/`-id`** — `--id` matching is inherently
   ambiguous when two checkpoints round to the same val_ema.
2. If it is gone: retrain it. This is the safer option in general (forensic provenance from a
   found file is unrecoverable per the table above), and cheap to get right this time — launch
   training with `UNCLAMP_DEPTH_FLAG` present (and, for the PRICE_REANCHOR-on variant,
   `PRICE_REANCHOR_FLAG` too — train the two variants as two separate runs, since this flag
   affects *training* conditioning, not just simulate-time), `SCHEDULED_SAMPLING_FLAG` **absent**,
   and **write the exact flag state and launch command into this file** (a `## Confirmed
   checkpoint` section below) the moment training starts, before anything else — the entire
   point of this document is to not repeat the provenance loss that produced this gap.

## Confirmed checkpoint

*(to be filled in once Track B step 1 or 2 above completes — do not run Exp 2/3/4 against
anything until this section is filled in with a specific file path and flag state)*
