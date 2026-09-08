# Instability paper — Track B runbook (remote GPU box)

Companion to `/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md`. Written for a remote
terminal with **no copy/paste** — every command below is short, typed by hand, and takes zero or
near-zero arguments. Nothing needs a pasted file path: the checkpoint is read automatically from
`analysis/model_under_test.md` once step 1 runs. Paste each command's OUTPUT back here for review
(that direction still works — it's typing INTO the remote that doesn't).

Numbered commands = run in this exact order. Do not skip ahead.

## 1. Pin the checkpoint (run this first, always)

```
bash scripts/exp1_pin_checkpoint.sh
```

**What it does:** lists every `.ckpt` in `data/checkpoints/TRADES/`, flags known decoys by name
(`0.627`/`0.681`/`0.719` = pre-fix, `0.7_epoch` = the old documented fallback — see
`analysis/model_under_test.md` for why), and checks how many files match `val_ema=0.724*`.

**What to do with the output:**
- Exactly one match → it writes `CKPT_PATH=...` into `analysis/model_under_test.md` itself and
  prints the next command to run. Go to step 2.
- Zero matches → the checkpoint isn't on this box. Stop; it needs to be copied here or retrained
  (see `analysis/model_under_test.md`, "What Track B must do").
- More than one match → **stop, do not proceed**. Paste the full output back — two checkpoints
  (e.g. the plain baseline and a scheduled-sampling-resumed one) can round to the same val_ema,
  and picking wrong here corrupts every later experiment.

## 2. Smoke test (GPU, ~15-20 min)

```
bash scripts/exp2_survival_sweep.sh --smoke
```

**What it does:** one day (2015-01-30), 2 seeds, 30-minute window, no price-reanchoring. Confirms
the checkpoint actually loads on the GPU, the flags took, and a run produces a scoreable CSV.

**Check:**

```
cat exp2_results/*/summary.md
```

Both runs must show real numbers (`time_to_freeze`, `time_to_diverge`, `survived_to_end`), not an
`ERROR` block. If either is an ERROR, stop and paste the matching `exp2_results/*/logs/*.txt` back
— do not go to step 3.

## 3. Pilot (GPU, a few hours)

```
bash scripts/exp2_survival_sweep.sh --pilot
```

**What it does:** 3 days, 5 seeds, 2-hour window, both price-reanchoring variants. Per the plan's
own suggestion: read this before paying for the full sweep.

**Check:**

```
cat exp2_results/*/summary.md
```

Does the failure pattern look like what the dissertation already reports (freezes/diverges within
the horizon, not everything surviving cleanly)? If it looks wrong, stop here and say so — don't
proceed to step 4 on a pilot that doesn't match expectations.

## 4. Full survival sweep (GPU, the dominant cost — hours to overnight)

```
bash scripts/exp2_survival_sweep.sh
```

**What it does:** all 20 days, 20 seeds, both price-reanchoring variants, 4-hour window (the
plan's headline evidence). Resumable — safe to re-launch the exact same command after a kill,
already-`.done` runs are skipped.

## 5. Teacher-forced control (GPU, faster than step 4)

```
bash scripts/exp3_teacher_forced.sh
```

**What it does:** same checkpoint, both reanchor variants, real history fed back at every step
instead of the model's own output (Experiment 3 — is Exp 2's failure specifically about compounding
self-generated error, or does per-step quality just degrade regardless?).

**Check:**

```
cat exp3_results/*/open_loop_noreanchor.txt
```

Compare the `bucket_early` vs `bucket_late` numbers it prints. Flat here + failing in step 4 =
compounding self-generated error confirmed. Degrading here too = a different story — write it up
honestly, don't force the first explanation.

## 6. Checkpoint robustness sweep (GPU, run after step 4's headline number is in)

```
bash scripts/exp4_checkpoint_sweep.sh
```

**What it does:** every checkpoint in `data/checkpoints/TRADES/` (minus known decoys, excluded by
default), a short battery (5 seeds, 1-hour cap), scored the same way as step 4 — is the failure
one unlucky checkpoint, or does it show up across the training run?

**Check:**

```
cat exp4_results/*/summary.md
```

## Running Experiment 0 (no GPU — safe to do in parallel with any step above)

Experiment 0 doesn't touch the checkpoint or the GPU at all — start it in a second terminal/tmux
window whenever, it doesn't block or get blocked by steps 1-6.

```
free -g
```

Read the "available" number, then (replace `N` with roughly half of it — a single 390-minute cell
hit 23GB locally, so leave headroom):

```
bash scripts/exp0_abides_replay_error.sh --max-mem-gb N
```

Resumable, same as above — safe to re-run the same command after a kill.

## Keeping the log current

After each numbered step above finishes, append one short dated paragraph to
`analysis/instability_paper_log.md` saying what ran and the headline number — don't wait until
everything's done.
