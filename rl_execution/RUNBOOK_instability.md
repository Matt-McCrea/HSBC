# Instability paper — Track B runbook (remote GPU box)

Companion to `/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md`. Written for a remote
terminal with **no copy/paste** — every command below is short, typed by hand, and takes zero or
near-zero arguments. Nothing needs a pasted file path: each variant's checkpoint is read
automatically from `analysis/model_under_test.md` once step 1 finishes. Paste each command's
OUTPUT back here for review (that direction still works — it's typing INTO the remote that
doesn't).

Numbered steps = run in this exact order. Do not skip ahead.

## 1. Train three clean checkpoints (GPU, ~15h total, fully unattended)

Pre-existing checkpoints on this box turned out to have exactly the provenance problem
`analysis/model_under_test.md` originally warned about — none verifiably clean. Training fresh is
cheap for this model (the original TRADES paper itself trained lightly): ~1.5h/epoch, stopping at
epoch 2, ~5h safety cap per variant. Three variants: `baseline` (the two data-pipeline fixes
only), `reanchor` (+ price re-anchoring), `ss` (+ scheduled sampling / teacher-forcing fix).

```
nohup bash scripts/train_three_variants.sh > train_three_variants.log 2>&1 &
disown
```

`nohup ... & disown` so a dropped connection doesn't kill it — safe to close the terminal and come
back later. Check progress any time with:

```
tail -f train_three_variants.log
```

Each variant's checkpoint gets pinned into `analysis/model_under_test.md` automatically the
moment its training stops — nothing to type. If the session ends before all three finish,
rerunning the exact same `train_three_variants.sh` command later skips whatever's already pinned
and continues with the rest.

**Check when done:**

```
grep CKPT_PATH_ analysis/model_under_test.md
```

Three lines (`CKPT_PATH_BASELINE`, `CKPT_PATH_REANCHOR`, `CKPT_PATH_SS`) means all three are
ready. Fewer means some are still running or failed — check `train_<variant>_*.log` for whichever
is missing.

## 2. Smoke test each variant (GPU, ~15-20 min each)

```
bash scripts/exp2_survival_sweep.sh --smoke --variant baseline
bash scripts/exp2_survival_sweep.sh --smoke --variant reanchor
bash scripts/exp2_survival_sweep.sh --smoke --variant ss
```

**What it does:** one day (2015-01-30), 2 seeds, 30-minute window, price-reanchoring forced to
match how that variant was trained. Confirms the checkpoint actually loads on the GPU and a run
produces a scoreable CSV, before spending real budget.

**Check after each:**

```
cat exp2_results/baseline_*/summary.md
```

(swap `baseline_` for `reanchor_`/`ss_` for the other two). Both runs per variant must show real
numbers (`time_to_freeze`, `time_to_diverge`, `survived_to_end`), not an `ERROR` block. If any is
an ERROR, stop and paste the matching `exp2_results/*/logs/*.txt` back — do not go to step 3 for
that variant.

## 3. Pilot each variant (GPU, ~9h each)

```
bash scripts/exp2_survival_sweep.sh --pilot --variant baseline
```

**What it does:** 5 days, 3 seeds, 30-minute window (resized from an original 2h-horizon preset
that measured at >5h/run — see `analysis/instability_paper_log.md`). Read this before paying for
the full sweep.

**Check:**

```
cat exp2_results/baseline_*/summary.md
```

Does the failure pattern look like what the dissertation already reports? If it looks wrong, stop
and say so — don't proceed to step 4 for that variant on a pilot that doesn't match expectations.
Repeat for `--variant reanchor` and `--variant ss` once `baseline`'s pilot looks right (or in
parallel if there's enough of the 3-day budget left to risk it before validating the first one).

## 4. Full survival sweep per variant (GPU, the dominant cost)

```
bash scripts/exp2_survival_sweep.sh --variant baseline
```

Size `--days`/`--seeds`/`--et` to whatever budget is left after steps 1-3, using the pilot's
measured seconds/run (see `exp2_results/baseline_*/summary.md`'s `(NNNNs)` per run) to do the
math — don't reuse the wide-open 20-day/20-seed/4h default without checking it fits first.
Resumable — safe to re-launch the exact same command after a kill, already-`.done` runs are
skipped. Repeat for `reanchor` and `ss`.

## 5. Teacher-forced control per variant (GPU, faster than step 4)

```
bash scripts/exp3_teacher_forced.sh --variant baseline
```

**What it does:** same checkpoint, reanchor state matched to training, real history fed back at
every step instead of the model's own output (Experiment 3 — is Exp 2's failure specifically about
compounding self-generated error, or does per-step quality just degrade regardless?).

**Check:**

```
cat exp3_results/baseline_*/open_loop_noreanchor.txt
```

Compare `bucket_early` vs `bucket_late`. Flat here + failing in step 4 = compounding
self-generated error confirmed. Degrading here too = a different story — write it up honestly.
Repeat for `reanchor` and `ss`.

## 6. Checkpoint-epoch robustness (GPU, run after step 4's headline number is in)

```
bash scripts/exp4_checkpoint_sweep.sh --ckpt-dir data/checkpoints/TRADES_baseline
```

**What it does:** every checkpoint epoch saved for that ONE variant's training run, short battery
(5 seeds, 1h cap), scored the same way as step 4 — is the failure specific to one epoch, or does
it show up across the run? Repeat with `--ckpt-dir data/checkpoints/TRADES_reanchor` /
`TRADES_ss` if time remains.

**Check:**

```
cat exp4_results/*/summary.md
```

## Running Experiment 0 (no GPU — safe to do in parallel with everything above)

Experiment 0 doesn't touch the checkpoint or the GPU at all — start it in a second terminal/tmux
window whenever, it doesn't block or get blocked by steps 1-6.

```
free -g
```

Read the "available" number, then (replace `N` with roughly half of it — a single 390-minute cell
hit 23GB locally, so leave headroom, and this box is also running training/sampling that wants its
own RAM):

```
bash scripts/exp0_abides_replay_error.sh --max-mem-gb N
```

Resumable, same as above — safe to re-run the same command after a kill.

## Keeping the log current

After each numbered step above finishes, append one short dated paragraph to
`analysis/instability_paper_log.md` saying what ran and the headline number — don't wait until
everything's done.
