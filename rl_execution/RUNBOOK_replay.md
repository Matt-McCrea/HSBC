# Replay arm — CPU runbook

The replay arm runs the same policies against the **real** INTC order flow instead of
generated flow. No diffusion sampling, so **no model is loaded and no GPU is needed**.
Run it on the CPU remote; it is independent of whatever the GPU box is doing.

Everything below writes to a fresh log and prints a summary to stdout, so the output
can be pasted back as-is.

---

## Step 1 — smoke test first (~minutes, 2 episodes)

The replay arm has never been run end-to-end. Do this before committing to a long run.

```bash
python -u -m rl_execution.benchmark \
    --n-episodes 2 --world-mode replay \
    --out logs/replay_smoke.jsonl --run-name replay_smoke 2>&1 | tail -60
```

Then check the invariants:

```bash
python -u -m rl_execution.preflight logs/replay_smoke.jsonl
```

**What should be true.** `[ExecutionEnv] world_mode=replay checkpoint=None` in the
header — if a checkpoint path appears, the model is being loaded and the flag did not
take. Every episode ends `rem_quantity=0`, trajectory length 10, per-step rewards
summing to −shortfall, and total fills ≤ Q. An episode should take **seconds to a couple
of minutes**, not the ~20 minutes the generative arm takes; if it is slow, something is
still sampling.

**Paste back**: the tail above plus the preflight report.

---

## Step 2 — replay evaluation (once the smoke test passes)

TWAP, Almgren-Chriss, and the trained RL policy on identical held-out seeds:

```bash
python -u -m rl_execution.evaluate \
    --world-mode replay --n-seeds 30 --skip-ddpm \
    --ac-kappa 2.0 --qtable checkpoints/qtable_lam.npz \
    --out logs/eval_replay.jsonl 2>&1 | tail -60
```

`--skip-ddpm` is not a compromise here: in replay there is no sampler to compare,
because the market is real data rather than something sampled. The flag is accepted and
the arm is skipped automatically.

`--n-seeds 30` is affordable precisely because there is no sampling cost — raise it if
episodes turn out to be as fast as expected. Tighter error bars are the main thing this
arm buys over the generative one.

If the frontier refits do not exist yet, drop `--qtable` and it runs TWAP + AC only,
which is still a usable result.

---

## Step 3 — analysis (seconds, pure pandas)

```bash
python -u -m rl_execution.analyze_logs logs/eval_replay.jsonl --by policy_name
python -u -m rl_execution.calibrate   logs/eval_replay.jsonl
python -u -m rl_execution.impact      logs/eval_replay.jsonl
```

`calibrate` on the replay log is the one that matters most for the write-up: it gives σ,
η and the drift t-statistic **measured on real data**. Comparing those three numbers
against the same three from the generative log is the quantitative answer to "is this
simulator usable for live trading" — same policies, same estimator, one simulated market
and one real.

Expect the drift t-statistic to fall towards insignificance here. The generative arm
measured **+11.94bps per 5 minutes (t=4.84)**, which flatters a seller; real prices are
near-martingale. The gap between the two is the size of the contamination, which is a
reportable result rather than a caveat.

---

## Interpreting the two arms together

Replay gives an **exact counterfactual**: replayed orders do not react to us, so
re-running a seed with the execution agent disabled reproduces the market exactly. That
licenses a clean causal impact estimate.

By the same token replay **understates** impact — a real market would have responded to
our orders and this one cannot. The generative arm overstates it if anything, since its
flow does react. Reporting both brackets the truth instead of picking a side, and that
bracketing is the honest form of the claim.
