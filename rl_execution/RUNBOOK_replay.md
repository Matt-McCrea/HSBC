# Historic-data arm — runbook

Runs the same agent and the same evaluation against the **real** INTC order flow instead
of generated flow. Replay does no diffusion sampling, so **no model is loaded and no GPU
is needed** for anything except stage 4.

Every step prints a summary to stdout, so output can be pasted back as-is.

## What this produces

Baselines (TWAP, Almgren–Chriss) need no training, so they run as free controls inside
whichever environment is being evaluated.

|  | Eval on **generative** | Eval on **replay** |
|---|---|---|
| Trained **generative** | done (`eval_frontier_lam.jsonl`) | **A** — does the sim-trained agent transfer to real data? |
| Trained **replay** | **C** — does a replay-trained agent survive a reactive market? | **B** — what a replay-only pipeline would have produced |

`generate_held_out_seeds` is independent of world mode, so the same seed list runs in
both arms and sim-vs-real is a **paired** comparison.

---

## Stage 0 — smoke test. Hard gate.

The replay arm has never run end-to-end. Do this before anything else.

```bash
python -u -m rl_execution.benchmark --n-episodes 2 --world-mode replay \
    --side SELL --out logs/replay_smoke.jsonl --run-name replay_smoke 2>&1 | tail -60
python -u -m rl_execution.preflight logs/replay_smoke.jsonl
```

**What must be true.** `world_mode=replay checkpoint=None` in the header — a checkpoint
path there means the flag did not take and the model is being loaded. `rem_quantity=0`,
trajectory length 10, per-step rewards summing to −shortfall, fills ≤ Q. Also check
`ignored_cancel` near zero in the WorldAgent diagnostics; a large count means replayed
cancels are failing to resolve against the cold-started book. And `unique_mid_count`
should be comparable to the generative arm's ~9 — a frozen mid means the replay is not
moving the market.

**Record the seconds per episode.** Every budget below is sized off it, and it is
currently unmeasured. For scale, a 5-minute window carries ~12,000 real messages
(5,962 new orders, 5,342 cancels, 515 executions, measured on 2015-01-30).

**Paste back**: the tail above plus the preflight report.

---

## Stage 1 — cell A, the transfer test (CPU)

The existing sim-trained table, run on real data, on the same held-out seeds as the
committed generative result:

```bash
# mixed-side: directly comparable to the numbers already in the chapter
python -u -m rl_execution.evaluate --world-mode replay --n-seeds 18 --skip-ddpm \
    --ac-kappa 2.0 --qtable checkpoints/qtable_lam.npz \
    --out logs/eval_replay_mixed.jsonl 2>&1 | tail -40

# SELL-only: in-distribution for a table trained with --side SELL
python -u -m rl_execution.evaluate --world-mode replay --n-seeds 18 --skip-ddpm \
    --side SELL --ac-kappa 2.0 --qtable checkpoints/qtable_lam.npz \
    --out logs/eval_replay_sell.jsonl 2>&1 | tail -40
```

`--skip-ddpm` is not a compromise here: in replay there is no sampler to compare, so
the arm is skipped automatically.

Both runs are wanted. The mixed-side one preserves comparability with the existing
chapter numbers; the SELL-only one corrects the side mismatch recorded as a limitation.

---

## Stage 2 — calibrate on real data (seconds, pure pandas)

```bash
python -u -m rl_execution.calibrate logs/eval_replay_sell.jsonl
python -u -m rl_execution.impact    logs/eval_replay_sell.jsonl
```

σ, η and the drift *t*-statistic measured on **real** flow. Comparing these against the
same three from the generative log is the quantitative core of the simulator-realism
claim. Expect the drift *t* to fall towards insignificance (generative: +7.09 bps,
*t* = 7.88) and the execution rate to fall from 17–18% toward the 4–6% real benchmark.
Both would confirm the biases already documented.

---

## Stage 3 — cell B, train on replay (CPU)

Same hyperparameters as the generative run, so the training environment is the only
variable. **λ stays at 42.12** deliberately: recalibrating per environment would
confound risk aversion with training environment and destroy the 2×2.

Set `--n-episodes` and `--max-hours` from the stage-0 rate. The generative run reached
only 114 episodes because each cost ~823 s of diffusion sampling; replay has no sampler,
so several hundred to a few thousand episodes should be affordable — which would lift
state coverage well above the 32/55 the generative run managed.

```bash
python -u -m rl_execution.train --world-mode replay --n-episodes 500 \
    --max-hours 6 --side SELL --alpha-mode visit-count \
    --epsilon-decay 0.99 --inventory-penalty 42.1214 \
    --reward-mode per-step --reward-benchmark arrival \
    --checkpoint checkpoints/qtable_replay.npz --out logs/train_replay.jsonl 2>&1 | tail -40
```

Then evaluate it in its own environment:

```bash
python -u -m rl_execution.evaluate --world-mode replay --n-seeds 18 --skip-ddpm \
    --side SELL --ac-kappa 2.0 --qtable checkpoints/qtable_replay.npz \
    --out logs/eval_replayagent_replay.jsonl 2>&1 | tail -40
```

---

## Stage 4 — cell C, the reactive-market test (**needs GPU**)

The only stage that loads the diffusion model. Run last, and only if GPU is available.
This is the argument *for* generative simulation: replay cannot represent market
response, so an agent trained there should be over-optimistic and degrade when the
market reacts back.

```bash
python -u -m rl_execution.evaluate --n-seeds 18 --skip-ddpm --side SELL \
    --ac-kappa 2.0 --qtable checkpoints/qtable_replay.npz \
    --ckpt-path data/checkpoints/TRADES/<file>.ckpt \
    --out logs/eval_replayagent_gen.jsonl 2>&1 | tail -40
```

Severable. Stages 0–3 and 5 complete the sim-to-real story on their own; the write-up
should be arranged so this stage's absence costs a paragraph, not a chapter.

---

## Stage 5 — analysis (seconds, runs anywhere)

```bash
python -m rl_execution.compare_policies logs/eval_replay_sell.jsonl --baseline twap
python -m rl_execution.compare_policies logs/eval_replay*.jsonl --baseline twap --by-side
python -m rl_execution.inspect_policy   checkpoints/qtable_replay.npz
python -m rl_execution.impact           logs/train_replay.jsonl
```

`inspect_policy` on the two Q-tables side by side is the qualitative result: **does an
agent trained without market response learn to be more aggressive, because nothing
pushes back?** Compare its greedy action distribution against the generative table's
(passive 28.1%, light 43.8%, neutral 12.5%, aggressive 6.2%, very aggressive 9.4%).

---

## Reading the two arms together

Replay gives an **exact counterfactual**: replayed orders do not react, so a seeded
replay episode reproduces bit-for-bit, and re-running with the execution agent disabled
reproduces the market exactly.

By the same token replay **understates** impact. The agent's orders still consume real
resting liquidity, so temporary impact is real, but nobody responds, so reactive impact
is absent. The generative arm overstates it if anything, since its flow does react.
Reporting both brackets the truth instead of picking a side.

One caveat specific to training on replay: there are only 20 trading days, so a
replay-trained agent sees the same days repeatedly and can overfit them in a way the
generative agent cannot. Whether its advantage on replay survives into stage 4 is the
check that matters.
