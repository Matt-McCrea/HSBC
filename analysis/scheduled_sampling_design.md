# Scheduled-sampling retrain — design note
*2026-07-26. Written before coding, per the Stage-3 plan. Decide the open questions below, then implement.*

## Why
Stage 1 (drift_sigma_sweep) confirmed the cross-day drift is decode-time-unfixable: on the drift days
every sigma in {0.10..0.30}, with and without the book-balancing cancel, drifts out of the real
envelope (uniq_mid 170-296 vs real 27-35). It is also **not** over-execution — at sigma=0.10 execution
is only 2.6% (below real 3.5%) yet the mid still walks ~150 ticks, and the driver is a one-sided
**limit-order-flow** imbalance (limOFI ~-7000) while execution aggression (B-S) stays roughly balanced.
So the failure is a closed-loop exposure-bias problem in the ORDER-FLOW channel: the model is teacher-
forced on real history in training but conditions on its own drifting output at simulation time, and a
directional limit-flow error compounds. Scheduled sampling is the standard cure: expose the model to its
own generated conditioning during training, targets stay real, so it learns to recover from its own drift.

## The obstacle
`training_step` (`models/diffusers/diffusion_engine.py:133`) receives one batch as
`(cond_orders, x_0, cond_lob)`: `cond_orders`/`cond_lob` are the real recent order history and book
state, `x_0` is the real next block to denoise. Scheduled sampling replaces the conditioning with the
model's own rollout while keeping `x_0` real. But the conditioning has two parts:
- `cond_orders` — the recent order sequence. Self-generating this is straightforward: call
  `self.sample()` (`diffusion_engine.py:88` -> `gaussian_diffusion.sample`, `:112`) to produce the next
  block and shift it into the conditioning window.
- `cond_lob` — the book state. At inference this is produced by ABIDES's matching engine from the
  generated orders (`WorldAgent` + exchange). **Training has no matching engine**, so a faithful
  self-generated `cond_lob` is the hard part.

## Options
| | approach | faithfulness | cost / risk |
|---|---|---|---|
| A | Full closed-loop rollout in training: reproduce the book update / matching inside the loop | highest | heavy; couples training to ABIDES; high risk; not a clean training change |
| **B** | **Self-generated order-history conditioning**: roll `cond_orders` with `self.sample()`, keep the real `cond_lob` (v1) or a light touch-level reconstruction (v2). Loss still vs real `x_0`. | medium-high — fixes the order-flow channel, which Stage 1 showed is the failure | moderate; k extra sampling passes per scheduled step |
| C | Light "unroll-1" self-conditioning: feed the model's own one-step prediction back as a regulariser | low | cheapest; least faithful; fallback only |

## Recommendation: Option B, v1, stop-gradient rollout
- **Self-generate `cond_orders` only; keep the real `cond_lob` for v1.** Stage 1 pinned the failure to
  directional limit *flow* (an order-sequence problem), so self-generating the order history is exactly
  the channel that needs correcting; the book-conditioning drift is second-order and avoiding the
  matching-engine dependency keeps this a clean, low-risk training change. Add v2 (approx `cond_lob`
  rebuilt from generated touch orders) only if v1 underfixes.
- **Stop-gradient on the rollout** (treat the self-generated conditioning as augmented input data), not
  backprop-through-rollout (Coletta et al. 2022). Stop-grad avoids the memory blow-up and instability of
  differentiating through k sampling passes, and is enough for the model to see and correct its own
  drifted inputs. Revisit backprop-through-rollout only if stop-grad underfixes.
- **Scheduled ramp**: probability p of using a self-generated conditioning on a given step ramps
  0 -> p_max over the first portion of training (teacher-forced early, more free-running later). Rollout
  depth k = how many blocks to self-generate before the predicted step.

## Hooks and flags (all additive, default off)
- `models/diffusers/diffusion_engine.py:133` `training_step`: with probability `p(epoch)`, replace
  `cond_orders` with a `torch.no_grad()` rollout (a helper that calls `self.sample()` k times and shifts
  the window), keep `cond_lob`; else the normal path. Add a `DIAG scheduled_sampling: p=.. k=.. used=..`
  line and count the self-conditioned fraction.
- New hyperparameters threaded through `configuration.py` / the TRADES hparam dict, default off:
  `SCHEDULED_SAMPLING` (bool), `SS_P_MAX`, `SS_K`, `SS_RAMP_FRAC`. Off => byte-for-byte the current loop.
- Prereqs already done: dataloader deadlock fixed (`num_workers=0`, `persistent_workers` gated in
  `preprocessing/DataModule.py` + `run.py`), and file-flag resume (`RESUME_TRAINING_FLAG` ->
  `ckpt_path` in `run.py`).

## Cost
Each scheduled step adds k sampling passes (k x ddim_nsteps NN calls) on top of the training forward.
At DDIM-ish nsteps, k=2, p_max=0.5, that is roughly a 1.5-2x step-time increase in the ramped phase.
Budget this against the 3-day window when setting max_epochs / p_max / k.

## Verification (before the 3-day launch)
1. `python -m py_compile` the changed files; flag-off run reproduces the current loss curve for a few steps.
2. IS_DEBUG smoke (256 samples) with scheduled sampling ON for ~50 steps: loss finite, no NaN, the DIAG
   line shows a rising self-conditioned fraction, GPU memory stable across the rollout.
3. One real epoch to measure throughput and confirm a checkpoint saves + resumes (touch
   `RESUME_TRAINING_FLAG`, restart, confirm it picks up).
4. Only then launch unattended (away_run-style) with `RESUME_TRAINING_FLAG` present.
5. Success = new checkpoint reduces drift on the Stage-1 drift days (uniq_mid toward real, mid stays in
   envelope) and does NOT regress the 30-min LOB-Bench winner or the Stage-2 predictive score.

## Open decisions for you
1. `cond_lob` handling: **v1 keep real (recommended)** vs v2 approx-reconstruct.
2. Rollout gradient: **stop-gradient (recommended)** vs backprop-through-rollout.
3. Defaults: propose `SS_P_MAX=0.5`, `SS_K=2`, `SS_RAMP_FRAC=0.4` (ramp over first 40% of epochs). Adjust
   for the 3-day compute budget.
