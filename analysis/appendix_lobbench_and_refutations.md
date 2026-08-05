# Appendix — Full LOB-Bench Results & Refuted Approaches

*Compiled 2026-08-05. Model/checkpoint selection between the candidates below is deliberately left
open here — see PROJECT_STATUS.md for the decision status.*

## A. Final-model candidate, full-month LOB-Bench (`val_ema=0.724_epoch=0`)

Confirmed stable across all 20 trading days (zero timeouts, see `appendix_checkpoint_evidence.md` §B).
Per-day, per-metric Wasserstein distance to real market data (lower = more realistic):

| Day | spread | inter-arrival | book imbalance | flow imbalance | limit depth | cancel depth |
|---|---|---|---|---|---|---|
| 2015-01-02 | 0.718 | 0.559 | 0.525 | 0.624 | 0.228 | 0.194 |
| 2015-01-05 | 0.719 | 0.453 | 0.356 | 0.688 | 0.219 | 0.194 |
| 2015-01-06 | 0.677 | 0.720 | 0.471 | 0.664 | 0.139 | 0.094 |
| 2015-01-07 | 0.654 | 0.410 | 0.464 | 0.820 | 0.246 | 0.176 |
| 2015-01-08 | 0.655 | 0.407 | 0.483 | 0.760 | 0.255 | 0.131 |
| 2015-01-09 | 0.837 | 0.525 | 0.428 | 0.811 | 0.194 | 0.159 |
| 2015-01-12 | 0.645 | 0.506 | 0.369 | 0.815 | 0.178 | 0.152 |
| 2015-01-13 | 0.748 | 0.843 | 0.368 | 0.748 | 0.157 | 0.142 |
| 2015-01-14 | 0.720 | 0.824 | 0.384 | 0.662 | 0.200 | 0.213 |
| 2015-01-15 | 0.864 | 0.877 | 0.364 | 0.641 | 0.253 | 0.387 |
| 2015-01-16 | 0.591 | 0.812 | 0.350 | 0.507 | 0.154 | 0.097 |
| 2015-01-20 | 0.710 | 0.763 | 0.464 | 0.623 | 0.133 | 0.098 |
| 2015-01-21 | 0.954 | 0.816 | 0.429 | 0.567 | 0.231 | 0.187 |
| 2015-01-22 | 0.759 | 0.691 | 0.439 | 0.652 | 0.204 | 0.126 |
| 2015-01-23 | 0.746 | 0.448 | 0.509 | 0.664 | 0.245 | 0.147 |
| 2015-01-26 | 0.708 | 0.584 | 0.533 | 0.601 | 0.217 | 0.149 |
| 2015-01-27 | 0.608 | 0.507 | 0.367 | 0.636 | 0.196 | 0.131 |
| 2015-01-28 | 0.727 | 0.655 | 0.446 | 0.635 | 0.204 | 0.138 |
| 2015-01-29 | 0.706 | 0.434 | 0.500 | 0.686 | 0.179 | 0.142 |
| 2015-01-30 | 0.625 | 0.671 | 0.384 | 0.664 | 0.172 | 0.164 |
| **mean** | **0.719** | **0.625** | **0.432** | **0.673** | **0.200** | **0.161** |
| **grand mean (all metrics, all days)** | | | | | | **0.468** |

## B. Scheduled-sampling retrain candidates, full-month LOB-Bench

Same real-data comparison, run on the SS-retrain epochs (see `appendix_checkpoint_evidence.md` §C for
their per-day stability/activity detail — all three cleared all 20 days with zero timeouts).

| Metric | Baseline (epoch 0) | Epoch 2 (`0.701`) | Epoch 3 (`0.697`) | Epoch 4 (`0.69`) |
|---|---|---|---|---|
| spread | 0.719 | 0.285 | 0.238 | 0.192 |
| inter-arrival | 0.625 | 0.538 | 0.514 | 0.504 |
| book imbalance | 0.432 | 0.400 | 0.389 | 0.394 |
| flow imbalance | 0.673 | 0.652 | 0.647 | 0.645 |
| limit depth | 0.200 | 0.170 | 0.185 | 0.186 |
| cancel depth | 0.161 | 0.145 | 0.161 | 0.152 |
| **grand mean** | **0.468** | **0.365** | **0.356** | **0.346** |

**Reading this table**: every retrain epoch scores better than the baseline on overall LOB-Bench
realism, driven mostly by a large spread-realism improvement (baseline's weakest metric) plus modest
gains on timing and book imbalance; limit/cancel depth are roughly flat. This sits alongside the
separate finding that none of these epochs increased `uniq_mid` (order-book activity level) toward
real levels, which was the retrain's original hypothesis — see `appendix_checkpoint_evidence.md` §D
and the reconciliation figure (`analysis/plots/hsbc_ss_retrain_*/3_reconciliation_panel.png`) for the
full discussion of why both findings hold at once.

Figures: `analysis/plots/hsbc_ss_retrain_*/1_ss_retrain_per_metric.png` (per-metric bars),
`2_ss_retrain_grand_mean.png` (grand mean by epoch), `3_reconciliation_panel.png` (activity vs realism).

## C. Same-day comparison vs the original 100-step baseline (DDPM)

For reference, all candidates above vs the un-accelerated 100-step DDPM baseline, on the single day
(2015-01-30) where DDPM has been run:

| Config | Grand mean (2015-01-30 only) |
|---|---|
| DDPM, 100 steps | 0.507 |
| `0.724_epoch=0` (final-model candidate) | 0.447 |
| An earlier, single-day-tuned config (never tested for cross-day generalisation) | 0.298 |

*(A full-month DDPM baseline, for a like-for-like comparison against the full-month numbers in §A/B
above, is not yet run — see PROJECT_STATUS.md "next steps.")*

## D. Refuted approaches — consolidated

Every decode-time or training-time lever tried and not adopted, across the whole project, in one
place. All of these were measured, not assumed — see `PROJECT_STATUS.md` and the git history for the
underlying sweep scripts/results behind each row.

| Lever | What it did | Result |
|---|---|---|
| `--dn-target-exec` (execution-rate feedback controller) | Adjusts decode noise to hold a target execution rate | Halved over-execution but did not stop cross-day drift; sigma-floor saturation |
| `--flow-balance` | Directional counter-bias on limit-order flow, targeting one-sided drift | No measurable effect once isolated on a genuinely drifting checkpoint (swamped by the underlying instability) |
| `--cancel-boost` | Biased type-decode toward cancels to lift cancel share | Nudged cancel share *down* instead — opposite of intended effect |
| `--cond-clip` | Capped fed-back book-size conditioning | No effect on long-horizon drift, alone or combined with other levers |
| `--depth-reshape` (quantile matching to real marginal) | "Smart" alternative to `--depth-noise` | Pinned the marginal distribution but broke the joint distribution in the closed loop — imported the real distribution's far tail, causing teleportation/drift |
| `--depth-temp` | Linear temperature slide on decoded depth | Just a linear slide of the same spike; didn't fix freezing |
| HYBRID / CHURN samplers | Alternative sampling schedules | Same cliff behaviour as DDPM/DDIM baselines, numerically unstable on the unclamped checkpoint |
| More DDIM steps (beyond 10) | Hypothesis: more steps = more fidelity | Diverges rather than improves — 10 is not a floor being pushed against, it's closer to a sweet spot |
| Plain retraining (no scheduled sampling) | Retrain without the self-conditioning mechanism | Fixes the sign/orientation issue in decoded depth but not the magnitude/variance collapse — retraining alone can't fix a sampling-variance problem |
| Scheduled-sampling retrain, activity-level hypothesis | Train on self-generated rollouts to raise `uniq_mid` toward real levels | **Did not raise `uniq_mid`** (flat to declining across 5 epochs) — but see §B above: the same retrain improved overall LOB-Bench realism substantially, an unanticipated result on a different axis. Not a clean refutation; a partial/redirected one. |
