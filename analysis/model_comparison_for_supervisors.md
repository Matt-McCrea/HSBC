# Model comparison: non-SS baseline vs scheduled-sampling retrain

*Compiled 2026-08-05 for supervisor discussion. All numbers computed fresh from the raw sweep and
LOB-Bench outputs — sources listed in §8. **No recommendation is made here**; the intent is to lay out
the evidence, including the places where the candidates disagree.*

All models: TRADES architecture, 10-step DDIM sampling, decode-time config
`--depth-noise 0.3 --size-reshape --type-decode prior`. All evaluated on the same 20 trading days
(INTC, January 2015), 30-minute sessions from 09:30.

---

## 1. Headline comparison

| | **0.724 e0** (no SS) | **0.701 e2** (SS) | **0.697 e3** (SS) | **0.69 e4** (SS) |
|---|---|---|---|---|
| Scheduled sampling | no | yes | yes | yes |
| Days cleared (of 20) | 20 | 20 | 20 | 20 |
| **LOB-Bench grand mean** ↓ | 0.468 | 0.365 | 0.356 | **0.346** |
| Days beating baseline | — | 19/20 | 20/20 | 20/20 |
| **Mean 1s return vol (bp)** | **1.69** | 1.39 | 1.43 | 1.44 |
| Days in real vol band (1.5–2.5bp) | **15/20** | 4/20 | 10/20 | 7/20 |
| Mean uniq_mid | **15.1** | 13.7 | 11.3 | 13.0 |
| Mean sim time (30-min session) | 21:05 | **17:14** | 18:18 | 19:41 |
| Speed vs real time | 0.70× | **0.57×** | 0.61× | 0.66× |

↓ = lower is better. **Bold** = best in row. Note the two "best" columns are different models: the
baseline wins on volatility realism, the SS models win on LOB-Bench and speed. That tension is the
substance of §5.

---

## 2. LOB-Bench, per metric (full-month means)

| Metric | 0.724 e0 | 0.701 e2 | 0.697 e3 | 0.69 e4 |
|---|---|---|---|---|
| spread | 0.719 | 0.285 | 0.238 | **0.192** |
| inter-arrival | 0.625 | 0.538 | 0.514 | **0.504** |
| book imbalance | 0.432 | 0.400 | **0.389** | 0.394 |
| flow imbalance | 0.673 | 0.652 | 0.647 | **0.645** |
| limit depth | 0.200 | **0.170** | 0.185 | 0.186 |
| cancel depth | 0.161 | **0.145** | 0.161 | 0.152 |
| **grand mean** | 0.468 | 0.365 | 0.356 | **0.346** |

### 2a. The improvement is heavily concentrated in one metric

Decomposing the baseline → e4 improvement (total summed gain 0.737 across the six metrics):

| Metric | Change | Share of total improvement |
|---|---|---|
| spread | −0.526 | **71.4%** |
| inter-arrival | −0.121 | 16.4% |
| book imbalance | −0.038 | 5.2% |
| flow imbalance | −0.028 | 3.8% |
| limit depth | −0.015 | 2.0% |
| cancel depth | −0.009 | 1.2% |

**Roughly 71% of the headline gain is spread alone.** Excluding spread entirely, the five-metric
mean moves 0.418 → 0.376 — still an improvement, but a far more modest one. Worth deciding how much
weight a single-metric-dominated improvement should carry.

### 2b. But it is consistent, not outlier-driven

Per-day grand means (all six metrics), each SS model vs baseline:

- **e2**: better on 19/20 days, mean improvement −0.104
- **e3**: better on **20/20** days, mean improvement −0.113
- **e4**: better on **20/20** days, mean improvement −0.123

So while the *metric* composition is lopsided, the *day-to-day* result is about as consistent as it
could be. The one day e2 loses on is 2015-01-06 (0.472 vs baseline 0.461).

---

## 3. Stability and timing

| | 0.724 e0 | 0.701 e2 | 0.697 e3 | 0.69 e4 | *0.682 e5* |
|---|---|---|---|---|---|
| Days completed | 20/20 | 20/20 | 20/20 | 20/20 | *15/20 — **timed out*** |
| Mean sim time | 21:05 | 17:14 | 18:18 | 19:41 | *22:59 (partial)* |
| Median | 21:38 | 17:02 | 18:20 | 19:09 | *17:33* |
| Fastest / slowest day | 8:39 / 30:15 | 7:29 / 26:32 | 7:21 / 27:56 | 6:46 / 30:56 | *6:56 / 39:38* |
| Total GPU time, full month | 7.0 h | 5.7 h | 6.1 h | 6.6 h | *5.7 h (15 days)* |

All four main candidates cleared every day with zero timeouts, under the same 40-minute-per-day cap.
**Epoch 5 is shown for context only** — it timed out on 2015-01-23 (a day epoch 4 handled in 27:47)
and was already running slower than epoch 4 on the days it did complete, which is why the retrain was
stopped there.

Sim time tracks generated activity rather than model quality (pooled correlation between wall-clock
and uniq_mid across all epochs/days: r = 0.578, n = 95) — so the SS models being faster is mostly a
restatement of them being quieter, not an independent efficiency gain.

---

## 4. Activity and volatility detail

| | 0.724 e0 | 0.701 e2 | 0.697 e3 | 0.69 e4 | Real (reference) |
|---|---|---|---|---|---|
| Mean uniq_mid | 15.1 | 13.7 | 11.3 | 13.0 | ~27–66 (day-dependent) |
| Median uniq_mid | 14.0 | 13.0 | 11.0 | 13.0 | — |
| Range uniq_mid | 11–25 | 9–19 | 7–19 | 7–23 | — |
| Mean ret1s_std (bp) | 1.69 | 1.39 | 1.43 | 1.44 | target band 1.5–2.5 |
| Range ret1s_std | 1.20–2.05 | 1.02–1.66 | 1.04–1.68 | 1.01–1.76 | — |
| Days inside vol band | 15/20 | 4/20 | 10/20 | 7/20 | — |
| Mean mid range (ticks) | 6.95 | 6.30 | 5.15 | 6.00 | — |

**Every model is under-active relative to real markets** on uniq_mid — none comes close to the real
27–66 range, and the SS retrain did not improve this (which was its original stated goal). The SS
models are, if anything, slightly quieter than the baseline.

---

## 5. The central tension for discussion

The two evaluation axes disagree about which model is better, and neither obviously subsumes the other:

**The case for the SS models (e3 / e4)**
- Lower LOB-Bench distance on 20/20 days, ~26% better grand mean
- Improvement is consistent across every day, not outlier-driven
- LOB-Bench is the established external benchmark this project is scored against
- Equally stability-confirmed (20/20 days, zero timeouts)
- Faster to simulate

**The case for the baseline (0.724 e0)**
- Notably closer to real **return volatility**: mean 1.69bp vs 1.39–1.44bp, and 15/20 days inside the
  real 1.5–2.5bp band versus 4–10/20 for the SS models
- Highest uniq_mid (least under-active), though all models fall well short of real
- 71% of the SS models' LOB-Bench advantage comes from a single metric (spread)
- Longest-standing and most-tested checkpoint; it is the one the earlier elimination search
  independently selected out of six candidates

**The crux**: LOB-Bench's six metrics measure order-flow microstructure (spread, timing, imbalance,
depth placement). **None of them directly measures return volatility.** So "SS models score better on
LOB-Bench" and "the baseline better reproduces realistic price volatility" are both true and not in
contradiction — they are simply different properties. Which matters more depends on what the simulator
is for.

Questions that may be worth putting to supervisors:

1. For a downstream **execution-agent / RL** use case, does realistic *price volatility* matter more
   than microstructure distributional fidelity, or the reverse? The agent experiences both, but
   arguably reacts most to price dynamics.
2. Is a 26% LOB-Bench improvement that is 71% attributable to one metric (spread) persuasive as a
   headline result, or should it be reported per-metric to avoid over-claiming?
3. Is the volatility-band regression (15/20 → 7/20 days) a material cost, or acceptable noise given
   every model already under-shoots real activity substantially?
4. Does it strengthen or weaken the dissertation narrative to change the final model *after* the
   elimination search already selected the baseline through a documented, systematic procedure?

---

## 6. What is not yet known

- **No full-month DDPM baseline.** DDPM (the un-accelerated 100-step reference) has only been run on
  a single day, 2015-01-30, scoring 0.507 there against the baseline's 0.447. Every full-month figure
  above therefore lacks a like-for-like un-accelerated comparison. This is the single biggest
  outstanding gap and needs GPU time (~15–20h).
- **No predictive-score (MAE) comparison.** This is the TRADES paper's *own* headline metric. The
  infrastructure exists in this repo and has been run once (0.4416 vs a 0.4996 real-on-real baseline)
  but never across models or days. Running it across these four candidates would add an independent
  third axis, and would likely be decisive where LOB-Bench and volatility disagree.
- **Epochs beyond 5 untested.** The retrain was stopped at epoch 5 after its timeout. Whether the
  LOB-Bench trend continues improving, plateaus, or reverses past that point is unknown.
- **One stock, one month.** Everything here is INTC, January 2015.

---

## 7. Suggested next step before deciding

Running the **predictive score across all four candidates** is comparatively cheap (LSTM training on
CSVs that already exist locally; no new simulation required) and would give an independent tiebreaker
on the exact axis where the current two measures disagree. It is also the metric the source paper
leads with, so having it strengthens the write-up regardless of which model is chosen.

---

## 8. Sources

| Data | Location |
|---|---|
| Baseline per-day stability/timing | `ckpt_search/long_20260729_145408_progress.csv` |
| SS epochs per-day stability/timing | `ckpt_search/ss_retrain_epochs_detail.csv` |
| Baseline LOB-Bench, per day × metric | `lob_bench_0724_full_month/SUMMARY_mean_wasserstein.csv` |
| SS LOB-Bench, per day × metric | `lob_bench_ss_retrain/epoch{2,3,4}_*/SUMMARY_mean_wasserstein.csv` |
| Figures (per-metric, grand mean, reconciliation) | `analysis/plots/hsbc_ss_retrain_20260805/` |
| Stylized-facts batteries (2015-01-29, all 4 models) | `stylized_custom_outputs/INTC_2015_01_29_*.png` |
| Fuller appendix write-ups | `analysis/appendix_checkpoint_evidence.md`, `analysis/appendix_lobbench_and_refutations.md` |
