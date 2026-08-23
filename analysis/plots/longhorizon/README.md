# Long-horizon & overnight paper runs — results folder

*2026-08-07. From the overnight GPU session. All five phases completed: P1 sampler ablation,
P2 two-hour runs, P3 step-count ablation, P4 book-balancing lever, P5 seed robustness. The single
exception is the `0.724` seed-32 cell, which did not finish before the window closed.*

> **Update 2026-08-24 — the published TRADES default is now measured.**
> Vanilla DDPM-100 (ckpt `val_ema=0.667`, no decode flags) ran two hours on **both** test days.
> LOB-Bench grand mean **0.533** (01-29) and **0.540** (01-30) against SS epoch 4's **0.257** —
> **51.8% closer to real, at 12.1× lower per-order cost.** It fails by over-volatility and drift,
> not by freezing: 4.04–5.24 bp one-second volatility against real 1.04–1.26, and a 222-tick range
> on 01-30 against the real day's 33.
> Full analysis, decomposition and figures 6–11: **`trades_default_results.md`**.
> Every "TRADES" row *below this line* is either a step-count ablation on our own checkpoint or the
> DDIM-1 single-step run — neither is the published default. Read them as such.

---

## The headline

**Both final-model candidates survive two hours.** Every prior evaluation stopped at 30 minutes —
an hour before the failure mode the fix exists to address even appears.

`1_longhorizon_2h.png`

| series | end price | range | ret1s_std | unique mids |
|---|---|---|---|---|
| Real market | 33.99 | 56 tk | 1.26 bp | 92 |
| **SS epoch 4** | **34.09** | 41 tk | 1.65 bp | 64 |
| 0.724 baseline | 33.91 | 41 tk | 1.96 bp | 62 |
| TRADES single-step | **31.86** | 242 tk | 4.73 bp | 307 |

The single-step replication crosses the conditioning boundary at ~minute 73 and never recovers;
both corrected configurations stay inside the real envelope for the full session.

---

## Scheduled sampling wins clearly at long horizon

`2_longhorizon_lobbench.png`, `lobbench_tables.md`

| metric | 0.724 baseline | SS epoch 4 |
|---|---|---|
| spread | 0.575 | **0.077** |
| inter-arrival | 0.371 | **0.247** |
| book imbalance | **0.371** | 0.377 |
| flow imbalance | 0.522 | **0.520** |
| limit depth | **0.158** | 0.197 |
| cancel depth | 0.135 | **0.123** |
| **grand mean** | 0.355 | **0.257** |

**This is new and it matters.** At 30 minutes the two models are near-indistinguishable (33 vs 33
unique mids). Over two hours SS epoch 4 pulls clearly ahead — 0.257 against 0.355, and roughly twice
the market activity (57 vs 29 unique mids, measured post-warm-up; real 113).

It also **partly reverses the earlier conclusion that scheduled sampling failed to raise activity**.
That verdict came from 30-minute evaluations only. The benefit is real; it was simply invisible at
the horizon we had been testing. Note the counterweight: SS over-executes more (12.5% vs 7.2%,
real 3.4%).

---

## Ten steps beat one hundred

`3_stepcount_ablation.png`

INTC 2015-01-30, checkpoint 0.724, identical decode config, only step count differs:

| | grand mean Wasserstein |
|---|---|
| DDPM, 100 steps | 0.575 |
| **DDIM, 10 steps** | **0.447** |

⚠️ **This is a step-count ablation on our own model — NOT a TRADES baseline.** This checkpoint
carries `PRICE_REANCHOR` and `UNCLAMP_DEPTH`, so it cannot stand in for the published pipeline. The
replication against TRADES is separate, in `analysis/replication_baselines/`.

---

## ⚠️ The freeze does not occur on the current checkpoints

The most consequential finding for how §5.2/§5.3 can be written.

Vanilla DDIM-10 (no depth-noise, no size-reshape, no type-decode), 30-min, INTC 2015-01-29:

| checkpoint | unique mids | ret1s_std | verdict |
|---|---|---|---|
| 0.627 (historical) | **3** | 0.15 bp | frozen |
| 0.724 | **33** | 1.07 bp | alive |
| SS epoch 4 | **32** | 1.10 bp | alive |
| *real reference* | 39 | 1.52 bp | — |

**Both current checkpoints produce a live market without any decode-time correction.** So on these
models the fix is not rescuing from freeze — it is adjusting volatility:

| | vanilla | + decode fixes | real |
|---|---|---|---|
| SS epoch 4 | 1.10 bp | **1.65 bp** | 1.52 bp |
| 0.724 | 1.07 bp | 2.05 bp | 1.52 bp |

On SS epoch 4 the fix moves volatility toward real (1.10 → 1.65 against a 1.52 target). On 0.724 it
overshoots (1.07 → 2.05). And on execution share the vanilla runs are *closer* to real
(5.4–5.7% vs 6.9–9.0%, real 3.7%).

### It is not the price reanchoring

`0.627` **also** had `PRICE_REANCHOR` and `UNCLAMP_DEPTH` and still froze, so the data-pipeline fixes
cannot be what distinguishes them.

### Which checkpoint — and a candidate explanation

**Both** current checkpoints. SS epoch 4 descends from 0.724, and 0.724 already fails to freeze, so
the non-freeze is inherited rather than created by scheduled sampling.

What differs is the **training run**. Verified via `scripts/check_checkpoint_configs.py`:

| checkpoint | conditional dropout | learning rate |
|---|---|---|
| 0.724 family (current) | **0.1** | 2.5e-4 |
| 0.7_epoch=2 (older lineage) | **0.0** | 1e-3 |

Conditional dropout is a plausible mechanism — the freeze was diagnosed as *depth-variance collapse
under deterministic sampling*, and dropout during training is exactly the kind of regulariser that
would leave more variance in the decoded output.

**But this is a hypothesis, not a verified cause.** `0.627` is not retained, so its dropout setting
cannot be checked directly, and the two runs differ in learning rate as well. State it as a
candidate explanation, not a finding.

### How to frame it honestly

This does **not** invalidate the freeze result — it is consistent with the project's own
checkpoint-hypersensitivity finding (0.681 froze, 0.719 exploded, same procedure). The correct claim
is that the freeze was necessary to fix **for 0.627**, not that it is universal. Worth stating
plainly; it is a stronger paper for acknowledging that the decode-time correction's role changed as
the training pipeline improved.

---

## Variance-ratio analysis — read this before quoting the activity numbers

`variance_ratio_analysis.md`

The two-hour traces look flat because the models **mean-revert violently**: VR(60s) = 0.047 (SS e4)
and 0.098 (0.724) against real's 0.931. Both have *more* 1-second volatility than real yet half the
price range — the movement cancels instead of accumulating.

Three things in there that matter:

- **TRADES's own released output has the same pathology** (VR(60s) 0.099 / 0.153), so it is inherited
  from the architecture rather than introduced here.
- **The book-balancing lever improves persistence** (VR(60s) 0.047 → 0.112, range 28 → 32 tk) —
  the opposite of what was predicted, since it is a mean-reverting mechanism by design. P4 result.
- **Seeds are very stable** (VR within 0.04 across three seeds), so the pathology is systematic, not
  noise. P5 result.

Next run: `scripts/drift_persistence_sweep.sh`, 12 cells, ~5.2 h.

## Files

| File | Shows |
|---|---|
| `1_longhorizon_2h.png` | mid-price traces, 2h, both candidates + single-step collapse + real |
| `2_longhorizon_lobbench.png` | per-metric LOB-Bench at 2h, baseline vs SS |
| `3_stepcount_ablation.png` | DDPM-100 vs DDIM-10 on a fixed checkpoint |
| `4_stylized_SSe4_2h.png` | stylised-fact battery, SS epoch 4 at 2h vs real |
| `5_stylized_0724_2h.png` | stylised-fact battery, 0.724 at 2h vs real |
| `summary.md` | price-path summary numbers |
| `lobbench_tables.md` | full per-metric LOB-Bench tables |

Raw scores: `lob_bench_paper/longhorizon_0129/`, `lob_bench_paper/stepcount_0130/`.
Source CSVs: `ABIDES/log/paper_runs_downloaded/`.

## Overnight completion

All five phases landed. **P4** (long-horizon with the book-balancing lever) and **P5** (seed
robustness) both completed and are analysed in `variance_ratio_analysis.md`. The only gap is the
`0.724` seed-32 cell, which did not finish before the window closed — immaterial, since the other
five seed cells agree closely.

---

## Files added 2026-08-24 (TRADES default)

| File | Shows |
|---|---|
| `trades_default_results.md` | **full analysis of the published-default runs — start here** |
| `6_longhorizon_2h_vs_default.png/.pdf` | 01-29: real, SS epoch 4, TRADES default |
| `7_longhorizon_2h_all.png/.pdf` | 01-29: all five series, both TRADES configurations together |
| `8_longhorizon_2h_0130_vs_default.png/.pdf` | 01-30, where the drift is most severe |
| `9_stylized_DDPMdefault_0129_2h.png` | stylised facts, default vs real, 01-29 |
| `10_stylized_DDPMdefault_0130_2h.png` | stylised facts, default vs real, 01-30 |
| `11_lobbench_vs_default_0129.png/.pdf` | LOB-Bench per metric + grand mean, three configurations |
| `ddpm_default_stats.csv` | every number behind the price-path tables |
| `real_0130_1000_1200.csv` | 01-30 real reference, built from LOBSTER |

Scripts: `scripts/ddpm_default_stats.py`, `scripts/make_longhorizon_fig_v2.py`,
`scripts/make_lobbench_default_fig.py`. Raw LOB-Bench: `lob_bench_paper/default_01{29,30}/`.
