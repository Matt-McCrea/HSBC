# Long-horizon runs — results brief

*7 August 2026. Overnight GPU session. INTC, January 2015.*

---


These are two-hour runs (10:00–12:00, 2015-01-29), matching the window in which the collapse happens.

---

## Result 1 — Both models hold for two hours

`figures: 1_longhorizon_2h.png`

| | end price | price range | 1s volatility | unique mids |
|---|---|---|---|---|
| **Real market** | 33.99 | 56 tk | 1.26 bp | 92 |
| **Ours: SS epoch 4** | 34.09 | 41 tk | 1.65 bp | 64 |
| **Ours: 0.724** | 33.91 | 41 tk | 1.96 bp | 62 |
| TRADES single-step | **31.86** | 242 tk | 4.73 bp | 307 |

The single-step replication crosses the conditioning boundary and never recovers — it ends \$2.13
below the real market. Both of our configurations stay inside the real envelope for the full session.


---

## Result 2 — Scheduled sampling pulls clearly ahead at long horizon

`figures: 2_longhorizon_lobbench.png`

LOB-Bench, two-hour window (lower = more realistic):

| metric | 0.724 baseline | **SS epoch 4** |
|---|---|---|
| spread | 0.575 | **0.077** |
| inter-arrival | 0.371 | **0.247** |
| book imbalance | **0.371** | 0.377 |
| flow imbalance | 0.522 | **0.520** |
| limit depth | **0.158** | 0.197 |
| cancel depth | 0.135 | **0.123** |
| **grand mean** | 0.355 | **0.257** |

Activity, measured after warm-up: SS reaches **57 unique mids** against the baseline's **29**
(real: 113) — roughly twice the market activity.

### Why this matters for the write-up

At **30 minutes** the two models are near-identical (33 vs 33 unique mids). The advantage only
emerges over a longer horizon.

This **partly reverses our earlier conclusion that scheduled sampling had failed to raise activity**.
That verdict was drawn from 30-minute evaluations only. The benefit is real — it was invisible at the
horizon we had been testing.

**Counterweight to state honestly**: SS over-executes more (12.5% of orders vs baseline 7.2%;
real is 3.4%).

---

## Result 3 — Ten denoising steps beat one hundred

`figures: 3_stepcount_ablation.png`

Same checkpoint, same decode configuration, only the step count differs (INTC 2015-01-30):

| | grand mean Wasserstein |
|---|---|
| DDPM, 100 steps | 0.575 |
| **DDIM, 10 steps** | **0.447** |

Acceleration is a fidelity **gain**, not a trade-off — now demonstrated on a current checkpoint
rather than the superseded one.

> **Note**: this is a step-count ablation on our own model. It is *not* a comparison against the
> published TRADES configuration — our checkpoint carries data-pipeline fixes that theirs does not.
> The replication against TRADES is a separate exercise.

---
---

## Where things stand

| | |
|---|---|
| Replication of TRADES | done — our DDPM-100 within 3% of their released model on LOB-Bench |
| Cross-day robustness | done — 20 trading days, zero failures |
| **Long-horizon stability** | **done — two hours, both candidates hold** |
| Acceleration claim | done — 10 steps beat 100 on a fixed checkpoint |
| Final model choice | open — SS epoch 4 now has materially stronger evidence |

**Not yet run**: long-horizon *with* the book-balancing lever (unknown whether the shipped model
still needs it), and seed robustness on the final models. Roughly 1.5 h each.

Full detail, figures and raw scores: `analysis/plots/longhorizon/`
