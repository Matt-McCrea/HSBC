# LOB-Bench, overnight paper runs

## Two-hour horizon (INTC 2015-01-29, 10:00-12:00)

*Third column added 2026-08-24: the genuine published default (DDPM-100, ckpt `val_ema=0.667`, no
decode-time flags). Full analysis and the metric-level decomposition in `trades_default_results.md`.*

| metric | TRADES default (DDPM-100) | 0.724 baseline | SS epoch 4 |
|---|---|---|---|
| spread | 1.0330 | 0.5752 | 0.0773 |
| inter-arrival | 0.3731 | 0.3711 | 0.2465 |
| book imbalance | 0.1744 | 0.3714 | 0.3769 |
| flow imbalance | 0.2669 | 0.5217 | 0.5196 |
| limit depth | 0.6309 | 0.1578 | 0.1965 |
| cancel depth | 0.7169 | 0.1347 | 0.1225 |
| **grand mean** | **0.5325** | **0.3553** | **0.2566** |

SS epoch 4 is **51.8% closer to real than the published default**. The gain is not uniform: spread
(+57.7% of the total gap) and depth (+62.1%) supply it, while the default is genuinely better on
both imbalance metrics (−27.5% combined). Quote the decomposition with the headline.

Same configuration on the other test day (2015-01-30): grand mean **0.5403**, so the default's
weakness is stable across both held-out days.

## Step-count ablation (INTC 2015-01-30, ckpt 0.724, same decode config)

| metric | DDPM-100 | DDIM-10 |
|---|---|---|
| spread | 0.9633 | 0.6249 |
| inter-arrival | 0.5585 | 0.6706 |
| book imbalance | 0.4285 | 0.3838 |
| flow imbalance | 0.6426 | 0.6643 |
| limit depth | 0.4586 | 0.1722 |
| cancel depth | 0.3974 | 0.1639 |
| **grand mean** | **0.5748** | **0.4466** |

Ten steps score *better* than one hundred on the same checkpoint. This is a step-count ablation on our own model, not a comparison against the published TRADES configuration.

**The comparison against the published configuration now exists** — see the first table above and
`trades_default_results.md`. Keep the two separate: this table isolates the effect of step count
with everything else held fixed; that one compares whole systems.
