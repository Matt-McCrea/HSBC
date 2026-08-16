# LOB-Bench, overnight paper runs

## Two-hour horizon (INTC 2015-01-29, 10:00-12:00)

| metric | 0.724 baseline | SS epoch 4 |
|---|---|---|
| spread | 0.5752 | 0.0773 |
| inter-arrival | 0.3711 | 0.2465 |
| book imbalance | 0.3714 | 0.3769 |
| flow imbalance | 0.5217 | 0.5196 |
| limit depth | 0.1578 | 0.1965 |
| cancel depth | 0.1347 | 0.1225 |
| **grand mean** | **0.3553** | **0.2566** |

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
