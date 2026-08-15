# SS-retrain comparison tables (2026-08-05)

## Per-metric, full-month mean Wasserstein

| metric | baseline (epoch 0) | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|
| spread | 0.719 | 0.285 | 0.238 | 0.192 |
| inter-arrival | 0.625 | 0.537 | 0.514 | 0.504 |
| book imbalance | 0.432 | 0.400 | 0.389 | 0.394 |
| flow imbalance | 0.673 | 0.652 | 0.647 | 0.645 |
| limit depth | 0.200 | 0.170 | 0.185 | 0.185 |
| cancel depth | 0.161 | 0.144 | 0.161 | 0.152 |
| **grand mean** | **0.468** | **0.365** | **0.356** | **0.345** |

## Activity level vs overall realism (the reconciliation)

| | epoch 0 (baseline) | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|
| mean uniq_mid | 15.10 | 13.65 | 11.30 | 12.95 |
| LOB-Bench grand mean | 0.468 | 0.365 | 0.356 | 0.345 |
