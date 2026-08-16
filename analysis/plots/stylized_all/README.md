# Stylised-fact batteries — all runs, one place

All computed on the same convention: **1-minute bars for the lag panels** (matching the TRADES
paper's "Lag (minutes)" 0-30 axis), **1-second bars for the mid-price trace**. Real references are
window-aligned to each generated session, so both sides cover the identical market period.
Lag range auto-caps at ~40% of available bars, so 30-minute sessions show ~11 lags rather than 30.

## Two-hour runs (INTC 2015-01-29, 10:00-12:00)

| File | Model |
|---|---|
| `2h_ss_epoch4.png` | SS epoch 4 — the long-horizon winner |
| `2h_0724.png` | 0.724 baseline |
| `2h_trades_singlestep.png` | TRADES single-step — the collapse |

## 30-minute runs (INTC 2015-01-29, 09:30-10:00)

| File | Model |
|---|---|
| `30min_0724.png` | 0.724 + decode fixes |
| `30min_ss_epoch2.png` | SS epoch 2 |
| `30min_ss_epoch3.png` | SS epoch 3 |
| `30min_ss_epoch4.png` | SS epoch 4 |
| `30min_vanilla_ddim1_sse4.png` | SS e4, **no decode fixes**, 1 step |
| `30min_vanilla_ddim10_sse4.png` | SS e4, **no decode fixes**, 10 steps — shows no freeze |

## Replication baselines

In `analysis/replication_baselines/figures/`:
`stylized_THEIRS_ddpm_INTC_0129.png`, `stylized_THEIRS_ddpm_INTC_0130.png`,
`stylized_OURS_ddpm100_INTC_0130.png`, `stylized_OURS_ddim1_INTC_0129.png`

## Note

`stylized_custom_outputs/` is gitignored and its contents have been wiped more than once. Everything
here is tracked in git instead. Regenerate any of these with
`evaluation/stylized_custom/paper_style_stylized_facts.py <real> <gen> <out.png>`, building the real
reference via `evaluation/stylized_custom/lobster_real_reference.py`.
