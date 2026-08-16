# Order-flow & market-order data (INTC 2015-01-30, generation phase)

Figures: `flow_mix_comparison.png`, `market_decode_rate.png`.

## Flow composition — measurable in all CSVs (real, their released, ours)

| source | Limit % | Cancel % | Executed % |
|---|---|---|---|
| Real (market replay) | 49.2 | 43.8 | 7.0 |
| Their TRADES-LOB, INTC 29 (DDPM) | 46.4 | 36.4 | 17.2 |
| Their TRADES-LOB, INTC 30 (DDPM) | 46.4 | 36.2 | 17.3 |
| Their TRADES-LOB, TSLA 29 (DDPM) | 45.9 | 39.4 | 14.7 |
| Their TRADES-LOB, TSLA 30 (DDPM) | 46.6 | 39.4 | 14.0 |
| Ours: DDPM-100 (ckpt 0.681) | 47.2 | 34.0 | 18.8 |
| Ours: DDIM-10 η=0 (frozen) | 59.4 | 35.8 | 4.8 |
| Ours: DDIM-10 η=1 | 47.8 | 36.5 | 15.7 |
| Ours: HYBRID DDPM→PP (8+2) | 49.1 | 39.1 | 11.8 |
| Ours: DDIM-10 depth-temp 2.0 | 40.4 | 24.0 | 35.7 |

Note: even their released gold-standard **over-executes vs real** (14–17% vs 7%). "Executed"
bundles market orders and marketable-limit executions; it cannot be split from the output CSV.

## Market-order decode rate — our internal diagnostic only (pre-drop)

Not measurable in any released CSV (market orders fold into "executed"). Real reference ≈ 2.8%
(open-loop next-event market fraction).

| run (ckpt 0.681 unless noted) | market decode % |
|---|---|
| DDPM-100 + prior | 0.08 |
| DDIM-10 η=0 (frozen) | 2.7 |
| DDPM-100 | 7.9 |
| DDIM-10 η=1 | 24.1 |
| DDIM-10 η=0 (ckpt 0.719) | 24.8 |
| HYBRID DDPM→PP (8+2) | 26.0 |
| DDIM-10 depth-temp 2.0 | 29.3 |

Takeaway: **DDPM is the only sampler that moves the market at a realistic market-order rate
(~8%)**; every other configuration that produces movement does so by inflating market/aggressive
orders to ~25–29% (≈9× real), which is the drift mechanism.
