# TRADES replication failure — summary numbers

INTC 2015-01-29, 10:00-12:00, 1-step DDIM, checkpoint val_ema=0.763.
Boundary reference: $33.50.

| series | start | end | min | max | first below boundary | max after | recovers? | worst 60s move |
|---|---|---|---|---|---|---|---|---|
| real | 33.74 | 33.76 | 33.47 | 34.12 | 11:12:53 | 33.81 | True | -0.220 |
| replication | 33.74 | 31.86 | 31.70 | 34.12 | 11:13:41 | 33.08 | False | -1.180 |
