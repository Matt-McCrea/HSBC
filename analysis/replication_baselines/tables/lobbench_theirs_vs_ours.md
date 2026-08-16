# LOB-Bench: TRADES released vs our replication

Wasserstein distance to real (lower = more realistic). `*` = excluded from the mean, see note.


## INTC 2015-01-30

| metric | TRADES released (DDPM) | Our DDPM-100 |
|---|---|---|
| spread | 1.5641 | 1.0193 |
| book imbalance | 0.2406 | 0.2132 |
| flow imbalance | 0.3039 | 0.4535 |
| limit depth | 0.8932 | 1.0432 |
| cancel depth | 0.9900 | 1.1409 |
| inter-arrival* | 1.0377 | 0.5749 |
| **mean (excl. inter-arrival)** | **0.7983** | **0.7740** |
| mean (all six) | 0.8382 | 0.7408 |

## INTC 2015-01-29

| metric | TRADES released (DDPM) | Our DDIM-1 |
|---|---|---|
| spread | 1.6537 | 0.9962 |
| book imbalance | 0.3195 | 0.1664 |
| flow imbalance | 0.2830 | 0.2608 |
| limit depth | 0.9350 | 0.9620 |
| cancel depth | 1.0832 | 0.9801 |
| inter-arrival* | 1.0434 | 0.3890 |
| **mean (excl. inter-arrival)** | **0.8549** | **0.6731** |
| mean (all six) | 0.8863 | 0.6258 |


`*` The released TRADES-LOB CSVs carry only 0.1s timestamp resolution (32% distinct stamps; up to 27 events share one), against microsecond resolution in our runs. This systematically inflates their inter-arrival distance and is an artifact of the released file format, not of their model. Exclude it from any headline comparison.
