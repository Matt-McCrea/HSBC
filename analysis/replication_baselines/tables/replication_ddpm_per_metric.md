# Per-metric LOB-Bench — DDPM-100 replication, INTC 2015-01-30

*Source: `analysis/replication_baselines/lob_bench/INTC_2015-01-30/lob_bench_scores.csv`.
Wasserstein-1 distance to real; lower is more realistic. The CI columns in the source file are
bootstrap intervals and are omitted here for readability.*

| Metric | THEIRS (released DDPM) | OURS (DDPM-100, ckpt 0.681) | Winner |
|---|---|---|---|
| spread | 1.564 | **1.019** | ours |
| log inter-arrival | 1.038 | **0.575** | ours *(see caveat)* |
| orderbook imbalance | 0.241 | **0.213** | ours |
| orderflow imbalance | **0.304** | 0.454 | theirs |
| limit depth (ask) | **0.893** | 1.043 | theirs |
| cancel depth (ask) | **0.990** | 1.141 | theirs |
| **mean, excl. inter-arrival (5)** | **0.798** | **0.774** | ours, by 3% |
| mean, all six | 0.838 | 0.741 | ours, by 12% |

## Both headline figures are FIVE-metric means

Verified by arithmetic from the source CSV: 3.991745 / 5 = **0.7983** (theirs) and
3.870138 / 5 = **0.7740** (ours). The comparison is like-for-like.

**Excluding inter-arrival works against us**, which is worth saying explicitly. Their inter-arrival
score is 1.038 against our 0.575 — the single metric where the gap is widest in our favour — so
dropping it narrows our margin from 12% to 3%. We exclude it because their released CSVs carry only
0.1-second timestamp resolution, which inflates that metric as a file-format artefact rather than a
model property. Stating that the exclusion is the *conservative* choice pre-empts any suggestion the
metric set was chosen to flatter the result.

## What the per-metric split shows that the mean hides

The near-tie in the grand mean (0.798 vs 0.774) conceals genuine differences in **both** directions:
we are substantially better on spread and book imbalance, and worse on order-flow imbalance and on
both depth metrics. The honest reading is **not** "our reproduction is slightly better" but "our
reproduction lands in the same regime, with a different error profile" — which is the stronger claim
for a replication anyway, since it demonstrates faithful reproduction rather than improvement.

## 2015-01-29, for the appendix

| | THEIRS (released DDPM) | OURS (DDIM-1, ckpt 0.763) |
|---|---|---|
| mean, excl. inter-arrival (5) | 0.855 | 0.673 |
| mean, all six | 0.886 | 0.626 |

⚠️ These are **not** a replication pair — theirs is DDPM-100, ours is single-step, and the ours row
is the run that diverges \$1.90 from real. That is the LOB-Bench blind-spot result
(`lobbench_blindspot_ddim1_0129.png`), not a like-for-like comparison.

## ⚠️ Open: window provenance for the OURS row

`WRITEUP_NUMBERS.md` records **two distinct DDPM-100 runs**, both labelled ckpt 0.681:

| | "Documented replication" | `sweep_results/DDPM_100/` |
|---|---|---|
| Window | 30-min | 09:30–10:30 (60-min) |
| Executed % | 18.8 | 10.17 |
| Unique mids | 23 | 101 |

`like_for_like_comparison.md` quotes the **sweep_results** numbers (101 unique mids, 09:30–10:30);
`WRITEUP_NUMBERS.md` says to cite the *documented* 30-minute one. **Which CSV produced the 0.774
score above is not recorded** — the `lob_bench/INTC_2015-01-30/` directory retains only the scores,
with no run log, and `data/` holds only the THEIRS files.

**Resolve before this table goes to print.** Their released output covers 09:45–11:00, so if our row
was scored over 09:30–10:30 the two are measured on overlapping but non-identical windows. That is
defensible — each is scored against its own matched real slice, which is standard — but it must be
stated in the caption rather than left implicit.

## Real reference, 2015-01-30

From `like_for_like_comparison.md`, window 09:45–11:00:

| | Real |
|---|---|
| 1s return volatility | **1.14 bp** |
| variance ratio (60 s) | 0.724 |
| unique mid-prices | 63 |
| mid-price range | 31 ticks |
| flow limit/cancel/exec | 49 / 47 / 3 % |

The 1.14 bp figure is the real volatility for that day and is the missing third leg of the
4.35 (theirs) / 4.54 (ours) pair — **both models are roughly 4× more volatile than real**, which is
the point worth making from that row.
