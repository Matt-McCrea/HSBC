# The published TRADES default at two hours — INTC 2015-01-29 and 2015-01-30

*Generated 2026-08-24 from the overnight vanilla DDPM-100 runs.*

## What is new here

Every earlier row in this folder labelled "TRADES" was **not** the published configuration:

| earlier row | what it actually was | why it could not carry the comparison |
|---|---|---|
| "DDPM-100" in `lobbench_tables.md` | step-count ablation on **our** checkpoint `0.724` | carries `PRICE_REANCHOR` + `UNCLAMP_DEPTH`, and the decode-time flags |
| "TRADES single-step" in fig. 1 | DDIM-**1** on `val_ema=0.763` | one step is our acceleration proposal, not the paper's sampler |

These two runs are the real thing: **DDPM, 100 steps, checkpoint `val_ema=0.667`, no decode-time
flags at all**, on both held-out test days, at the full two-hour horizon.

Source: `ABIDES/log/paper_runs_downloaded/world_agent_INTC_2015-01-{29,30}_12-00-00_30_DDPM_0.0_100_val_ema=0.667/`.
The folder name's `12-00-00_30` is the **end** stamp; the sessions run **10:00–12:00**.

---

## Headline

**LOB-Bench grand mean, INTC 2015-01-29, 10:00–12:00**

| | grand mean Wasserstein |
|---|---|
| TRADES default (DDPM-100) | 0.533 |
| Ours: 0.724 baseline (DDIM-10) | 0.355 |
| **Ours: SS epoch 4 (DDIM-10)** | **0.257** |

**51.8% closer to real than the published default, at 12.1× lower per-order cost.**

This replaces the previous framing, which could only claim an improvement over our own earlier
checkpoint. It is now a claim against the published method.

---

## LOB-Bench, per metric

`11_lobbench_vs_default_0129.png`

| metric | TRADES default | 0.724 baseline | SS epoch 4 |
|---|---|---|---|
| spread | 1.033 | 0.575 | **0.077** |
| inter-arrival | 0.373 | 0.371 | **0.246** |
| book imbalance | **0.174** | 0.371 | 0.377 |
| flow imbalance | **0.267** | 0.522 | 0.520 |
| limit depth | 0.631 | **0.158** | 0.197 |
| cancel depth | 0.717 | 0.135 | **0.123** |
| **grand mean** | 0.533 | 0.355 | **0.257** |

### Decomposition — report this alongside the headline

Of the 1.656 total metric gap between SS epoch 4 and the default:

| metric | contribution | share |
|---|---|---|
| spread | +0.956 | **+57.7%** |
| cancel depth | +0.594 | +35.9% |
| limit depth | +0.434 | +26.2% |
| inter-arrival | +0.127 | +7.6% |
| book imbalance | −0.202 | **−12.2%** |
| flow imbalance | −0.253 | **−15.3%** |

**The default is genuinely better on both imbalance metrics.** Spread and depth supply 120% of the
gain; the imbalances give 28% back. This is a trade, not a sweep, and stating it that way is
stronger than the headline alone — it matches the 72%-spread decomposition already recorded for the
earlier INTC comparison, so the pattern is consistent rather than a one-off.

**Second test day** (2015-01-30, default only — our checkpoints were scored on 01-29):

| metric | TRADES default |
|---|---|
| spread | 0.833 |
| inter-arrival | 0.408 |
| book imbalance | 0.045 |
| flow imbalance | 0.261 |
| limit depth | 0.835 |
| cancel depth | 0.857 |
| **grand mean** | **0.540** |

0.540 against 01-29's 0.533 — the default's weakness is stable across both test days, not a
single-day artefact.

---

## Price-path statistics

`6_longhorizon_2h_vs_default.png`, `7_longhorizon_2h_all.png`, `8_longhorizon_2h_0130_vs_default.png`

**Two conventions, deliberately.** Price-path columns are **full session**, matching the headline
table in `README.md` and what the figures plot. **VR and nz% discard a 15-minute warm-up**, the
convention set in `variance_ratio_analysis.md`. The distinction is not cosmetic: on SS epoch 4,
full-session VR(60s) is 0.521 against 0.050 post-warm-up, because the opening minutes are dominated
by the model settling out of its conditioning block and that inflates the one-period variance.
Always state which convention a VR figure uses.

### INTC 2015-01-29, 10:00–12:00

| series | start | end | min | max | range (tk) | ret1s_std (bp) | uniq mids | VR(10s) | VR(60s) | VR(300s) | nz % | exec % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real market | 33.97 | 33.99 | 33.47 | 34.02 | 56 | 1.26 | 92 | 0.902 | 0.816 | 0.776 | 14.8 | 3.4 |
| Ours: SS epoch 4 | 33.74 | 34.09 | 33.70 | 34.12 | 41 | 1.65 | 64 | 0.213 | **0.050** | 0.012 | 31.0 | 12.5 |
| Ours: 0.724 baseline | 33.74 | 33.91 | 33.70 | 34.12 | 41 | 1.96 | 62 | 0.289 | 0.085 | 0.032 | 48.8 | 7.2 |
| TRADES default (DDPM-100) | 33.74 | 34.37 | 33.70 | 34.38 | 68 | **4.04** | 117 | 0.461 | **0.188** | 0.110 | 71.2 | 11.9 |
| TRADES single-step (DDIM-1) | 33.74 | 31.86 | 31.70 | 34.12 | 242 | 4.73 | 307 | 0.767 | 0.822 | 0.840 | 71.7 | 11.6 |

### INTC 2015-01-30, 10:00–12:00

| series | start | end | min | max | range (tk) | ret1s_std (bp) | uniq mids | VR(10s) | VR(60s) | VR(300s) | nz % | exec % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real market | 33.52 | 33.52 | 33.35 | 33.69 | 33 | 1.04 | 62 | 0.795 | 0.731 | 0.802 | 11.3 | 3.2 |
| Ours: SS epoch 4 | 33.81 | 34.06 | 33.62 | 34.06 | 45 | 1.59 | 82 | 0.214 | **0.044** | 0.011 | 34.5 | 10.7 |
| Ours: 0.724 baseline | 33.81 | 33.87 | 33.62 | 33.98 | 37 | 1.65 | 67 | 0.307 | 0.078 | 0.023 | 39.2 | 7.0 |
| TRADES default (DDPM-100) | 33.81 | **35.81** | 33.62 | 35.84 | **222** | **5.24** | 277 | 0.709 | **0.574** | 0.488 | 72.7 | 11.7 |

These reproduce `variance_ratio_analysis.md` on the model rows — SS epoch 4 VR(60s) 0.050 against
the recorded 0.047, 0.724 baseline 0.085 against 0.098. The **real** row differs more (0.816 against
0.931) because the reference here is `lobster_real_reference.py` rather than the ABIDES market
replay; the two are known to differ slightly, and the LOBSTER-derived one is what the rest of this
folder now uses.

### How the default fails

It does **not** freeze. It is over-volatile and it drifts:

- **One-second volatility 3.2–5.0× real** (4.04 / 5.24 bp against 1.26 / 1.04 bp). Our corrected
  configuration sits at 1.65 / 1.59 bp — the closest of any configuration tested.
- **Price range 222 ticks on 01-30 against the real day's 33.** On 01-29 it is 68 against 56, so
  the severity is day-dependent, but the direction is the same on both.
- **A single +$0.94 second at minute 39 on 01-30 supplies 47% of the $2.00 end-of-session gap.**
  The rest accrues as drift. On 01-29 there is no jump at all — the largest one-second move is
  9.5 c — so that day is pure accumulated drift.

### The counterweight: the default mean-reverts *less* than we do

This must be stated plainly, because it cuts against the headline.

| VR(60s), post warm-up | 01-29 | 01-30 |
|---|---|---|
| Real market | 0.816 | 0.731 |
| **TRADES default (DDPM-100)** | **0.188** | **0.574** |
| Ours: 0.724 baseline | 0.085 | 0.078 |
| Ours: SS epoch 4 | 0.050 | 0.044 |

**On both test days the published default is closer to real on price persistence than either of our
configurations**, by 3.8× on 01-29 and 13× on 01-30. Our models accumulate under 5% of the variance
a random walk would at the one-minute horizon; the default manages 19–57%.

Two things stop this from overturning the result, and both belong in the write-up:

1. **The default buys persistence with drift.** Its 01-30 VR of 0.574 comes from a 222-tick
   traverse — a trend, and a trend scores well on VR by construction. Ranking on VR alone would
   place the DDIM-1 single-step run (0.822, 0.840) at the top of the table, and that run collapses
   $2 through the floor. VR must be read with the range and volatility columns, never alone.
2. **The mean reversion is inherited, not introduced.** `variance_ratio_analysis.md` already records
   VR(60s) of 0.099 and 0.153 for TRADES's *own released output*. Our 0.044–0.085 is worse than
   this checkpoint's 0.188–0.574, so the decode-time corrections and scheduled sampling do deepen
   it — but they are deepening a pathology that is present in the architecture to begin with.

**Read VR alongside nz%.** The default's nz% is 71–73% against real's 11–15%, so its VR figures rest
on a dense return series and are not subject to the sparse-jump bias that made the earlier vanilla
VR = 0.956 reading meaningless (55 non-zero returns out of 877). All rows in these tables have
nz% > 11, so every VR figure here is on a sound sample.

### Flow composition

| day | series | limit % | cancel % | market % |
|---|---|---|---|---|
| 01-29 | Real market | 49.7 | 46.9 | **3.4** |
| 01-29 | Ours: SS epoch 4 | 51.9 | 35.7 | 12.5 |
| 01-29 | Ours: 0.724 baseline | 51.3 | 41.5 | 7.2 |
| 01-29 | TRADES default | 48.1 | 40.0 | 11.9 |
| 01-29 | TRADES single-step | 48.2 | 40.1 | 11.6 |
| 01-30 | Real market | 49.4 | 47.4 | **3.2** |
| 01-30 | Ours: SS epoch 4 | 52.3 | 37.0 | 10.7 |
| 01-30 | Ours: 0.724 baseline | 51.6 | 41.4 | 7.0 |
| 01-30 | TRADES default | 48.2 | 40.1 | 11.7 |

**Every configuration over-executes**, the default included (11.9% against real 3.4%). The 0.724
baseline is closest at 7.0–7.2%. This is not a defect introduced by our interventions — it is
present in the published default at the same magnitude as in the scheduled-sampling model, so it is
architectural.

---

## Cost

| | per order | of which NN | throughput | wall clock, 2 h session |
|---|---|---|---|---|
| TRADES default (DDPM-100) | 139.14 ms | 132.64 ms | 7.2 /s | 13,170 s |
| Ours: SS epoch 4 (DDIM-10) | 11.52 ms | 11.07 ms | 86.8 /s | 2,628 s |
| **ratio** | **12.1×** | **12.0×** | **12.1×** | **5.0×** |

The per-order figure is the model-level acceleration and is the one to quote for the sampler claim.
**The 5.0× end-to-end figure is lower for a reason that must be stated**: the two runs did not
generate the same number of orders for the same simulated window — 94,654 for the default against
228,131 for ours. The default produces a thinner market, so it does less work per simulated second.
End-to-end wall clock is still the honest answer to "how long to simulate two hours of INTC", but
it is not a like-for-like throughput comparison.

---

## Stylised facts

`9_stylized_DDPMdefault_0129_2h.png`, `10_stylized_DDPMdefault_0130_2h.png` — house red/blue
palette, real in blue, generated in red.

The clearest failure is **volume–volatility correlation**: real holds ≈0.47 and decays slowly to
0.28 across 30 minutes of lag; the default starts at 0.21 and crosses into negative territory by
lag 4, oscillating between −0.20 and +0.10 thereafter. The relationship is absent rather than weak.
Return–volatility is also the wrong sign at short lag (positive to +0.06 out to lag 5, against
real's −0.13 to −0.07 throughout).

---

## Files

| file | shows |
|---|---|
| `6_longhorizon_2h_vs_default.png/.pdf` | 01-29, three series: real, SS epoch 4, TRADES default |
| `7_longhorizon_2h_all.png/.pdf` | 01-29, all five series — both TRADES configurations together |
| `8_longhorizon_2h_0130_vs_default.png/.pdf` | 01-30, where the drift is most severe |
| `9_stylized_DDPMdefault_0129_2h.png` | stylised-fact battery, default vs real, 01-29 |
| `10_stylized_DDPMdefault_0130_2h.png` | stylised-fact battery, default vs real, 01-30 |
| `11_lobbench_vs_default_0129.png/.pdf` | LOB-Bench per metric + grand mean, three configurations |
| `ddpm_default_stats.csv` | every number in the price-path tables |
| `real_0130_1000_1200.csv` | 01-30 real reference, built from LOBSTER (01-29's already existed) |

Reproduce with:

```bash
python3 scripts/ddpm_default_stats.py          # price-path, VR, flow tables
python3 scripts/make_longhorizon_fig_v2.py     # figures 1, 6, 7, 8
python3 scripts/make_lobbench_default_fig.py   # figure 11 + decomposition
# LOB-Bench scores (local, needs the py3.10 env):
external/lob_bench_env/bin/python evaluation/lob_bench/score_batch.py \
  --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-29_34140000_57660000_orderbook_10.csv \
  --gen "TRADES_DDPM100_2h=ABIDES/log/paper_runs_downloaded/world_agent_INTC_2015-01-29_12-00-00_30_DDPM_0.0_100_val_ema=0.667/processed_orders.csv" \
  --out-dir lob_bench_paper/default_0129 --window 10:00
```
