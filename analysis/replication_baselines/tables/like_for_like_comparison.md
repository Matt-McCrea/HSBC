# Like-for-like replication comparison — theirs vs ours

*One table: market statistics and LOB-Bench, for TRADES's released output against our reproduction,
same instrument and same trading day.*

---

## ⚠️ What can and cannot be compared

TRADES released **one output file per stock-day** (`INTC_2015-01-29`, `INTC_2015-01-30`,
`TSLA_2015-01-29`, `TSLA_2015-01-30`) with no sampler distinction. These are their **main DDPM
model**. They did **not** release DDIM-1 output.

| Comparison | Possible? | |
|---|---|---|
| Their DDPM vs our DDPM | **yes** | 2015-01-30, both present |
| Their DDIM-1 vs our DDIM-1 | **no** | they published no DDIM-1 data — only the predictive score in their Table 2 |

So the DDIM-1 row below is **ours only**. The sole DDIM-1 figure they provide is a predictive score
(Intel: **0.486**, against their DDPM's **0.307**), which is a different metric on a different
protocol and not comparable to anything in this table.

---

## The table

LOB-Bench = mean Wasserstein distance to real, **excluding inter-arrival** (their released CSVs carry
only 0.1 s timestamp resolution, which inflates that metric — a file-format artefact, not model
behaviour). Statistics measured after a 15-minute warm-up. Flow = limit / cancel / executed %.

| Day | Model | Window | 1s vol (bp) | range (tk) | uniq mids | flow % | VR(60s) | **LOB-Bench** |
|---|---|---|---|---|---|---|---|---|
| 2015-01-30 | *Real market* | 09:45–11:00 | 1.14 | 31 | 63 | 49/47/3 | 0.724 | — |
| 2015-01-30 | **THEIRS** DDPM (released) | 09:45–11:00 | 4.35 | 23 | 46 | 46/36/17 | 0.099 | **0.798** |
| 2015-01-30 | **OURS** DDPM-100 (ckpt 0.681) | 09:30–10:30 | 4.54 | 97 | 101 | 49/41/10 | **0.835** | **0.774** |
| 2015-01-29 | *Real market* | 09:45–11:00 | 1.42 | 50 | 101 | 50/47/3 | 1.059 | — |
| 2015-01-29 | **THEIRS** DDPM (released) | 09:45–11:00 | 4.41 | 25 | 51 | 46/36/17 | 0.153 | 0.855 |
| 2015-01-29 | **OURS** DDIM-1 (ckpt 0.763) | 10:00–12:00 | 4.93 | 229 | 309 | 48/40/12 | 0.765 | 0.673 |

---

## Reading it

**The replication holds.** On 2015-01-30, the only true like-for-like row pair, our DDPM-100 scores
**0.774** against their released **0.798** — within 3%, measured against the same real data on the
same benchmark. Volatility matches closely too (4.54 vs 4.35 bp), as does flow composition.

**Both over-execute against real by a wide margin** — 10–17% executed against real's 3%. This is a
property of the *published* model, visible in their own released output, not something introduced
here. Stating it early is what licenses the diagnosis chapter.

**Our reproduction is closer to real on price-path behaviour.** Their released output is
strikingly range-bound (23–25 ticks against real's 31–50) and strongly mean-reverting
(VR(60s) 0.099 / 0.153 against real's 0.724 / 1.059). Ours random-walks much more naturally
(0.835 / 0.765) and covers a wider range. We did not set out to find this; it emerged from the
variance-ratio diagnostic.

**Caveat on windows.** Their released output covers 09:45–11:00; our DDPM-100 run covers
09:30–10:30 and our DDIM-1 run 10:00–12:00. Each is scored against *its own* matching real slice,
which is the standard approach, but this is a comparison of models each measured against real rather
than a strictly controlled head-to-head. The real rows are given at their window for reference.

**The DDIM-1 row is not a replication comparison.** With no released counterpart it stands alone, and
its wide range (229 tk) and high volatility reflect the divergence documented in
`ddim1_collapse_numbers.md` — it crosses the conditioning boundary at ~minute 73 and never recovers.
Its favourable LOB-Bench score (0.673) despite that collapse is the benchmark blind spot noted in the
folder README.

---

## To complete the grid

The only missing cell that *could* be filled is **our DDPM-100 on 2015-01-29** — checkpoint 0.681 is
retained, so a 75-minute run matching their window would give DDPM-vs-DDPM on both days rather than
one. Roughly 75 minutes of GPU.

Nothing can fill the their-DDIM-1 cell short of them releasing it.

---

## Sources

- Their released CSVs, timestamps repaired: `analysis/replication_baselines/data/`
- LOB-Bench raw scores: `analysis/replication_baselines/lob_bench/INTC_2015-01-{29,30}/`
- Our DDPM-100: `sweep_results/DDPM_100/`
- Our DDIM-1: `ABIDES/log/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763/`
- Their published numbers: `tables/paper_reported_numbers.md`
