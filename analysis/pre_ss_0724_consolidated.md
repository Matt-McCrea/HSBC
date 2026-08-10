# The pre-scheduled-sampling model — `val_ema=0.724_epoch=0`, consolidated

*2026-08-10. Everything held on the baseline checkpoint: full-month benchmark, 20-day stability,
long-horizon runs, seeds, timing, and the vanilla (no-decode-correction) control. This is the
model as it stood **before** the scheduled-sampling retrain.*

**Configuration throughout unless stated:** DDIM, 10 steps, η = 0, `--depth-noise 0.3
--size-reshape --type-decode prior`, `UNCLAMP_DEPTH` + `PRICE_REANCHOR`, seed 30, INTC,
09:30–10:00 window, statistics after a 09:45 warm-up cutoff.

---

## 1. Full-month LOB-Bench — the headline number

Per-day Wasserstein distance to real, all 20 trading days.
Source: `analysis/appendix_lobbench_and_refutations.md` §A.

| | spread | inter-arrival | book imb. | flow imb. | limit depth | cancel depth |
|---|---|---|---|---|---|---|
| **mean over 20 days** | 0.719 | 0.625 | 0.432 | 0.673 | 0.200 | 0.161 |

| | value |
|---|---|
| **grand mean, all six metrics** | **0.468** |
| **grand mean, excluding inter-arrival** | **0.437** |

The five-metric figure is the one to quote against TRADES's released output (0.798 on 2015-01-30),
because their released CSVs carry only 0.1 s timestamps and inflate inter-arrival. **Quoting 0.468
against 0.798 compares six metrics against five** — use 0.437.

Best days: 2015-01-16 (spread 0.591), 2015-01-27 (0.608). Worst: 2015-01-21 (spread 0.954),
2015-01-15 (cancel depth 0.387). Full per-day table in the appendix.

## 2. Cross-day stability — 20/20 days, zero timeouts

Source: `analysis/appendix_checkpoint_evidence.md` §B. 40-minute per-day cap; a timeout counts as
unstable.

| | value |
|---|---|
| days cleared | **20 / 20** |
| timeouts | **0** |
| `uniq_mid` | median 14.0, mean 15.10 |
| `ret1s_std` | **inside the real 1.5–2.5 bp band on every day** |
| wall-clock per 30-min session | 519 s (0106) → 1815 s (0127), mean ~1250 s |

Hardest day (2015-01-07, real ≈27 mids / 13 tk): 15 mids, 2.01 bp, 7 tk — cleared in 1758 s.

This is the result that made 0.724 the baseline, and it is the criterion any replacement has to
match.

## 3. Long-horizon — two hours, INTC 2015-01-29, 10:00–12:00

Source: `analysis/plots/longhorizon/summary.md`, `lobbench_tables.md`.

| series | start | end | min | max | range | ret1s_std | uniq mids |
|---|---|---|---|---|---|---|---|
| Real market | 33.97 | 33.99 | 33.47 | 34.02 | 56 tk | 1.26 bp | 92 |
| **0.724 baseline** | 33.74 | **33.91** | 33.70 | 34.12 | **41 tk** | **1.96 bp** | **62** |
| *(SS epoch 4, for reference)* | 33.74 | 34.09 | 33.70 | 34.12 | 41 tk | 1.65 bp | 64 |
| TRADES single-step | 33.74 | **31.86** | 31.70 | 34.12 | 242 tk | 4.73 bp | 307 |

**It survives two hours.** Single-step crosses the conditioning boundary at ~minute 73 and never
recovers; 0.724 stays inside the real envelope for the full session.

### LOB-Bench at the two-hour horizon

| metric | 0.724 | *SS epoch 4* |
|---|---|---|
| spread | 0.575 | *0.077* |
| inter-arrival | 0.371 | *0.247* |
| book imbalance | **0.371** | *0.377* |
| flow imbalance | 0.522 | *0.520* |
| limit depth | **0.158** | *0.197* |
| cancel depth | 0.135 | *0.123* |
| **grand mean** | **0.355** | *0.257* |

0.724 is **better than SS on book imbalance and limit depth**; the SS advantage is concentrated in
spread. Worth stating if the two are compared.

### Variance ratio

Source: `analysis/plots/longhorizon/variance_ratio_analysis.md`.

| | 1s vol | range | VR(10s) | VR(60s) | VR(300s) |
|---|---|---|---|---|---|
| Real | 1.18 | 56 tk | 0.897 | 0.931 | 1.019 |
| **0.724** | 1.77 | 14 tk | 0.300 | **0.098** | 0.027 |

⚠️ **Range is reported as 41 tk in §3 and 14 tk here** — different warm-up conventions on the same
run. Recompute both under one convention before either goes in print; see the note in
`eval_methodology_handoff.md` about the 1s-bar vs raw-event distinction.

## 4. Seed robustness — INTC 2015-01-30, 30-minute

| series | 1s vol | range | uniq mids | VR(60s) |
|---|---|---|---|---|
| Real | 1.43 | 52 tk | 90 | 1.046 |
| 0.724, seed 30 | 1.72 | 18 tk | 33 | 0.521 |
| 0.724, seed 31 | 1.69 | 21 tk | 39 | 0.517 |
| 0.724, seed 32 | *(completed 2026-08-08, on the remote)* | | | |

Volatility within 0.03 bp and VR within 0.004 across the two scored seeds — the behaviour is
systematic, not seed noise.

## 5. Step-count ablation — 10 steps beat 100

INTC 2015-01-30, checkpoint 0.724, identical decode configuration, only step count differs:

| | grand-mean Wasserstein |
|---|---|
| DDPM, 100 steps | 0.575 |
| **DDIM, 10 steps** | **0.447** |

⚠️ This is an ablation **on our own model**, not a TRADES baseline — the checkpoint carries
`PRICE_REANCHOR` and `UNCLAMP_DEPTH`. The comparison against TRADES is separate, in
`analysis/replication_baselines/`.

## 6. Computational cost

From `timing_summary.txt`, two-hour run on 2015-01-29:

| | value |
|---|---|
| orders generated | 205,503 |
| per order | **11.61 ms** (augmenter 0.46 + network 11.15) |
| throughput | 86.2 orders/s |
| network wall-clock | 2385.1 s |

Against DDPM-100 on the same machine at 141.56 ms/order — a **12.2× per-order** advantage.
⚠️ Timings are **not comparable across machines**: the same 100-step sampler measures 251 ms/order
on the older box.

## 7. The vanilla control — no decode corrections

Vanilla DDIM-10 (no depth-noise, no size-reshape, no type-prior), 30-min, INTC 2015-01-29:

| checkpoint | uniq mids | ret1s_std |
|---|---|---|
| 0.627 (historical) | **3** | 0.15 bp |
| **0.724** | **33** | 1.07 bp |
| *real reference* | 39 | 1.52 bp |

**0.724 does not freeze without the decode corrections.** On this checkpoint the fix moves
volatility 1.07 → 2.05 bp against a 1.52 target — i.e. it *overshoots* — and the vanilla run is
closer to real on execution share. This is the checkpoint-dependence finding, and it applies to the
baseline as much as to the SS lineage.

---

## What is local, and what needs pulling

**Local, with CSVs** — `ABIDES/log/paper_runs_downloaded/`:

- 30-minute DDIM-10 + fixes for the full month (all 20 days)
- 2-hour run, 2015-01-29 (`..._2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3`)
- DDPM-100 on 2015-01-29 and 2015-01-30
- **vanilla** DDIM-10 and DDIM-1 on 2015-01-29
- seed 31 on 2015-01-30

**On the remote, not yet pulled** — both completed 2026-08-08:

```
world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3   # 2h, 2nd day
world_agent_INTC_2015-01-30_10-00-00_32_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3   # seed 32
market_replay_INTC_2015-01-30_12-00-00_30                                            # its real ref
```

```bash
cd /cs/student/project_msc/2025/cf/mmccrea/HSBC/HSBC/ABIDES/log
tar czf ~/pre_ss_gap.tgz \
  world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3 \
  world_agent_INTC_2015-01-30_10-00-00_32_DDIM_0.0_10_val_ema=0.724_tdprior_sr_dn0.3 \
  market_replay_INTC_2015-01-30_12-00-00_30
```

Those two close the only gaps: a **second day** for the long-horizon claim, and the **third seed**
for the robustness table.

## Open items

1. **The 41 tk / 14 tk range conflict** in §3 — same run, two conventions. Resolve before print.
2. **Seed 32** is measured but unpulled, so the seed table reads as n=2 rather than n=3.
3. **The 2-hour result rests on one day** until the 0130 pair is scored.
