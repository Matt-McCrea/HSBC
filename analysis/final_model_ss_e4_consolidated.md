# The final model — scheduled-sampling epoch 4 (`val_ema=0.69_epoch=4`)

*2026-08-10. Everything held on the adopted checkpoint, computed from local CSVs. Companion to
`pre_ss_0724_consolidated.md`, which covers the pre-scheduled-sampling baseline.*

**Configuration:** DDIM, 10 steps, η = 0, `--depth-noise 0.3 --size-reshape --type-decode prior`,
`UNCLAMP_DEPTH` + `PRICE_REANCHOR`, seed 30 unless stated. Full provenance in
`analysis/final_model_config.md`.

**Convention for every recomputed table below:** 1-second bars, 15-minute warm-up discarded.
`nz%` = percentage of 1-second bars carrying a non-zero return — report it alongside any variance
ratio, since it is what separates a real VR from a small-sample artefact.

---

## 1. Full-month LOB-Bench — the headline

All 20 trading days, Wasserstein-1 to real. Source: `lob_bench_ss_retrain/epoch4_0.69/`.

| | spread | inter-arrival | book imb. | flow imb. | limit depth | cancel depth |
|---|---|---|---|---|---|---|
| **mean over 20 days** | **0.192** | 0.504 | 0.394 | 0.645 | 0.185 | 0.152 |

| | SS epoch 4 | 0.724 baseline | improvement |
|---|---|---|---|
| **grand mean, 6 metrics** | **0.3455** | 0.4676 | **26%** |
| **grand mean, 5 (excl. inter-arrival)** | **0.3138** | 0.4370 | **28%** |

Best day 2015-01-29 (0.275); worst 2015-01-15 (0.444). Per-day table in the appendix below.

### ⚠️ Decompose this in the same breath

**72% of the total gain is bid-ask spread alone** — 0.719 → 0.192, which contributes 0.088 of the
0.122 grand-mean improvement. Excluding spread, the remaining five metrics move **0.418 → 0.376**,
a 10% improvement rather than 26%.

Both statements are true and the honest framing states them together. Volunteering the
decomposition is more persuasive than having a reader derive it.

## 2. Cross-day stability and behaviour — 20/20 days

Every day cleared with zero timeouts (`appendix_checkpoint_evidence.md` §C, epoch 4).
Recomputed price-path statistics, 30-minute sessions:

| day | mids | range | 1s vol | nz% | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|---|
| 02 | 17 | 8 tk | 1.76 | 47% | 0.074 | 52.2/37.9/9.9 |
| 05 | 17 | 8 tk | 1.54 | 42% | 0.060 | 52.2/35.6/12.2 |
| 06 | 13 | 6 tk | 1.00 | 29% | 0.062 | 56.2/32.8/11.0 |
| 07 | 15 | 7 tk | 1.46 | 36% | 0.117 | 52.6/34.9/12.5 |
| 08 | 21 | 10 tk | 1.68 | 40% | 0.039 | 52.2/35.3/12.5 |
| 09 | 13 | 6 tk | 1.71 | 46% | 0.057 | 52.4/36.7/11.0 |
| 12 | 13 | 6 tk | 1.46 | 37% | 0.065 | 52.8/35.3/12.0 |
| 13 | 9 | 4 tk | 1.34 | 38% | 0.012 | 54.0/36.9/9.1 |
| 14 | 11 | 5 tk | 1.25 | 37% | 0.063 | 53.9/36.8/9.3 |
| 15 | 7 | 3 tk | 1.14 | 26% | 0.043 | 55.7/34.5/9.8 |
| 16 | 9 | 4 tk | 1.33 | 34% | 0.059 | 53.8/35.2/11.0 |
| 20 | 9 | 4 tk | 1.23 | 33% | 0.053 | 54.0/35.8/10.3 |
| 21 | 11 | 5 tk | 1.37 | 38% | 0.083 | 53.9/37.3/8.7 |
| 22 | 9 | 4 tk | 1.36 | 35% | 0.041 | 53.4/35.5/11.1 |
| 23 | 15 | 7 tk | 1.55 | 41% | 0.068 | 52.5/35.1/12.4 |
| 26 | 11 | 5 tk | 1.49 | 37% | 0.078 | 53.2/35.2/11.5 |
| 27 | 12 | 5 tk | 1.56 | 39% | 0.042 | 52.6/34.3/13.0 |
| 28 | 11 | 5 tk | 1.49 | 36% | 0.055 | 53.0/35.9/11.1 |
| 29 | 13 | 6 tk | 1.49 | 34% | 0.104 | 52.4/35.1/12.6 |
| 30 | 10 | 5 tk | 1.49 | 37% | 0.083 | 53.7/36.2/10.1 |

Volatility 1.00–1.76 bp, tightly clustered; VR(60 s) 0.012–0.117 on every day.

**The counterweight to state:** execution share is **8.7–13.0%** against real's ~7%, and against the
0.724 baseline's ~7.5% on the same days. **SS epoch 4 over-executes more than the model it
replaces.** It is also *less* active at 30 minutes (mids 7–21 against the baseline's 10–25).

## 3. Long horizon — two hours, INTC 2015-01-29

| config | mids | range | 1s vol | nz% | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|---|
| **+ corrections** | 57 | 28 tk | 1.35 | 31% | **0.047** | 52.3/33.9/13.9 |
| + corrections + book lever | **64** | **32 tk** | 1.77 | 42% | **0.112** | 51.6/37.1/11.3 |
| *real* | *~95* | *~55 tk* | *~1.33* | — | *~0.93* | *~49/44/7* |

**The book-balancing lever improves persistence 2.4×** (0.047 → 0.112) and raises both range and
activity — the opposite of the predicted direction, since it is a restoring force by construction.
Its cost is volatility accuracy (1.35 → 1.77 against real's ~1.33) and it raises cancel share toward
real (33.9 → 37.1, real ~44).

### LOB-Bench at two hours (2015-01-29)

| metric | SS epoch 4 | 0.724 |
|---|---|---|
| spread | **0.077** | 0.575 |
| inter-arrival | **0.247** | 0.371 |
| book imbalance | 0.377 | **0.371** |
| flow imbalance | **0.520** | 0.522 |
| limit depth | 0.197 | **0.158** |
| cancel depth | **0.123** | 0.135 |
| **grand mean** | **0.2566** | 0.3553 |

At 30 minutes the two are near-indistinguishable; over two hours SS pulls clearly ahead. **The
evaluation horizon determines which model you would select** — the methodological point worth making
from this table.

## 4. Seed robustness — INTC 2015-01-30, 30-minute

| seed | mids | range | 1s vol | nz% | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|---|
| 30 | 10 | 5 tk | 1.49 | 37% | 0.083 | 53.7/36.2/10.1 |
| 31 | 11 | 5 tk | 1.55 | 39% | 0.057 | 53.6/36.7/9.7 |
| 32 | 11 | 5 tk | 1.67 | 44% | 0.081 | 53.1/37.6/9.2 |

Volatility spans 0.18 bp, range identical, VR spans 0.026. n = 3, systematic not seed noise.

## 5. The vanilla control — INTC 2015-01-29, 30-minute

| config | mids | range | 1s vol | nz% | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|---|
| vanilla DDIM-10 | 7 | 3 tk | 0.55 | **12%** | 0.061 | 52.1/**43.0**/5.0 |
| vanilla DDIM-1 | 36 | 18 tk | 3.41 | 68% | 0.187 | 46.0/34.8/**19.2** |
| + corrections DDIM-10 | 13 | 6 tk | 1.49 | 34% | 0.104 | 52.4/35.1/12.6 |
| *real* | *39* | — | *1.52* | — | — | *49.2/43.8/7.0* |

Two things worth noting, both of which cut against the corrections:

- **Vanilla cancel share is 43.0% against real's 43.8%** — nearly exact. The corrections push it
  down to 35.1%. The cancel-share gap is *created* by the decode corrections on this checkpoint.
- **Vanilla execution share is 5.0% against real's 7.0%**; corrected is 12.6%. The vanilla run is
  closer to real on execution too.

The corrections buy volatility (0.55 → 1.49 bp against a 1.52 target) and activity (7 → 13 mids) at
the cost of flow composition. Worth stating as a trade rather than a free win.

⚠️ The vanilla VR(60 s) of 0.061 rests on only **12% non-zero bars** — do not read it as evidence
either way. See `numbers_audit_20260810.md` §4 for why sparse series make VR unreliable.

## 6. Computational cost

Two-hour run, 2015-01-29, from `timing_summary.txt` (**augmenter + network only** — excludes the
ABIDES matching engine):

| | SS epoch 4 | 0.724 |
|---|---|---|
| orders generated | **228,131** | 205,503 |
| per order | 11.52 ms | 11.61 ms |
| throughput | 86.8 orders/s | 86.2 orders/s |
| network wall-clock | 2628.4 s | 2385.1 s |

**Per-order cost is indistinguishable (0.8%, within noise).** SS generates 11% more events over the
same simulated window and therefore takes correspondingly longer in wall-clock. Since real is more
active than either, read the extra activity as fidelity bought with runtime — **do not claim SS is
faster.**

---

## Long horizon on BOTH days — added 2026-08-10

*1-second bars, 15-minute warm-up. The 2015-01-30 run was pulled and scored after the tables above.*

| run | mids | range | 1s vol | nz% | VR(10s) | VR(60s) | VR(300s) | flow L/C/E |
|---|---|---|---|---|---|---|---|---|
| **SS e4, 0129** | 57 | 28 tk | 1.35 | 31% | 0.205 | **0.047** | 0.016 | 52.3/33.9/13.9 |
| **SS e4, 0130** | 27 | 13 tk | 1.32 | 34% | 0.234 | **0.054** | 0.008 | 53.0/34.6/12.4 |
| 0.724, 0129 | 29 | 14 tk | 1.77 | — | 0.300 | **0.098** | 0.027 | 51.6/41.0/7.4 |
| 0.724, 0130 | 21 | 10 tk | 1.40 | 39% | 0.309 | **0.106** | 0.038 | 52.1/40.4/7.5 |
| *real, 0130* | *104* | *61 tk* | *1.11* | *14%* | *0.767* | *0.630* | *0.739* | *49.9/45.8/4.3* |

**SS epoch 4 survives two hours on both days**, and its VR replicates closely (0.047 / 0.054), so the
long-horizon claim no longer rests on a single day.

### ⚠️ The head-to-head is genuinely mixed — read before choosing the final model

On **both** days at the two-hour horizon:

| | SS epoch 4 | 0.724 | closer to real |
|---|---|---|---|
| LOB-Bench grand mean (0129) | **0.257** | 0.355 | **SS** |
| activity — unique mids | **57 / 27** | 29 / 21 | **SS** |
| range | **28 / 13 tk** | 14 / 10 tk | **SS** |
| **variance ratio VR(60s)** | 0.047 / 0.054 | **0.098 / 0.106** | **0.724 — by ~2×** |
| execution share | 13.9 / 12.4% | **7.4 / 7.5%** | **0.724** (real 4.3%) |
| cancel share | 33.9 / 34.6% | **41.0 / 40.4%** | **0.724** (real 45.8%) |

**The baseline has roughly twice the persistence of the adopted model at long horizon, consistently
on both days** — and the variance ratio is the diagnostic built specifically to catch what LOB-Bench
cannot see. The baseline is also markedly closer to real on both flow components.

So the two models differ on *which* axis they are realistic on: SS wins distributional fidelity and
activity; the baseline wins price-path persistence and flow composition. This is not a case of one
model dominating, and the choice depends on the intended downstream use — a point that matters
directly for the reinforcement-learning execution task, where persistence and execution share are
the properties an agent trades against.

---

## What needs pulling

**Nothing — all runs are now local**, including the two-hour 2015-01-30 run pulled on 2026-08-10.
The original gap, for the record:

```
world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3
```

The **two-hour run on the second day**, completed 2026-08-08 in 80 m 56 s. Its 0.724 counterpart and
the matched real replay are already local, so this one file completes the long-horizon comparison
on both days for both checkpoints.

```bash
cd /cs/student/project_msc/2025/cf/mmccrea/HSBC/HSBC/ABIDES/log
tar czf ~/ss4_2h_0130.tgz \
  world_agent_INTC_2015-01-30_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3
```

## Open items

1. ~~The 2-hour SS result rests on 2015-01-29~~ — **RESOLVED**: both days now scored, VR 0.047 and
   0.054. See the head-to-head section above, which surfaces a persistence result favouring the
   baseline.
2. **The book-balancing lever** improves persistence at 2 h but is not in the adopted configuration
   and has no cross-day evidence. Decision deferred.
3. **Type-decode `l1` vs `prior` on this checkpoint** has never been measured — see
   `numbers_audit_20260810.md` §3. One 30-minute run with stdout captured.

---

## Appendix — per-day LOB-Bench, SS epoch 4

| day | spread | inter-arr. | book imb. | flow imb. | limit depth | cancel depth | mean6 |
|---|---|---|---|---|---|---|---|
| 2015-01-02 | 0.188 | 0.378 | 0.368 | 0.595 | 0.231 | 0.211 | 0.328 |
| 2015-01-05 | 0.106 | 0.297 | 0.310 | 0.657 | 0.217 | 0.209 | 0.299 |
| 2015-01-06 | 0.299 | 0.622 | 0.721 | 0.645 | 0.120 | 0.154 | 0.427 |
| 2015-01-07 | 0.117 | 0.346 | 0.399 | 0.799 | 0.250 | 0.180 | 0.348 |
| 2015-01-08 | 0.134 | 0.384 | 0.377 | 0.737 | 0.270 | 0.119 | 0.337 |
| 2015-01-09 | 0.251 | 0.377 | 0.359 | 0.780 | 0.140 | 0.122 | 0.338 |
| 2015-01-12 | 0.106 | 0.352 | 0.295 | 0.784 | 0.175 | 0.151 | 0.311 |
| 2015-01-13 | 0.309 | 0.736 | 0.318 | 0.725 | 0.140 | 0.139 | 0.395 |
| 2015-01-14 | 0.148 | 0.738 | 0.353 | 0.624 | 0.185 | 0.200 | 0.375 |
| 2015-01-15 | 0.256 | 0.816 | 0.416 | 0.592 | 0.253 | 0.333 | 0.444 |
| 2015-01-16 | 0.172 | 0.688 | 0.357 | 0.500 | 0.106 | 0.056 | 0.313 |
| 2015-01-20 | 0.270 | 0.633 | 0.461 | 0.627 | 0.107 | 0.101 | 0.367 |
| 2015-01-21 | 0.520 | 0.713 | 0.411 | 0.547 | 0.220 | 0.187 | 0.433 |
| 2015-01-22 | 0.160 | 0.560 | 0.379 | 0.616 | 0.193 | 0.124 | 0.339 |
| 2015-01-23 | 0.122 | 0.332 | 0.389 | 0.625 | 0.242 | 0.150 | 0.310 |
| 2015-01-26 | 0.070 | 0.428 | 0.473 | 0.574 | 0.198 | 0.136 | 0.313 |
| 2015-01-27 | 0.192 | 0.340 | 0.360 | 0.592 | 0.168 | 0.108 | 0.293 |
| 2015-01-28 | 0.120 | 0.505 | 0.372 | 0.594 | 0.173 | 0.113 | 0.313 |
| 2015-01-29 | 0.062 | 0.277 | 0.383 | 0.667 | 0.161 | 0.101 | 0.275 |
| 2015-01-30 | 0.244 | 0.562 | 0.372 | 0.625 | 0.161 | 0.148 | 0.352 |
| **mean** | **0.192** | **0.504** | **0.394** | **0.645** | **0.185** | **0.152** | **0.3455** |
