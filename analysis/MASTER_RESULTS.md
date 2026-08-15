# Master results reference

*Consolidated 2026-08-10. Every number held on the project, in one place, on one convention.
Supersedes the scattered per-topic documents for reference purposes; those remain as sources.*

**Everything here is local.** No run referenced below requires pulling from the remote.

---

# Part 1 — Setup and conventions

## 1.1 Data and split

INTC (Intel), LOBSTER level-10, January 2015, 20 trading days.

`SPLIT_RATES = (.85, .05, .10)` (`constants.py:271`), applied **by trading day** in
`preprocessing/LOBSTERDataBuilder.py:225-234`, files enumerated with `sorted(os.listdir(path))` so
the split is chronological and sequential:

| split | days | dates |
|---|---|---|
| train | 17 | 2015-01-02 → 2015-01-27 |
| validation | 1 | 2015-01-28 |
| **test** | **2** | **2015-01-29, 2015-01-30** |

The evaluation days are **held out**, and coincide with TRADES's own declared test set.

⚠️ The 20-day cross-day stability sweep runs on *every* day, training days included. Fidelity
results on 0129/0130 are out-of-sample; the stability result is partly in-sample. Defensible —
closed-loop divergence is not the quantity the model was fitted to — but state it.

## 1.2 Invariant experimental conditions

| | value |
|---|---|
| session start | 09:30:00 |
| warm-up | **09:45** — first 15 min are replayed real orders, excluded from every statistic |
| seed | 30 (31, 32 in robustness cells only) |
| simulator | ABIDES, real matching engine, single world agent |
| default sampler | DDIM, 10 steps, η = 0 |

## 1.3 The three experimental phases

| | Phase 1 — Diagnosis (5–11 Jul) | Phase 2 — Stability (late Jul–Aug) | Phase 3 — Long horizon (6–8 Aug) |
|---|---|---|---|
| days | 2015-01-30 only | all 20 | 0129 and 0130 |
| window | 09:30–10:00 | 09:30–10:00 | 2 h / 60 min / 75 min |
| checkpoints | 0.627, 0.681, 0.719 (pre-fix) | 0.724 + SS epochs | 0.724, SS e4 |
| decode config | vanilla, or one fix at a time | **frozen** at the adopted set | adopted set + lever arms |
| what varies | the sampler | the day and checkpoint | horizon and seed |

Phase 1 deliberately uses one day and a checkpoint chosen *because it fails* — a model that does not
freeze cannot be used to diagnose a freeze. Phases 2–3 invert this: configuration becomes the
control, the day becomes the variable. Say so; it pre-empts the obvious objection.

## 1.4 Run directory naming

```
world_agent_{TICKER}_{DATE}_{ET}_{SEED}_{SAMPLER}_{ETA}_{NSTEPS}_{ckpt[:13]}{flags}
```

| suffix | flag |
|---|---|
| `_tdprior` | `--type-decode prior` |
| `_sr` | `--size-reshape` |
| `_dn0.3` | `--depth-noise 0.3` |
| `_bt2.0r0.5` | `--book-target-thick 2.0 --book-cancel-rate 0.5` |
| `_dd0.25p0.9998` | `--depth-drift 0.25 --depth-drift-phi 0.9998` |
| `_te0.045` | `--dn-target-exec 0.045` |
| `_cc5.0` | `--cond-clip 5.0` |

**No suffix = vanilla run** (no decode corrections).

Two traps: **only the END time is encoded** (`_11-00-00_` is both 10:00–11:00 and 09:45–11:00), and
**the checkpoint is truncated to 13 characters** so `0.69_epoch=2` and `0.69_epoch=4` are
indistinguishable by path.

## 1.5 Evaluation metrics

| metric | what it measures | computed from |
|---|---|---|
| **LOB-Bench** (6 Wasserstein) | distributional fidelity | CSV + real replay |
| **Price-path**: `uniq_mid`, `ret1s_std`, `mid_range_tk` | did the market move | CSV |
| **Flow composition** limit/cancel/executed % | flow balance | CSV (TYPE) |
| **Decode diagnostics** (`DIAG` lines) | variance contraction | **run stdout only** |
| **Variance ratio** VR(q) | persistence | CSV |
| Predictive score (MAE) | TRADES's headline metric | **cited, not reproduced** |

**LOB-Bench conventions that change the number:**
- Inter-arrival is excluded when scoring against TRADES's released output — their CSVs carry only
  0.1 s timestamps, which inflates that metric. A file artefact, not model behaviour.
- **A five-metric mean and a six-metric mean are not comparable.** Both are given throughout below.

**Variance ratio caveat.** VR(q) = Var(q-period return) / (q · Var(1-period return)). ≈1 random
walk, <1 mean-reverting, >1 trending. ⚠️ **A sparse series is biased toward VR ≈ 1**: if the price
jumps rarely and holds flat between, both numerator and denominator scale with the jump rate and
cancel, giving VR ≈ 1 regardless of underlying dynamics. **Always report the non-zero-return count
(`nz%`) beside a variance ratio.** VR on a series with nz% below ~20 carries no information.

**Decode diagnostics are stdout-only.** A `processed_orders.csv` records what was *placed*; the
collapse evidence is what the model *decoded before drops*. The original logs are gone; those
numbers survive only where pasted into markdown at the time. Any new diagnostic run must capture
stdout deliberately.

## 1.6 Recomputation convention used throughout Parts 2–3

**1-second bars, 15-minute warm-up discarded.** Where a figure differs from an older document, it is
because that document used raw events or no warm-up. Both are correct; only one should appear in
print.

---

# Part 2 — The two candidate models

## 2.1 Configuration

Both share: DDIM 10 steps η = 0; `--depth-noise 0.3 --size-reshape --type-decode prior`;
`UNCLAMP_DEPTH` + `PRICE_REANCHOR` (**file-gated — set by the presence of a file, recorded nowhere
in the command line or output path, and unrecoverable from a finished run**).

Architecture is TRADES unchanged: sequence length **256**, augmenter MLP dim **64**, CDT depth **8**
(8 blocks alternating self- and cross-attention), **1 attention head** (`run.py:169` sets
`aug_dim // 64` = 1), MLP ratio 4, concatenation conditioning, `T` = 100, cosine schedule with
offset **s = 0.008**.

Training: lr **2.5e-4**, batch **256**, conditional dropout **0.1**, 50 epochs configured, stopped
at 5.

`val_ema` = validation loss under **exponential-moving-average weights (decay 0.999)**, averaged
over validation batches; the hybrid diffusion objective `L_simple + L_vlb`. It is the
checkpoint-selection metric and **not** a fidelity measure.

**SS epoch 4 specifics:** scheduled sampling with `SS_P_MAX = 0.5`, `SS_RAMP_FRAC = 0.0` (one
teacher-forced epoch then full strength, no ramp), rollouts generated with the **same DDIM-10 η=0
sampler used at simulation time** — corrected in `bb87b79` (2026-07-31), before which rollouts used
the 100-step schedule. The lineage **resumes** from 0.724 via Lightning `ckpt_path` (weights,
optimizer state and epoch counter), not a fresh fine-tune.

## 2.2 Full-month LOB-Bench — 20 days

| | spread | inter-arr. | book imb. | flow imb. | limit depth | cancel depth |
|---|---|---|---|---|---|---|
| **0.724 baseline** | 0.719 | 0.625 | 0.432 | 0.673 | 0.200 | 0.161 |
| **SS epoch 4** | **0.192** | **0.504** | **0.394** | **0.645** | **0.185** | **0.152** |

| | 0.724 | SS epoch 4 | gain |
|---|---|---|---|
| **grand mean, 6 metrics** | 0.4676 | **0.3455** | 26% |
| **grand mean, 5 (excl. inter-arrival)** | 0.4370 | **0.3138** | 28% |

Use **0.437 / 0.314** against TRADES's released 0.798 — the released comparison is five-metric.

**Decompose in the same breath: 72% of the gain is bid-ask spread alone** (0.719 → 0.192,
contributing 0.088 of the 0.122 improvement). Excluding spread, the other five move **0.418 →
0.376** — 10%, not 26%.

SS epoch 4 best day 2015-01-29 (0.275), worst 2015-01-15 (0.444).

## 2.3 Cross-day stability

Both cleared **20/20 days with zero timeouts** under a 40-minute per-day cap. SS epochs 2 and 3 also
cleared all 20; epoch 5 timed out on 2015-01-23 and was already slower on days it completed, which
is where the lineage stopped.

### 0.724 baseline — 30-minute sessions

| day | mids | range | 1s vol | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|
| 02 | 13 | 6 tk | 1.97 | 0.059 | 51.8/42.2/6.0 |
| 05 | 19 | 9 tk | 1.88 | 0.072 | 51.8/40.3/7.9 |
| 06 | 18 | 9 tk | 1.28 | 0.084 | 53.7/36.5/9.8 |
| 07 | 14 | 7 tk | 2.02 | 0.058 | 51.9/40.3/7.8 |
| 08 | 19 | 9 tk | 2.01 | 0.093 | 51.6/40.8/7.5 |
| 09 | 17 | 8 tk | 1.88 | 0.110 | 51.9/41.5/6.6 |
| 12 | 13 | 6 tk | 1.85 | 0.022 | 52.1/40.4/7.5 |
| 13 | 11 | 5 tk | 1.38 | 0.113 | 52.8/40.2/6.9 |
| 14 | 14 | 7 tk | 1.30 | 0.234 | 52.5/40.4/7.1 |
| 15 | 10 | 5 tk | 1.20 | 0.100 | 53.8/39.1/7.1 |
| 16 | 13 | 6 tk | 1.56 | 0.172 | 52.9/39.2/7.9 |
| 20 | 11 | 5 tk | 1.34 | 0.101 | 52.8/39.7/7.5 |
| 21 | 10 | 5 tk | 1.52 | 0.122 | 52.7/40.7/6.6 |
| 22 | 10 | 5 tk | 1.59 | 0.032 | 52.6/40.3/7.1 |
| 23 | 23 | 11 tk | 1.84 | 0.072 | 51.9/40.4/7.7 |
| 26 | 14 | 6 tk | 1.71 | 0.142 | 52.3/40.1/7.6 |
| 27 | 12 | 6 tk | 1.98 | 0.062 | 52.0/40.4/7.7 |
| 28 | 11 | 5 tk | 1.76 | 0.082 | 52.3/40.4/7.3 |
| 29 | 25 | 12 tk | 2.05 | 0.100 | 51.9/40.5/7.6 |
| 30 | 12 | 5 tk | 1.71 | 0.109 | 52.3/40.1/7.5 |

### SS epoch 4 — 30-minute sessions

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

**Counterweight:** SS epoch 4 over-executes more (8.7–13.0% against the baseline's ~7.5%, real ~7%)
and is *less* active at 30 minutes.

## 2.4 Long horizon — two hours, both days

| run | mids | range | 1s vol | nz% | VR(10s) | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|---|---|
| SS e4, 0129 | 57 | 28 tk | 1.35 | 31% | 0.205 | 0.047 | 52.3/33.9/13.9 |
| SS e4, 0130 | 27 | 13 tk | 1.32 | 34% | 0.234 | 0.054 | 53.0/34.6/12.4 |
| 0.724, 0129 | 29 | 14 tk | 1.77 | 49% | 0.300 | 0.098 | 51.6/41.0/7.4 |
| 0.724, 0130 | 21 | 10 tk | 1.40 | 39% | 0.309 | 0.106 | 52.1/40.4/7.5 |
| *real, 0129* | *95* | *55 tk* | *1.33* | *18%* | *0.986* | *0.958* | *49.9/44.9/5.2* |
| *real, 0130* | *104* | *61 tk* | *1.11* | *14%* | *0.767* | *0.630* | *49.9/45.8/4.3* |
| **TRADES single-step, 0129** | **307** | **242 tk** | **4.73** | — | — | — | — |

**Both candidates survive two hours on both days.** The single-step replication crosses the
conditioning boundary at ~minute 73 and never recovers, closing at \$31.86 against real's \$33.99.

### LOB-Bench at two hours (2015-01-29)

| metric | 0.724 | SS epoch 4 |
|---|---|---|
| spread | 0.575 | **0.077** |
| inter-arrival | 0.371 | **0.247** |
| book imbalance | **0.371** | 0.377 |
| flow imbalance | 0.522 | **0.520** |
| limit depth | **0.158** | 0.197 |
| cancel depth | 0.135 | **0.123** |
| **grand mean** | 0.3553 | **0.2566** |

At 30 minutes the two are near-indistinguishable; over two hours SS pulls clearly ahead. **The
evaluation horizon determines which model you would select** — the methodological point.

### The book-balancing lever at two hours (SS e4, 0129)

| | mids | range | 1s vol | VR(60s) | flow L/C/E |
|---|---|---|---|---|---|
| no lever | 57 | 28 tk | 1.35 | 0.047 | 52.3/33.9/13.9 |
| **+ `bt2.0 r0.5`** | **64** | **32 tk** | 1.77 | 0.112 | 51.6/37.1/11.3 |

Raises activity, range and cancel share (toward real's ~44%) at the cost of volatility accuracy.
Not in the adopted configuration; no cross-day evidence.

## 2.5 Seed robustness — 2015-01-30, 30-minute, n = 3

| seed | 0.724 mids / vol / VR60 | SS e4 mids / vol / VR60 |
|---|---|---|
| 30 | 12 / 1.71 / 0.109 | 10 / 1.49 / 0.083 |
| 31 | 17 / 1.65 / 0.095 | 11 / 1.55 / 0.057 |
| 32 | 16 / 1.78 / 0.096 | 11 / 1.67 / 0.081 |

Volatility spans 0.13 bp (0.724) and 0.18 bp (SS e4). Systematic, not seed noise.

## 2.6 Vanilla controls — no decode corrections, 2015-01-29, 30-minute

| run | mids | range | 1s vol | nz% | flow L/C/E |
|---|---|---|---|---|---|
| 0.627 (historical) | 3 | 1 tk | 0.15 | — | 54.0/46.0/**0.0** |
| **0.724 vanilla DDIM-10** | 13 | 8 tk | 0.47 | 6% | 52.6/**43.3**/4.1 |
| **SS e4 vanilla DDIM-10** | 7 | 3 tk | 0.55 | 12% | 52.1/**43.0**/5.0 |
| 0.724 vanilla DDIM-1 | 36 | 18 tk | 3.72 | 68% | 45.7/34.9/19.4 |
| SS e4 vanilla DDIM-1 | 36 | 18 tk | 3.41 | 68% | 46.0/34.8/19.2 |
| *real* | *39* | — | *1.52* | — | *49.2/43.8/7.0* |

**Neither current checkpoint freezes without the corrections** — 13 and 7 unique mids against
0.627's 3, and 0.627 executed **nothing at all**. On these models the decode fix is a volatility
calibration, not a rescue.

**Vanilla cancel share is near-exact** (43.3% and 43.0% against real's 43.8%); the corrections push
it to ~35%. Vanilla execution share is also closer to real. The corrections buy volatility and
activity at the cost of flow composition — a trade, not a free win.

## 2.7 Touch sizes — no wall formation

2015-01-30, 30-minute, post-warm-up:

| | bid mean | ask mean |
|---|---|---|
| **0.724 + corrections** | **2,672** | **4,033** |
| *real* | *3,899* | *2,117* |
| frozen DDIM-10 (ckpt 0.681) | **115,158** | **103,431** |

Within a factor of two of real on both sides, against the frozen configuration's ~30×.

## 2.8 Computational cost

Two-hour runs, 2015-01-29. `timing_summary.txt` measures **augmenter + network only** — confirmed at
`gaussian_diffusion.py:81-83`; it excludes the ABIDES matching engine, order placement and I/O.

| | SS epoch 4 | 0.724 | DDPM-100 |
|---|---|---|---|
| per order | 11.52 ms | 11.61 ms | **141.6 ms** |
| throughput | 86.8/s | 86.2/s | 7.1/s |
| orders (2 h) | 228,131 | 205,503 | — |

**Per-order acceleration: 12.2×.** SS and the baseline are indistinguishable (0.8%, noise); SS
generates 11% more events so takes *longer* in wall-clock — **do not claim SS is faster**.

End-to-end, same machine, 2015-01-30, 30-minute: DDIM-10 **938 s** vs DDPM-100 **1,336 s** = 1.42×.
The gap between 12.2× and 1.42× is that DDPM-100 generated a third as many events and ABIDES
overhead (~30 ms/order, independent of step count) dominates. **Quote per-order and state the
event-count difference** — end-to-end compares a busy market against a quiet one.

⚠️ **Timings are not comparable across machines**: the same 100-step sampler measures 251 ms/order
on the older box and 142 ms/order on the RTX 4070.

---

# Part 3 — Replication baselines

## 3.1 Our DDPM-100 against their released output, 2015-01-30

| metric | THEIRS (released) | OURS (ckpt 0.681, vanilla) |
|---|---|---|
| spread | 1.564 | **1.019** |
| log inter-arrival | 1.038 | **0.575** |
| orderbook imbalance | 0.241 | **0.213** |
| orderflow imbalance | **0.304** | 0.454 |
| limit depth | **0.893** | 1.043 |
| cancel depth | **0.990** | 1.141 |
| **mean, excl. inter-arrival (5)** | **0.798** | **0.774** |
| mean, all six | 0.838 | 0.741 |

Both headline figures verified as **five-metric** means (3.991745/5 and 3.870138/5).

**Excluding inter-arrival works against us** — it is our widest winning margin, and dropping it
narrows the gap from 12% to 3%. Saying so makes the exclusion visibly conservative.

The per-metric split shows differences in **both** directions. The honest claim is "same regime,
different error profile", which is a stronger replication claim than "slightly better".

## 3.2 2015-01-29, for the appendix

| | THEIRS DDPM | OURS DDIM-1 |
|---|---|---|
| mean, excl. inter-arrival | 0.855 | 0.673 |
| mean, all six | 0.886 | 0.626 |

⚠️ Not a replication pair — theirs is DDPM-100, ours is single-step, and ours is the run that
diverges \$1.90 from real. This is the **LOB-Bench blind spot**: the collapsing run scores better.

## 3.3 The single-step collapse

INTC 2015-01-29, 10:00–12:00. Numbers depend on convention:

| | raw events | 1-second bars |
|---|---|---|
| no warm-up | 333 mids / 242 tk | **307 / 242 / 4.73 bp** |
| 15-min warm-up | **309 / 229 / 4.93 bp** | 284 / 229 / 4.93 |

Real, same window, 1s bars + warm-up: **95 mids / 55 tk / 1.33 bp**. Collapse occurs at 11:13:41 —
**73 minutes in, 13 minutes after the end of the authors' plotted window**.

## 3.4 A note on their released CSVs

Timestamps are `MM:SS.s` with the hour dropped. Reconstructing by counting minute wraps from a 09:45
anchor gives exactly **09:45–11:00 (75.0 min)**, and their first mid (33.92) matches real at 09:45
(33.930). The 0.1 s resolution is why inter-arrival is excluded from their scores throughout.

---

# Part 4 — Diagnosis evidence

All on ckpt 0.681, INTC 2015-01-30, 09:30–10:00 unless stated. Real reference: 69 unique mids,
33.605–33.975 (37 tk), flow 49.2/43.8/7.0.

## 4.1 The freeze

| | real | DDIM-10 η=0 |
|---|---|---|
| unique mids | 69 | **6** |
| mid range | 37 tk | **2.5 tk** |
| flow L/C/E | 49.2/43.8/7.0 | 59.4/35.8/**4.8** |
| bid_size_1 mean | 3,899 | **115,158** |
| ask_size_1 mean | 2,117 | **103,431** |

Touch sizes ~30× real: liquidity accumulates because nothing consumes it.

## 4.2 Variance contraction

From the same run: `decoded_pre_drop: limit=16214 cancel=10107 market=729` (market **2.7%** of
decodes); `drops: size_range=6527` (**24%** of decodes discarded for invalid size);
`cond_z[price]: min=-3.81 mean=-3.36 max=-3.12` — conditioning pinned ~3.4σ below the training mean
for the entire run, which is what motivates `PRICE_REANCHOR`.

Depth histogram (different run, `cancel_sweep_table.md`): `depth_pre_drop: neg=1694 0=6515
1-2=3941 3-5=1905 6+=1138` with `execution_channels: A_market_order=0 B_crossing_limit=1412`.

## 4.3 Restoring η does not restore DDPM

| | η=0 | η=1 | real |
|---|---|---|---|
| decoded market | 729 (2.7%) | **5,741 (24.1%)** | ~3% |
| executed % | 4.8 | **15.7** | 7.0 |
| unique mids | 6 | 136 | 69 |
| mid range | 33.935–33.960 | 33.935–**35.160** | 33.605–33.975 |

Movement returns via the **type channel flooding to market orders**, not depth-driven price
discovery, and the price drifts **+\$1.22** out of the real envelope.

## 4.4 Noise placement, not noise quantity

Same checkpoint, same day, same NFE — only placement differs
(`analysis/churn_results.md`, closed-loop table):

| sampler | unique mids | neg-depth % |
|---|---|---|
| DDIM-10 η=0 (control) | **6** | 2.2 |
| **HYBRID_DDPM_PP** (DDPM head, ODE tail) | **113** | 35.8 |
| **HYBRID_PP_DDPM** (ODE head, DDPM tail) | **5** | — |
| DDPM-100 | 23 | 23.7 |
| *real* | *69* | *~24* |

A 20× swing from placement alone. Two qualifications: it is **checkpoint-dependent** (the DDPM-head
hybrid gives 12 on ckpt 0.656), and **113 overshoots real's 69** at 35.8% negative depth. The
defensible claim is that **where the noise goes determines which failure mode you get** — ODE head
freezes, DDPM head diverges, neither lands on real.

⚠️ Do not source this from `hypothesis_results.md`: `A_DDIM_20_eta1` there reads **112** unique mids
and is easily mistaken for the 113. It is a 20-step η=1 run that runs away to \$35.74.

## 4.5 The deterministic operating range is empty

Three failure regimes, one conclusion:

- **Too few steps** → freeze (DDIM-1, DDIM-10 η=0)
- **Scalar temperature** (`--depth-temp κ`, multiplies the decoded depth z-score) → **switch, not
  dial**: κ=1 → 2%, 1.5 → 55%, 2 → 54%, 3 → 52% marketable; no value lands in the 3–24% target band
- **Too many steps** → divergence: 100-step η=0 gives ~70% market decodes, conditioning +265σ

Multiplication is **rank-preserving**, so it relocates a collapsed spike as a whole; additive
per-sample noise re-ranks orders and can peel off a controlled fraction. Contrast with the
dose-response of §4.7.

⚠️ **The κ figures above have no located source** — see Part 6.

## 4.6 Validation loss does not select a usable sampler

`0.681` froze under DDIM η=0 while working under DDPM; `0.719` — the **better** validation loss —
exploded under the same sampler (109 mids, 36.3% negative depth, with zero added stochasticity).
Loss moved the wrong way against sampler behaviour. This is what makes the behavioural search of
Part 2.3 principled rather than improvised.

## 4.7 Dose-response — the dial

Checkpoint 0.627, 2015-01-30, only σ varies:

| σ | mids | range | 1s vol | executed % |
|---|---|---|---|---|
| **none** | **3** | 1 tk | 0.15 | **0.0** |
| 0.10 | 7 | 3 tk | 0.37 | 1.9 |
| 0.125 | 7 | 3 tk | 0.50 | 3.6 |
| 0.15 | 10 | 5 tk | 0.64 | 6.3 |
| 0.16 | 13 | 7 tk | 0.70 | 6.9 |
| 0.18 | 9 | 4 tk | 0.78 | 8.6 |
| 0.20 | 9 | 4 tk | 0.94 | 9.6 |
| **0.30** | **15** | 7 tk | **1.42** | 12.2 |
| 0.40 | 23 | 11 tk | 1.93 | 15.0 |
| *real* | *69* | *37 tk* | *~1.5* | *7.0* |

**Volatility and execution share are monotonic across nine settings with no reversals.** This is the
"dial, not switch" figure, and it lands hardest directly beside §4.5's κ cliff. Unique mids and
range are noisier — plot volatility and executed share as the smooth curves.

---

# Part 5 — Numbers reconciliation

## 5.1 The four grand means on 2015-01-30

**All are six-metric.** All 09:30–10:00.

| metric | **a** DDPM-100<br>ckpt 0.627 +tdprior | **b** DDIM-10<br>ckpt 0.724 +corr | **c** DDIM-10<br>ckpt 0.627 +corr | **d** DDPM-100<br>ckpt 0.724 +corr |
|---|---|---|---|---|
| spread | 0.765 | 0.625 | **0.151** | 0.963 |
| log inter-arrival | 0.565 | 0.671 | **0.482** | 0.559 |
| orderbook imbalance | 0.456 | **0.384** | 0.461 | 0.429 |
| orderflow imbalance | **0.446** | 0.664 | 0.455 | 0.643 |
| limit depth | 0.425 | 0.172 | **0.119** | 0.459 |
| cancel depth | 0.388 | 0.164 | **0.121** | 0.397 |
| **mean (6)** | **0.507** | **0.447** | **0.298** | **0.575** |
| mean (5, excl. i-a) | 0.496 | 0.402 | 0.262 | 0.578 |

## 5.2 ⚠️ 0.507 is NOT the published baseline

It is **our** DDPM-100 on checkpoint **0.627**, carrying `--type-decode prior` — one of our own
corrections. Any label calling it "the original 100-step baseline" is wrong.

Three distinct DDPM-100 figures exist on 2015-01-30:

| figure | checkpoint | decode config | window | metrics |
|---|---|---|---|---|
| **0.507** | 0.627 | `--type-decode prior` | 30 min | 6 |
| **0.575** | 0.724 | full corrections | 30 min | 6 |
| **0.774** | 0.681 | **vanilla** | 09:30–10:30 | 5 |

**For the headline claim use 0.447 against 0.575** — the only controlled pair (same checkpoint, same
corrections, same day, same window, same six metrics; only step count differs). **Do not use 0.447
against 0.507**: that compares two different checkpoints. 0.774 belongs only in Part 3, against
their released 0.798.

Also: the single-day-tuned lineage is checkpoint **0.627**, not 0.635.

## 5.3 Conventions that change a number

| | effect |
|---|---|
| raw events vs 1-second bars | changes unique-mid counts (333 → 307), leaves range unchanged |
| warm-up vs none | changes both (242 → 229 tk; 62 mids → 29 on the 2 h run) |
| five- vs six-metric mean | 0.468 vs 0.437 on the same month |
| machine | 251 vs 142 ms/order for the same 100-step sampler |

Every one of these has already produced a conflicting pair somewhere in the drafts.

---

# Part 6 — Open items

| # | item | cost |
|---|---|---|
| 1 | **`--type-decode l1` vs `prior` on ckpt 0.724** — never measured; every 0.724 run is either fully corrected or fully vanilla. Needs one 30-min run **with stdout captured** (the market-decode share is a `DIAG` line). | ~20 min |
| 2 | **The κ depth-temperature figures** (§4.5) — quoted but no source located in any retained file. Re-derive with a three-point sweep, or drop the specific percentages. | ~1 h |
| 3 | **Vanilla DDPM-100 30-minute timing on the current machine** — the 40–60 min figure is derived (July run halved, hardware-adjusted to ~53 min), not measured. | ~1 h |
| 4 | **Full-month vanilla DDPM-100** — would give a like-for-like month against the DDIM-10 month. ~20–25 h; decide whether it is worth a dedicated session. | 20–25 h |
| 5 | **Predictive score (MAE)** — infrastructure exists, never run comparably. Currently cited, not reproduced. | — |
| 6 | **The book-balancing lever** — improves activity and cancel share at 2 h but has no cross-day evidence and is not in the adopted config. | — |

## Sources

`pre_ss_0724_consolidated.md` · `final_model_ss_e4_consolidated.md` ·
`numbers_audit_20260810.md` · `final_model_config.md` · `experimental_setup.md` ·
`eval_methodology_handoff.md` · `replication_baselines/` · `appendix_checkpoint_evidence.md` ·
`appendix_lobbench_and_refutations.md` · `churn_results.md` · `hypothesis_results.md`
