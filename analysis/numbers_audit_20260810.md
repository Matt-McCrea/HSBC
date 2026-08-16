# Numbers audit — 2026-08-10

*Five requests, all resolved from retained score files and local CSVs. No new simulation was needed
except where stated at the end.*

---

## 1. Per-metric LOB-Bench, INTC 2015-01-30 — all four configurations

Wasserstein-1 distance to real, lower is better. **All four quoted grand means are SIX-metric**
(they include `log_inter_arrival`); the five-metric row is given for comparability with the
replication table, which excludes it.

| metric | **a** DDPM-100<br>ckpt 0.627 +tdprior | **b** DDIM-10<br>ckpt 0.724 +corr | **c** DDIM-10<br>ckpt 0.627 +corr | **d** DDPM-100<br>ckpt 0.724 +corr |
|---|---|---|---|---|
| spread | 0.765 | 0.625 | **0.151** | 0.963 |
| log inter-arrival | 0.565 | 0.671 | **0.482** | 0.559 |
| orderbook imbalance | 0.456 | **0.384** | 0.461 | 0.429 |
| orderflow imbalance | **0.446** | 0.664 | 0.455 | 0.643 |
| limit depth (ask) | 0.425 | 0.172 | **0.119** | 0.459 |
| cancel depth (ask) | 0.388 | 0.164 | **0.121** | 0.397 |
| **mean (6 metrics)** | **0.507** | **0.447** | **0.298** | **0.575** |
| mean (5, excl. inter-arrival) | 0.496 | 0.402 | 0.262 | 0.578 |

All four reconcile to the quoted figures exactly.

| | window | checkpoint | decode config | source |
|---|---|---|---|---|
| a | 09:30–10:00 (30 min) | **0.627** | `--type-decode prior` only | `lob_bench_reanchored/` row `DDPM` |
| b | 09:30–10:00 (30 min) | **0.724** | dn0.3 + sr + tdprior | `lob_bench_paper/stepcount_0130/` row `DDIM10` |
| c | 09:30–10:00 (30 min) | **0.627** | dn0.3 + sr + tdprior | `lob_bench_reanchored/` row `dn0.30` |
| d | 09:30–10:00 (30 min) | **0.724** | dn0.3 + sr + tdprior | `lob_bench_paper/stepcount_0130/` row `DDPM100` |

⚠️ **Config (c) is checkpoint 0.627, not 0.635.** The single-day-tuned lineage throughout is `0.627`.

## 1b. The 0.507 conflict — resolved, and it changes the claim

**0.507 is not the published 100-step approach.** It is *our* DDPM-100 on checkpoint **0.627**,
carrying `--type-decode prior` — one of our own decode corrections. The label "the original
100-step baseline (DDPM)" in `appendix_lobbench_and_refutations.md` §C is wrong and should be
changed.

There are **three** distinct DDPM-100 numbers on 2015-01-30 and they measure three different things:

| figure | checkpoint | decode config | window | metrics |
|---|---|---|---|---|
| **0.507** | 0.627 | `--type-decode prior` | 30 min | 6 |
| **0.575** | 0.724 | dn0.3 + sr + tdprior | 30 min | 6 |
| **0.774** | 0.681 | **vanilla** | 09:30–10:30 (60 min) | 5 |

They differ by checkpoint, decode configuration, window *and* metric count, which is why they
disagree by a third.

**For the headline claim, use 0.447 against 0.575.** That pair is the only controlled comparison:
same checkpoint (0.724), same corrections, same day, same window, same six metrics — only step
count differs. It is already in `lob_bench_paper/stepcount_0130/`.

**Do not use 0.447 against 0.507** — that compares checkpoint 0.724 to checkpoint 0.627 and is
confounded. And 0.774 belongs only in the replication section, where it is compared against
TRADES's released 0.798 on the same five metrics.

## 2. Touch sizes — no wall formation on the adopted model

Checkpoint 0.724 + corrections, INTC 2015-01-30, 30-minute, post-warm-up:

| | bid_size_1 mean | ask_size_1 mean | bid max | ask max |
|---|---|---|---|---|
| **0.724 + corrections** | **2,672** | **4,033** | 18,424 | 40,539 |
| *real (09:45–10:00)* | *3,899* | *2,117* | *19,803* | *10,005* |
| frozen DDIM-10, ckpt 0.681 | **115,158** | **103,431** | — | — |

The adopted model sits **within a factor of two of real on both sides** — thinner than real on the
bid, about twice real on the ask — against the frozen configuration's ~30×. That is the
"no wall formation" evidence for the shipped model rather than for 0.627.

*(Real figures from `hypothesis_results.md`; the 30-minute 2015-01-30 replay is not local, so the
real row is quoted rather than recomputed. Windows differ slightly — real is 09:45–10:00.)*

## 3. Type decode on 0.724 — NOT MEASURABLE, needs a run

There is **no `l1` type-decode run on checkpoint 0.724**. Every 0.724 run either carries the full
correction set (`_tdprior_sr_dn0.3`) or is fully vanilla — no run isolates `--type-decode` at
matched σ.

To measure it, one 30-minute simulation is needed:

```bash
python ABIDES/abides.py -c world_agent_sim -t INTC -date 20150130 \
  -st 09:30:00 -et 10:00:00 -d True -m TRADES -type DDIM -nsteps 10 -eta 0.0 \
  --ckpt-path <0.724 ckpt> -seed 30 \
  --depth-noise 0.3 --size-reshape                    # l1 decode: omit --type-decode
```

That produces `..._sr_dn0.3` (no `tdprior`), directly comparable to the existing
`..._tdprior_sr_dn0.3`. **Capture stdout** — the market-decode share is a `DIAG decoded_pre_drop`
line and is not recoverable from the CSV.

## 4. The vanilla variance-ratio test — RESOLVED: artefact, not finding

The question was whether vanilla runs genuinely random-walk (VR ≈ 1), which would mean the decode
corrections *introduce* the mean reversion rather than the architecture carrying it.

INTC 2015-01-29, 30-minute, 1-second bars, 15-minute warm-up. **`nonzero` = number of 1-second bars
with a non-zero return**, out of 877:

| run | bars | non-zero | % | mids | 1s vol | VR(10s) | VR(60s) |
|---|---|---|---|---|---|---|---|
| 0.724 vanilla DDIM-10 | 877 | **55** | **6.3%** | 13 | 0.47 | 0.483 | **0.956** |
| SS e4 vanilla DDIM-10 | 877 | **101** | **11.5%** | 7 | 0.55 | 0.284 | **0.061** |
| 0.724 vanilla DDIM-1 | 877 | 597 | 68.2% | 36 | 3.72 | 0.651 | **0.194** |
| SS e4 vanilla DDIM-1 | 877 | 595 | 67.9% | 36 | 3.41 | 0.598 | **0.187** |
| 0.724 + corrections DDIM-10 | 877 | 514 | 58.7% | 25 | 2.05 | 0.242 | **0.100** |

**The 0.956 is a small-sample artefact.** Three independent reasons:

1. **It rests on 55 non-zero returns.** A 60-second variance ratio over 877 seconds has ~14
   non-overlapping blocks; with 55 jumps in total, most blocks contain three or four. Both variance
   estimates are dominated by a handful of events.
2. **The other vanilla DDIM-10 run, same day and sampler, reads 0.061.** If vanilla genuinely
   preserved persistence, both checkpoints would show it. They differ by a factor of fifteen, and
   the one with *fewer* non-zero returns reads higher — the signature of noise, not of structure.
3. **Where vanilla runs are active enough for VR to mean anything, it is low.** The DDIM-1 vanilla
   runs have 68% non-zero returns — comparable to the corrected run's 59% — and read **0.194 and
   0.187**, the same regime as the corrected run's 0.100.

**Conclusion: the mean reversion is inherited, not introduced.** §5.7 stands as written, and the
`variance_ratio_analysis.md` framing (supported independently by TRADES's own released output at
0.099 / 0.153) does not need revising.

**Report the non-zero-return count alongside any VR figure.** It is the diagnostic that separates a
real variance ratio from an artefact, and it is what settled this question.

## 5. Timing

### What `timing_summary.txt` measures — confirmed from source

`models/diffusers/gaussian_diffusion.py:81-83`:

```python
self._t_aug   = 0.0   # seconds in augment/deaugment
self._t_step  = 0.0   # seconds in NN forward + reconstruction
self._n_calls = 0     # number of sample() calls (= orders generated)
```

Accumulated per `sample()` call and printed once at end of simulation
(`gaussian_diffusion.py:528`). **It is model time only** — augmenter plus network forward. It
excludes the ABIDES matching engine, order placement, book updates and I/O. So the
`0.46 + 11.15 = 11.61 ms` decomposition is exactly augmenter + network, as inferred.

End-to-end session time is the **`Time taken to run simulation`** line in the run's stdout, which is
a different and larger number.

### A1 / A2 — end-to-end, same machine, same day, same window

RTX 4070, INTC 2015-01-30, 09:30–10:00, checkpoint 0.724 + corrections:

| | end-to-end | orders generated | model time | per order |
|---|---|---|---|---|
| **DDIM-10 + corrections** | **938 s** (15 m 38 s) | ~20,000 | ~235 s | 11.7 ms |
| **DDPM-100 + corrections** | **1,336 s** (22 m 16 s) | 7,661 | 1,095 s | 142.9 ms |

**End-to-end ratio: 1.42×. Per-order ratio: 12.2×.**

The two differ because DDPM-100 generates roughly a third as many events, and ABIDES overhead
(~30 ms/order, independent of step count) dominates the DDIM-10 total. **Quote the per-order figure
as the model acceleration and state the event-count difference**, because end-to-end session time
compares a busy market against a quiet one and is not a clean speed measurement.

⚠️ The DDPM-100 row is the degenerate configuration — 7,661 orders is ~4.3 events/second against
real's ~45. A vanilla DDPM-100 producing realistic activity would take substantially longer
(~50–60 min for a 30-minute session on this hardware, derived from the July run).

---

## Summary of what changed

| | status |
|---|---|
| 0.507 mislabelled as the published baseline | **must be corrected** — it is ckpt 0.627 with a decode correction |
| Headline comparison | use **0.447 vs 0.575**, the controlled pair |
| "ckpt 0.635" | it is **0.627** throughout |
| Vanilla VR = 0.956 | **artefact**, resolved — inherited framing stands |
| Touch sizes on adopted model | within 2× of real, no walls |
| Type-decode l1 vs prior on 0.724 | **needs one 30-minute run** |
| Timing | per-order 12.2×, end-to-end 1.42× — quote per-order |
