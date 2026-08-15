# TRADES paper — reported numbers (for citation)

Berti, Prenkaj & Velardi (2025), *TRADES: Generating Realistic Market Simulations with Diffusion
Models*, arXiv:2502.07071. Extracted from the paper PDF 2026-08-06. **These are their numbers, to be
cited as such — not reproductions.**

---

## Table 1 — Predictive score (MAE), average over two days

Lower is better. "Market Replay" is the real-data reference, i.e. the target every model is trying
to approach.

| Method | Tesla | Intel |
|---|---|---|
| Market Replay *(reference)* | 0.923 | 0.149 |
| IABS | 1.870 | 1.866 |
| CGAN | 3.453 | 0.699 |
| **TRADES** | **1.213** | **0.307** |

Their claim: TRADES beats the second-best by ×3.27 (Tesla) and ×3.48 (Intel), computed as the ratio
of second-best to TRADES after subtracting the market-replay score from each.

---

## Table 2 — DDPM-100 vs DDIM-1 *(the two reference points)*

This is the table that defines the gap the dissertation closes.

| Method | Tesla | Intel |
|---|---|---|
| DDIM *(single step)* | 3.146 | 0.486 |
| DDPM *(original, 100 steps)* | 1.213 | 0.307 |

**Their characterisation, quoted:** *"the performance degradation is significant but not disastrous
despite a remarkable 100-fold increase in computational efficiency."*

That sentence is worth quoting directly in §5.1. It establishes that (a) TRADES themselves offer
single-step DDIM as a speed/quality trade-off, and (b) they judge the quality cost acceptable. The
dissertation's contribution is then sharply framed: the trade-off need not be accepted at all.

---

## Table 3 — Ablations (A) and sensitivity (S), Tesla, predictive score

| | Method | 29/01 | 30/01 |
|---|---|---|---|
| A | TRADES w/o LOB conditioning | 2.642 | 4.728 |
| A | TRADES w/o augmentation | 1.442 | 4.942 |
| S | LSTM backbone | 8.391 | 6.153 |
| S | TRADES w/ CA | 11.90 | 4.891 |

---

## Other reported figures

- **PCA distribution coverage** (TSLA 29/01): TRADES **67.04%** of the real distribution, vs IABS
  52.92% and CGAN 57.49%.
- **Order volume**: ~50,000 orders produced per two-hour simulation (their stated average).

## Their experimental setup

| | |
|---|---|
| Training | 70,000 steps to convergence, then layers frozen |
| Data split | first 17 days train, 18th validation, last 2 days for simulation |
| Warm-up | first **15 minutes of real orders**, then the model takes full control autoregressively |
| Stated window | **10:00 → 12:00** (see discrepancy below) |
| Hardware | RTX 3090 and a portion of an A100 |
| Baselines | Market Replay, IABS (Reference Market Simulation Config: 5000 noise, 100 value, 25 momentum agents, 1 market maker), CGAN (reimplemented by them from scratch) |

---

## ⚠ Two discrepancies worth knowing

**1. Their stated window does not match their released data or their own figures.**

- Paper text: *"We begin the simulation at 10:00 and terminate it at 12:00."*
- Their **Figure 5** (responsiveness) x-axis runs **09:30 → 11:00**.
- Their **released `TRADES-LOB` CSVs** reconstruct to **09:45 → 11:00** (75 minutes).

The released-CSV anchor is well-evidenced: their generated series opens at mid 33.92, and real INTC
on 2015-01-30 at 09:45 is 33.930 — a near-exact match, as expected given a real-order warm-up. Two
independent signals (the literal `45` minute value in the timestamps, and the price match) agree.
So the released data covers 09:45–11:00, consistent with their Figure 5 and not with their text.

Practical consequence: quote **their Table 1/2/3 numbers** as published, but do not assume their
released CSVs correspond to the 10:00–12:00 protocol described in the text.

**2. Their evaluation horizon ends before our single-step failure appears.**

Their released/plotted window ends at **11:00**. Our single-step replication runs 10:00–12:00 and
collapses at **~11:13** — roughly 13 minutes past the end of their evaluation window.

State this carefully. It is a factual observation about horizons, **not** a claim that their
checkpoint would collapse: our run uses our own checkpoint (0.763) and our own training. What it
does support is the methodological point that single-step behaviour over longer horizons is
under-evaluated in the original work, and that a 75-minute window would not have surfaced it.

---

## Comparing our predictive score to theirs

Not currently possible on a like-for-like basis. Our single predictive-score run gave **0.4416**
against a real-on-real baseline of **0.4996**; their Intel figures are 0.307 (DDPM) and 0.486 (DDIM)
against a 0.149 market-replay reference. The absolute scales differ because the protocol differs
(day selection, window, LSTM configuration, averaging over two days). To place our models in their
Table 2 the predictive score would need re-running under their stated protocol — worth doing, since
it is their headline metric and would let the dissertation report a directly comparable number.
