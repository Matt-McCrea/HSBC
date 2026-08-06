# §5.1 Replication — numbers ready to write up

*Compiled 2026-08-06. Every figure below is sourced; provenance notes at the end matter, because
two different DDPM-100 runs exist and only one is the documented replication.*

---

## Table 1 — Flow composition, INTC 2015-01-30

Source: `analysis/figures/flow_and_market_data.md`. Compared against TRADES's **actual released
`TRADES-LOB` CSVs**, not merely their published summary numbers.

| Source | Limit % | Cancel % | Executed % |
|---|---|---|---|
| **Real (market replay)** | 49.2 | 43.8 | **7.0** |
| Their `TRADES-LOB`, INTC 2015-01-29 (DDPM) | 46.4 | 36.4 | 17.2 |
| Their `TRADES-LOB`, INTC 2015-01-30 (DDPM) | 46.4 | 36.2 | 17.3 |
| Their `TRADES-LOB`, TSLA 2015-01-29 (DDPM) | 45.9 | 39.4 | 14.7 |
| Their `TRADES-LOB`, TSLA 2015-01-30 (DDPM) | 46.6 | 39.4 | 14.0 |
| **Ours: DDPM-100 (ckpt 0.681)** | **47.2** | **34.0** | **18.8** |

**The point to make**: our DDPM-100 lands inside the band their own released outputs occupy
(14.0–17.3% executed across four stock-days; ours 18.8%). The replication is sound.

**The point NOT to skip**: their released gold-standard *also over-executes against real* — 14–17%
against real's 7.0%. This is a property of the published model, not of our reproduction, and saying
so early is what licenses the whole diagnosis chapter. It reframes §5.2 from "we broke it" to "the
published model already has this signature and acceleration amplifies it".

Caveat worth one sentence: "executed" bundles market orders with marketable-limit executions and
cannot be separated from the released CSV format.

---

## Table 2 — Sampler comparison on a common checkpoint (val_ema=0.681)

Source: `analysis/deterministic_sampler_findings.md` §8.

| Sampler | Market decode | Executions | Unique mids | Mid drift | Verdict |
|---|---|---|---|---|---|
| **Real target** | 2.8% | 7.0% | 69 | \$0.37 | — |
| Their DDPM (released `TRADES-LOB`) | — | 17.3% | 46 | \$0.23 | benchmark: structured, moving |
| **Our DDPM-100** | 7.9% | 18.8% | 23 | \$0.11 | works, matches theirs |
| Our DDIM-10, η=0 | 2.7% | 4.8% | **6** | \$0.03 | frozen (depth collapse) |
| Our DDIM-10, η=1 | 24.1% | 15.7% | 136 | **\$1.23** | moves via market-order drift |
| Our DDIM-100, η=0 | 70.8% | 48.2% | 49 | **−\$11.5** | diverges (stiff ODE) |
| DDIM-1 (repo default) | — | — | (moves) | — | *row previously incomplete — see Table 3* |

"Market decode" is an internal diagnostic (the share decoded as market type pre-drop); it is **not
measurable in any released CSV**, so no value exists for their model. Real reference ≈ 2.8%
(open-loop next-event market fraction).

---

## Table 3 — Single-step DDIM, now quantified

The DDIM-1 row in Table 2 was previously blank. These are new numbers (2026-08-05/06) and fill it.

**Conditions**: ckpt val_ema=0.763, INTC 2015-01-29, 10:00–12:00, vanilla (no decode-time
interventions), against a matched real replay over the identical window.

| | Real | DDIM-1 |
|---|---|---|
| Unique mids | 111 | **309** |
| 1s return volatility | 1.33 bp | **4.93 bp** |
| Price range | 55 tk | **229 tk** |
| Executed % | 5.4 | **11.6** |
| Limit % | 50.0 | 48.2 |
| Cancel % | 44.7 | 40.1 |

**The collapse** (`analysis/plots/methodology_replication/summary.md`):

| | First below \$33.50 | Max after | Recovers? | Worst 60s move | Session end |
|---|---|---|---|---|---|
| Real | 11:12:53 | \$33.81 | **yes** | −\$0.220 | \$33.76 |
| DDIM-1 | 11:13:41 | \$33.08 | **no** | **−\$1.180** | \$31.86 |

Both start at \$33.74. Real dips to \$33.47 and recovers; DDIM-1 crosses the same level and craters
to \$31.70. Figure: `analysis/plots/methodology_replication/1_replication_price_ood_collapse.png`.

⚠️ **State the conditions explicitly.** This is a different checkpoint (0.763), day (01-29) and
window (2h) from Tables 1–2. It cannot be presented as a row alongside them. Frame it as: *"TRADES's
single-step configuration was run separately over a two-hour window, where its failure mode becomes
visible."* The 2-hour window is not optional — the collapse occurs at **minute ~73**, so any 30-minute
evaluation misses it entirely. That fact is itself worth stating.

Corroboration available: the paper's own reported predictive-score degradation on TSLA
(1.213 → 3.146). Attribute clearly as **their** number, not a reproduction.

---

## Provenance notes — read before citing

**Two distinct DDPM-100 runs exist, and they disagree.**

| | Documented replication | `sweep_results/DDPM_100/` |
|---|---|---|
| Executed % | 18.8 | 10.17 |
| Unique mids | 23 | 101 |
| Window | 30-min | 09:30–10:30 (60-min) |
| Source | `deterministic_sampler_findings.md` | July-5 sampler sweep |

Both are labelled ckpt 0.681. **Cite the documented one** (Tables 1–2); `sweep_results/` was a
different exercise (a broad sampler sweep, also covering DPM-Solver/UniPC) and its numbers should not
be mixed in.

**Checkpoint retention.** The checkpoints behind these runs — the July-lineage 0.681, plus 0.763 and
0.774 — are not retained. Surviving checkpoints are a later family (dropout 0.1, lr 2.5e-4, 28 July).
State this plainly as a limitation; it is unremarkable, and neither claim in §5.1 depends on
re-running them:

- The DDPM-100 replication claim is measured against **their released CSVs**, not against our other runs.
- The DDIM-1 failure claim is measured against **real market data** on a matched window.

Neither is a comparison of our runs against each other, so nothing here requires a controlled pair.

**Flags are not recoverable from checkpoints.** `UNCLAMP_DEPTH`/`PRICE_REANCHOR` are file-flags —
filesystem state at training time, not saved into the `.ckpt`. So the pre/post-fix status of any
surviving checkpoint cannot be certified retrospectively. Relevant if a reviewer asks whether the
replication used the original TRADES pipeline: the answer is that these runs predate both fixes by
project chronology, but that is chronological evidence, not a property readable from the artefact.

---

## Suggested §5.1 ordering

1. **The accurate baseline** — Table 1, then Table 2's DDPM rows. Land the replication, then note
   that their model already over-executes against real (7.0% → 14–17%).
2. **The fast baseline** — Table 3 and the collapse figure. Note the minute-73 timing and that a
   30-minute evaluation would miss it.
3. **The gap** — two sentences. DDPM-100 is accurate but slow; DDIM-1 is fast but unusable. Forward-
   reference §5.6.

## Still open

- The **runtime** figures for DDPM-100 vs DDIM-1 aren't in these tables. Timing exists in
  `sweep_results/summary.csv` (DDPM-100 11,280s vs DDIM-10 3,063s) but from the July sweep, on a
  different window — so it is not directly comparable to the ~15 min vs 40–60 min figures used in
  §5.3. Worth one clean timing statement per baseline, from a single consistent source.
- Whether the **decode-time fix is still needed once the data-pipeline fixes are in** — the pending
  ablation (DDIM-1 and vanilla DDIM-10 on ckpt 0.724). Belongs in §5.2/§5.3, not here.
