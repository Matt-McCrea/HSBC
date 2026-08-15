# The Final Model — handoff notes for the write-up

*Prepared 2026-08-04. This document is self-contained — it doesn't assume you've seen the project
before. It covers what the model is, why it's the final one, how we know it works, what we tried
that didn't work, and where to find the underlying data if you need to check or cite a number.*

**UPDATE, 2026-08-05 — read before treating §2 as settled.** Section 5 below (the scheduled-sampling
retrain) was written when that retrain looked like a clean negative result. It wasn't — full LOB-Bench
scoring, done after this document was first drafted, shows the retrain epochs (2, 3, 4) all score
*better* overall realism than `0.724_epoch=0` (grand mean Wasserstein 0.468 → as low as 0.346), driven
mainly by a large spread-realism gain, even though the original activity-level goal still wasn't met.
All three retrain epochs are also independently confirmed stable. **Which checkpoint is actually
"the final model" is now an open question, to be decided with input from supervisors — not yet
resolved.** Full comparison data and figures: `analysis/appendix_lobbench_and_refutations.md`,
`analysis/appendix_checkpoint_evidence.md`, `analysis/plots/hsbc_ss_retrain_*/`. Everything else in
this document (the acceleration result, the selection methodology, the mechanism) still stands
regardless of how that decision lands.

---

## 1. What this is, in plain terms

The dissertation builds an accelerated market simulator: a diffusion model (a type of generative AI,
the same family behind image generators like Stable Diffusion, but applied to sequences of stock
market orders) that can generate realistic limit-order-book trading activity much faster than the
standard approach.

The starting point was a published architecture called **TRADES** (Berti, Prenkaj, Velardi, 2025,
arXiv:2502.07071), which normally generates a simulated trading session using 100 sequential
"denoising" steps (like a sculptor removing marble in 100 small passes to reveal a statue). This
project's core contribution is showing that **10 steps work just as well — actually slightly
better** — provided a specific set of corrections are applied at generation time. That's roughly a
**2.7× speedup** with no loss in realism.

The second contribution, and the subject of this document, is showing that this fast approach
**doesn't just work on one hand-picked day — it works reliably across an entire month of real
trading days**, which is a materially harder and more useful claim.

## 2. The final model

**Checkpoint**: `val_ema=0.724_epoch=0` (an internal identifier — `0.724` is a validation loss value,
`epoch=0` marks which point in training it was saved at; not meaningful numbers to quote directly,
just the model's "name" for reference purposes)

**What it does differently from the raw published model**:
- Uses the fast 10-step generation process instead of the standard 100-step one
- Applies four small corrections at generation time (details in §6, if the write-up wants to explain
  the mechanism):
  - restores some randomness the fast process otherwise loses (this is what stops the market
    "freezing" — see §6)
  - matches the realistic distribution of order sizes
  - picks order types (buy/sell/cancel) more sensibly
- Was trained with two data-preprocessing bugs fixed (a clamp that had silently discarded every
  price-moving order from the training data, and a price-normalisation issue that broke down late in
  a trading session) — both are described in more detail in the main methodology chapter if this
  section needs to reference them

## 3. Why this is *the* model — the selection process

This matters for the write-up because it's a methodological contribution in its own right, not just
"we picked the best number."

A prior version of the project's best-performing checkpoint was lost to a storage failure. Recovery
efforts surfaced six candidate checkpoints from a later training run, and rather than assume any of
them behaved like the lost one, **each was tested systematically against every one of 20 real trading
days in January 2015** (a full trading month, order-book data sourced from LOBSTER/NASDAQ TotalView).

The test: run each checkpoint on each day, capped at 40 minutes of computer time per day. If a
checkpoint failed to produce a stable, realistic session within that cap on even **one** day, it was
discarded immediately — no point spending further compute confirming a checkpoint already known to
fail. This is an efficient elimination search, not an exhaustive one.

**Result**: 5 of the 6 candidates failed this way, most within the first day or two tested. **One
checkpoint — the model described in §2 — cleared all 20 days**, with no instance of the market
"freezing" (going eerily quiet) or "drifting" (the price wandering unrealistically far from where it
started).

## 4. The evidence it actually works

Two separate kinds of evidence, both important for the write-up:

### 4a. It's stable
Across all 20 days, a measure of price volatility (how much the price wiggles second-to-second)
stayed inside the same band real market data shows (1.5-2.5 basis points), on every single day. It
never froze and never ran away.

### 4b. It's realistic — scored against real data with an established benchmark
The benchmark is **LOB-Bench** (Nagy et al., 2025), an academic tool that measures how different a
generated trading session is from a real one across six dimensions: bid-ask spread, order
inter-arrival timing, order-book imbalance, order-flow imbalance, and the depth of limit/cancel
orders in the book. The metric is Wasserstein distance — lower means more realistic; there's no fixed
"good" threshold, it's used comparatively.

**Full-month average** (20 days), the model on its own:

| Metric | Mean | Best day | Worst day |
|---|---|---|---|
| Cancel-order depth | 0.161 | 0.094 | 0.387 |
| Limit-order depth | 0.200 | 0.133 | 0.255 |
| Order-book imbalance | 0.432 | 0.350 | 0.533 |
| Order timing | 0.625 | 0.407 | 0.877 |
| Order-flow imbalance | 0.673 | 0.508 | 0.820 |
| Bid-ask spread | 0.719 | 0.591 | 0.954 |
| **Overall average** | **0.468** | | |

Reading this: the model is strongest at reproducing order depth/cancellation behaviour, weakest at
reproducing spread and order timing — a genuine, specific limitation worth stating plainly rather
than glossing over.

**Same-day, apples-to-apples comparison** (2015-01-30, scored against two other configurations on
that identical day):

| Configuration | Overall score | Note |
|---|---|---|
| Original 100-step baseline (DDPM) | 0.507 | the un-accelerated published approach |
| An earlier, single-day-tuned version of this project's method | 0.298 | tuned specifically to this one day, never tested elsewhere |
| **This final model** | **0.447** | |

This is worth stating honestly in the write-up: the final model beats the original 100-step baseline,
but doesn't beat the earlier single-day-tuned version *on that version's own home turf*. That's the
expected trade-off — a model tuned to one specific day can look better on that day than a model
designed to generalise across many days, without the tuned version having ever been shown to work
anywhere else. The full-month result (§4b, above) is the more meaningful comparison for the
project's actual claim.

## 5. What was tried afterward, and didn't work (a real negative result)

Once the final model was confirmed stable, one further idea was tried: could the model be
**retrained** to be a bit more "active" — its order-book activity level was found to be somewhat
quieter than real markets, even though stable. The technique, called *scheduled sampling*, trains the
model partly on its own previous output rather than only ever on real historical data, the idea being
it learns to correct its own small errors instead of only ever seeing "perfect" inputs.

**This did not work.** Across five further rounds of training, the activity level did not move closer
to real levels — it stayed flat, if anything slightly declining. One round even introduced a new
stability failure (a timeout on a day the previous round had handled fine), with no compensating
benefit. This retraining attempt was stopped, and the original model from §2 was kept unchanged as
final.

**Why this is worth including in the write-up, not omitting**: it's methodologically honest, it's
consistent with how every other unsuccessful technique in this project was documented (there's a
running list of refuted approaches — an execution-rate controller, a "cancel-boost" bias, a
conditioning clip — all tried and reported, not hidden), and it demonstrates the checkpoint-selection
methodology in §3 generalises: it's not just for picking between existing checkpoints, it's how any
future training attempt should be evaluated before being adopted.

## 6. The mechanism, if the write-up wants to explain *why* the fast version works at all

(Optional detail — include if there's room for a "how it works" passage.)

The 10-step process is faster because it takes bigger, smarter jumps instead of 100 small ones. The
problem with doing this naively is that it makes the model's output too *deterministic* — it loses
the natural randomness that makes some orders cross the spread and actually execute a trade, so the
naive fast version tends to "freeze" (the market goes quiet because nothing generated is realistic
enough to trigger a trade). The fix that makes 10 steps work is to deliberately re-inject a small,
carefully-tuned amount of randomness back in at generation time — restoring just enough variability
for realistic trades to occur, without overdoing it and causing runaway, unrealistic price swings.

## 7. Caveats and honest limitations

- Everything above is on one stock (Intel, ticker INTC) and one calendar month (January 2015). It has
  not yet been tested on other stocks or other months.
- The full-month score in §4b (0.468) doesn't yet have a like-for-like full-month comparison against
  the 100-step baseline — that comparison currently only exists for the single day in the table. A
  full-month baseline run is planned but not yet done (needs further compute time).
- The model is somewhat quieter (less order-book activity) than real markets, even though stable —
  see §5. This wasn't fixed by the retraining attempt and remains an open characteristic of the model
  rather than a solved problem.

## 8. Where the underlying data lives (for verification/citation)

- Full-month LOB-Bench scores: `lob_bench_0724_full_month/SUMMARY_mean_wasserstein.csv`
- Checkpoint-selection search results: `ckpt_search/` (various dated subfolders, `progress.txt` +
  `STATUS.txt` in each)
- Project status / running log of what's been tried and found: `analysis/PROJECT_STATUS.md`
- Public-facing progress summary (same results, presentation format): the HSBC progress page
  (artifact link — ask if you need it re-shared)
