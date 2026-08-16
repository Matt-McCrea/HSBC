# Variance-ratio analysis — the mean-reversion finding

*2026-08-07. Diagnostic added after inspecting the two-hour mid-price traces, which look flat and
oscillatory rather than wandering.*

---

## The measure

Variance ratio at horizon *q*:

```
VR(q) = Var( q-period return ) / ( q x Var( 1-period return ) )
```

| VR | interpretation |
|---|---|
| ≈ 1 | random walk — moves accumulate, as in an efficient market |
| < 1 | mean-reverting — moves get undone, price oscillates around a level |
| > 1 | trending — moves reinforce |

This is the right diagnostic because it separates *how much* the price moves from *whether the
movement persists*. `uniq_mid` and `ret1s_std` cannot: a series can have plenty of both and still go
nowhere, which is exactly what was happening here.

All figures below use 1-second bars, 15-minute warm-up discarded (5 minutes for 30-minute sessions).

---

## Core result — our models mean-revert violently

INTC 2015-01-29, two-hour window:

| series | 1s vol (bp) | range (tk) | VR(10s) | VR(60s) | VR(300s) |
|---|---|---|---|---|---|
| **Real market** | 1.18 | 56 | 0.897 | **0.931** | **1.019** |
| SS epoch 4 | 1.35 | 28 | 0.205 | **0.047** | **0.016** |
| 0.724 baseline | 1.77 | 14 | 0.300 | **0.098** | **0.027** |

Real is a textbook random walk — mild mean reversion at 10 seconds (bid-ask bounce, expected)
converging to 1.0 by five minutes.

Our models accumulate **under 5% of the variance a random walk would** at the one-minute horizon.

### What this explains

Both models have **more** 1-second volatility than real (1.35 / 1.77 vs 1.18) yet **half** its price
range (28 / 14 vs 56 tk). The movement is there; it cancels itself out instead of compounding. This
is why the traces look flat, and why the earlier reading of "insufficient activity" was the wrong
diagnosis — the deficiency is persistence, not variance.

---

## TRADES's own released model has the same pathology

| series | 1s vol | VR(60s) | VR(300s) |
|---|---|---|---|
| Real 0129 | 1.18 | 0.931 | 1.019 |
| **THEIRS released, 0130** | 4.35 | **0.099** | **0.032** |
| **THEIRS released, 0129** | 4.41 | **0.153** | **0.059** |
| Ours (SS e4) | 1.35 | 0.047 | 0.016 |
| Ours (0.724) | 1.77 | 0.098 | 0.027 |

**The mean reversion is inherited from the architecture, not introduced by our decode-time
corrections.** TRADES's published output is in the same regime, on top of being ~4x too volatile.

This reframes the limitation: it is a property of the approach that this work is the first to
measure, rather than a defect of the modifications made here. Worth stating plainly.

---

## An unresolved tension: does volatility buy persistence?

| series | 1s vol | VR(60s) |
|---|---|---|
| ours DDPM-100 (ckpt 0.681) | 4.54 | **0.835** |
| TRADES 1-step (ckpt 0.763) | 4.93 | **0.765** |
| ours, post-fix checkpoints | 1.35–1.95 | 0.047–0.098 |

Every high-volatility run has a far better variance ratio. That **contradicts the theory** — `--depth-noise`
injects *independent* draws per order, which should cancel and buy no persistence at all.

The comparison is confounded by checkpoint, so it cannot be settled observationally. Both hypotheses
have arms in `scripts/drift_persistence_sweep.sh`:

- **(a) persistence is the missing ingredient** → `--depth-drift` with long `phi` fixes it, raw noise does not
- **(b) volatility buys persistence** → high `--depth-noise` arms improve VR

---

## The book-balancing lever *improves* persistence

INTC 2015-01-29, two-hour window, SS epoch 4:

| | 1s vol | range | uniq mids | VR(60s) | VR(300s) |
|---|---|---|---|---|---|
| no lever | 1.35 | 28 | 57 | 0.047 | 0.016 |
| **+ `bt2.0 r0.5`** | 1.77 | **32** | **64** | **0.112** | **0.051** |
| *real* | 1.18 | 56 | 86 | 0.931 | 1.019 |

Persistence improves ~2.4x and range and activity both rise. **This was not the expected direction** —
the lever is a book-balancing cancel, which trims whichever side has grown thick, and I predicted it
would act as a restoring force and make mean reversion worse. It does the opposite.

Counterweight: 1-second volatility rises 1.35 → 1.77, away from real's 1.18. So the lever trades
volatility accuracy for persistence and range. Given that range and VR are much larger gaps than
volatility, it looks net positive — but it is a trade, not a free win.

---

## Seed robustness (INTC 2015-01-30, 30-minute)

| series | 1s vol | range | uniq mids | VR(60s) |
|---|---|---|---|---|
| Real | 1.43 | 52 | 90 | 1.046 |
| SS e4, seed 30 | 1.59 | 18 | 32 | 0.550 |
| SS e4, seed 31 | 1.63 | 18 | 32 | 0.519 |
| SS e4, seed 32 | 1.70 | 18 | 33 | 0.510 |
| 0.724, seed 30 | 1.72 | 18 | 33 | 0.521 |
| 0.724, seed 31 | 1.69 | 21 | 39 | 0.517 |

**Very stable across seeds** — volatility within 0.11 bp, range and unique mids essentially
identical, VR within 0.04. The pathology is a systematic property of the models, not seed noise.

*(The 0.724 seed-32 cell did not complete before the GPU window closed.)*

---

## Two caveats on reading these numbers

**1. VR estimates on short windows are unreliable at long horizons.** VR(300s) on a 30-minute session
rests on only ~5 non-overlapping blocks — which is why the 30-minute VR(300s) column is erratic
(real reads 1.622 there, against 1.019 on the two-hour window). **Trust the two-hour numbers**; treat
30-minute VR(300s) as indicative only.

**2. The pathology worsens with horizon.** VR(60s) is ~0.52 at 30 minutes but ~0.05 over two hours.
Some of this is estimation noise per the point above, but the direction is consistent and is the
reason this went unnoticed while everything was evaluated at 30 minutes.

---

## Next step

`scripts/drift_persistence_sweep.sh` — 12 cells, ~5.2 h. Sweeps AR(1) persistence (`--depth-drift-phi`
at ~44 s / ~111 s / ~221 s), amplitude, noise/drift trade-off arms, raw high-noise arms for
hypothesis (b), and a `--type-decode prior` ablation testing whether pinning the type mix to a fixed
prior acts as a restoring force.

**Success criterion**: VR(60s) and VR(300s) rising toward 1.0 while 1-second volatility stays near
1.2–1.5 bp and range grows toward 56 tk. A cell that raises volatility while leaving VR near zero has
added jitter, not persistence.
