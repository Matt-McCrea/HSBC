# Simulator validation against real order flow — drop-in text

All figures: identical policies, identical held-out seeds, identical estimator code.
Only the background market differs — TRADES generative vs the real INTC message stream.
Generative arm: 54 episodes. Replay arm: 54 episodes (28,871 fills for the impact figures).

---

## Sentences to lift

**Volatility is reproduced.** Realised volatility in the simulated market is 155.6 price
units per 30 s against 166 measured on real order flow over the same windows, a 6%
discrepancy.

**The simulator drifts; the real market does not.** Mid-price drift over a five-minute
episode is +5.33 bps (*t* = 4.30) in the simulator against +1.62 bps (*t* = 0.81) on real
data, so the martingale assumption underlying Almgren–Chriss holds on real flow and fails
in simulation.

**Orders fill roughly three times too readily.** The execution rate is 17–18% in the
simulated market against 6% measured on real flow, so simulated fills are provided that a
real book would have withheld.

**Temporary impact is overstated by about two and a half times.** The impact coefficient
is 4.751 in simulation against 1.894 on real data.

**The simulated mid-price moves about twice as much.** A five-minute episode visits 8.94
distinct mid prices in simulation against 4.50 on real flow.

**The cost of crossing the spread is roughly doubled.** The passive-to-aggressive slippage
swing is 5.40 bps in simulation against 2.46 bps on real data.

**Policy rankings survive the transfer to real data.** TWAP is cheapest and carries most
inventory risk in both arms, the Almgren–Chriss schedule is dearest and safest in both, and
the learned policy lies between them in both; no policy dominates another in either market.

**Verdict.** The simulator is usable for relative policy comparison and unusable for
absolute cost estimation: rankings are preserved, but every cost level is optimistic, and
any shortfall measured in it should be read as a lower bound on live trading cost.

---

## Table

| Property | Simulated | Real | Direction of error |
|---|---|---|---|
| Volatility σ (per 30 s) | 155.6 | 166 | **matched (−6%)** |
| Drift per episode | +5.33 bps (*t*=4.30) | +1.62 bps (*t*=0.81) | drifts when it should not |
| Execution rate | 0.17–0.18 | 0.06 | ~3× too high |
| Temporary impact η | 4.751 | 1.894 | ~2.5× too high |
| Distinct mid prices | 8.94 | 4.50 | ~2× too many |
| Passive→aggressive swing | 5.40 bps | 2.46 bps | ~2.2× too high |
| Policy ranking | TWAP / RL / AC | TWAP / RL / AC | preserved |

Drift is also +7.09 bps (*t* = 7.88) over the 114-episode generative training log, so the
simulated drift is not an artefact of the evaluation sample.

---

## Two caveats that must be stated

**This validates the simulator as configured, not the depth-noise fix.** Every run used
`depth_noise = 0.3`; the accelerated sampler was never compared against the unfixed
configuration in this setting, so these results characterise the fixed simulator's biases
rather than demonstrating an improvement over the alternative.

**Execution rate is measured slightly differently in the two arms** — per placed order in
the simulator, and as executions per new order on real flow, where an order filling in
several parts contributes more than once. The real figure is therefore a mild upper bound
and the ~3× multiple a mild lower bound.
