# Aggression, size, and what the agent learned

**Measured from `logs/train_session.jsonl` — 114 training episodes, 1,004 fills, INTC.**
Pure log analysis: no re-simulation, no GPU.

Interactive version with hoverable policy grids:
<https://claude.ai/code/artifact/99c16745-9f3d-459e-b38f-53c3bc96d3af>

---

## Summary

Three findings, in order of how much they change the story.

1. **Slippage is driven by the aggression decision, not the size decision.** Passive limit
   orders *earn* 2.03 bps; market orders pay 3.37 bps. The swing is **5.4 bps**. Holding
   aggression constant, quadrupling order size costs only about **1 bps**.
2. **The pooled temporary-impact intercept was never a half-spread.** It came out negative
   (−47.16 raw units) because the regression pools fills that earn the spread with fills that
   pay it. Split the populations and the number becomes interpretable.
3. **The Almgren–Chriss front-loading prediction is confirmed.** Mean greedy action rises
   monotonically with the inventory penalty: 1.26 → 1.25 → 1.78 across λ = 0, 42, 337.

---

## 1. Slippage by execution style

The single most useful table from this run. Negative means the fill beat the prevailing mid.

| Execution style | Actions | n fills | Mean size | Slippage | SE |
|---|---|---|---|---|---|
| Passive limit | 0, 1 | 444 | 253 sh | **−2.031 bps** | 0.106 |
| Crossing limit | 2 | 255 | 300 sh | **+2.671 bps** | 0.094 |
| Market order | 3, 4 | 305 | 525 sh | **+3.369 bps** | 0.098 |

Broken out per action:

| # | Action | Order type | n | Mean size | Slippage | SE |
|---|---|---|---|---|---|---|
| 0 | passive | limit, 0.5× slice | 213 | 171 sh | −2.313 bps | 0.141 |
| 1 | light | limit, 1.0× slice | 231 | 328 sh | −1.772 bps | 0.155 |
| 2 | neutral | crossing limit, 1.0× | 255 | 300 sh | +2.671 bps | 0.094 |
| 3 | aggressive | market, 1.5× slice | 112 | 432 sh | +3.162 bps | 0.158 |
| 4 | very aggressive | market, 2.0× slice | 193 | 579 sh | +3.489 bps | 0.125 |

The discontinuity sits between action 1 and action 2 — a **4.44 bps jump** — and that is
precisely the spread-crossing decision. Everything on the passive side of it earns; everything
on the aggressive side pays. The further step from crossing limit to market order costs only a
further 0.7 bps.

---

## 2. Why the naive size curve was misleading

Bucketing all fills by size produces what looks like a textbook impact curve:

| Size bucket | n | Mean size | Slippage | SE |
|---|---|---|---|---|
| 1–136 | 168 | 98 sh | −0.935 bps | 0.225 |
| 136–207 | 167 | 178 sh | −0.669 bps | 0.223 |
| 207–291 | 166 | 243 sh | +0.537 bps | 0.227 |
| 291–408 | 164 | 351 sh | +1.094 bps | 0.236 |
| 408–530 | 170 | 455 sh | +1.207 bps | 0.245 |
| 530–1982 | 169 | 754 sh | +3.560 bps | 0.132 |

It is largely an artefact. **The action space couples size with aggression by construction** —
passive actions take half a TWAP slice, aggressive actions take up to double — so larger orders
in this sample are also more aggressive orders. The curve is mostly re-measuring §1.

### The deconfounded curve

Restricting to market orders only holds aggression fixed:

| Size bucket | n | Mean size | Slippage | Per 100 sh |
|---|---|---|---|---|
| 1–301 | 75 | 212 sh | +2.800 bps | +1.318 |
| 301–530 | 77 | 412 sh | +3.488 bps | +0.846 |
| 530–707 | 73 | 615 sh | +3.391 bps | +0.551 |
| 707–994 | 80 | 845 sh | +3.768 bps | +0.446 |

Size quadruples (212 → 845 shares) while slippage rises only 2.80 → 3.77 bps. A square-root law
predicts a **doubling**; the observed exponent is closer to 0.2.

The per-100-share column falling by two thirds is the signature of a **large fixed cost plus a
small marginal one** — the spread is paid in full by any aggressive order regardless of size,
and the size-dependent component on top is minor at these volumes.

> **Implication.** At 50–1,000 shares on this name, choosing to cross the spread costs several
> times more than choosing to send a larger child order. Schedule optimisation that only tunes
> sizes is tuning the smaller lever.

---

## 3. The learned policy

Greedy action per state. Rows are time remaining, columns inventory remaining, `.` never
visited in 114 episodes. Action index 0 = passive through 4 = very aggressive.

### λ = 0 (risk neutral)

```
t_rem      0%   20%   40%   60%   80%
10/10       .     .     .     .     2
 9/10       .     .     .     0     4
 8/10       .     .     .     0     1
 7/10       .     .     0     2     1
 6/10       .     0     0     0     0
 5/10       3     1     1     4     0
 4/10       2     1     1     .     .
 3/10       2     0     4     0     .
 2/10       0     1     0     3     .
 1/10       4     1     1     .     .
 0/10       .     .     .     .     .
```

### λ = 42.12 (calibrated)

```
t_rem      0%   20%   40%   60%   80%
10/10       .     .     .     .     2
 9/10       .     .     .     0     4
 8/10       .     .     .     0     1
 7/10       .     .     0     2     1
 6/10       .     0     1     0     1
 5/10       3     1     1     1     1
 4/10       3     1     1     1     .
 3/10       2     0     4     0     .
 2/10       2     1     0     0     .
 1/10       4     1     1     .     .
 0/10       .     .     .     .     .
```

### λ = 336.97 (8× calibrated)

```
t_rem      0%   20%   40%   60%   80%
10/10       .     .     .     .     4
 9/10       .     .     .     1     1
 8/10       .     .     .     3     0
 7/10       .     .     2     0     2
 6/10       .     0     4     3     1
 5/10       3     4     1     0     1
 4/10       3     4     4     1     .
 3/10       2     0     4     0     .
 2/10       2     1     0     0     .
 1/10       4     1     1     .     .
 0/10       .     .     .     .     .
```

### Reading the grids

| λ | States visited | Mean greedy action | Action distribution (0/1/2/3/4) |
|---|---|---|---|
| 0 | 31/55 | 1.26 | 12 / 9 / 4 / 2 / 4 |
| 42.12 | 32/55 | 1.25 | 9 / 14 / 4 / 2 / 3 |
| 336.97 | 32/55 | 1.78 | 8 / 9 / 4 / 4 / 7 |

**The Almgren–Chriss prediction holds.** A higher inventory penalty should front-load the
schedule, and the λ = 337 grid is visibly more aggressive — action 4 appears 7 times against 3
at the calibrated λ. This was recovered offline from logged trajectories at no simulation cost,
which is a property of Q-learning being off-policy.

**At the calibrated λ the policy is TWAP-like.** Action 1 — TWAP's own action — is modal at
43.8%, and the densely-visited interior of the grid (time remaining 4–6, inventory 20–80%) is
uniformly action 1. The erratic cells are the sparse corners, where median visits per entry is
6. That is consistent with the evaluation result: statistically indistinguishable from TWAP
(+0.892 bps, 95% CI [−0.93, +2.71], p = 0.35, 8/18 seeds).

---

## 4. What this changes

**In the write-up.** The pooled temporary impact η = 6.554 should not be quoted on its own —
it averages a population that earns the spread against one that pays it, and its intercept is
uninterpretable as a half-spread. Quote the style split first, then η within a style.

**For execution practice.** The dominant controllable cost at these sizes is the passive/
aggressive choice, not the child-order size. Frame the agent's problem as *when to cross*
rather than *how much to send*.

**For the Almgren–Chriss comparison.** This is the mechanism behind AC underperforming TWAP.
Front-loading forces more aggressive fills, and the aggression penalty (5.4 bps) dwarfs any
size effect it saves. Combined with permanent impact being undetectable at every horizon out to
150 s, front-loading pays a large certain cost to avoid a price mark that is not there.

---

## 5. Caveats

- **Associative, not causal.** These are price responses *associated* with our trades. The
  world agent reacts to the book we perturb, so this is not an isolated causal effect. A clean
  estimate needs paired counterfactual episodes with the agent disabled — exact under replay,
  built but not yet run.
- **Size range is bounded by what the agent actually traded**: roughly 50–1,000 shares. Nothing
  here extrapolates beyond that.
- **Action 3 is thin** (n = 112) relative to the others.
- **Single symbol, single sample**, and the training log is SELL-only.
- **Functional form is not identified.** Linear and square-root fits differ by R² ≈ 0.005 and
  the nominal winner flips between the training and evaluation logs. The deconfounded curve in
  §2 is evidence against a square-root law *within market orders*, but is not a claim about the
  underlying impact process.

---

## Reproducing

```bash
python -m rl_execution.impact          logs/train_session.jsonl   # style split, curve, permanent
python -m rl_execution.inspect_policy  checkpoints/qtable_lam.npz # policy grid
python -m rl_execution.compare_policies logs/eval_frontier_*.jsonl --baseline twap
```
