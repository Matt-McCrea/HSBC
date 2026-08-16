# Decode-time repair results — 2026-07-15 overnight (the decisive night)

Full data: `overnight_reshape/20260714_234241/summary.md` (auto-table at top). Checkpoint `0.635`
(unclamped retrain), DDIM/DPM++ at 10 steps unless noted, window 09:30–10:00 (real: 69 mids,
7.0% exec), UNCLAMP flag on.

## Headline

**Per-sample decode noise (`--depth-noise`) is the first and only lever in this project that
unfreezes deterministic sampling controllably** — a dial, not a cliff. Combined with size-reshape
and prior type-decode it produces a realistic, moving, wall-free market at 10-step cost:

| | DDIM10 + dn0.3 + sr + prior | real |
|---|---|---|
| flow mix (limit/cancel/exec) | 51.5 / 40.5 / **7.9%** | 49.2 / 43.8 / 7.0% |
| unique mid-prices | 47 | 69 |
| bid/ask_1 mean size | 3270 / 1443 (**no walls**) | 3899 / 2117 |
| mid range | 33.725–33.955 (inside real's envelope) | 33.605–33.975 |
| B_crossing_limit / A_market | 1076 / 0 | — |
| cond_z[depth] | −1.63..4.33 (sane) | — |

Seed-robust: B = 1076 / 1855 / 1305, mids = 47 / 43 / 53, exec = 7.9 / 12.1 / 8.9% across seeds
30/31/32. Runtime ~15 min per 30-min session vs ~40–60 min for DDPM-100 — the 10× acceleration
with near-DDPM realism that was the thesis target.

Dose-response (the dial): σ = 0.15 → 0.3 → 0.5 gives B = 19 → 248 → 492 (dn-only cells), mids
30 → 37 → 49, monotonic, no instability anywhere on the curve. σ=0.5 overshoots (exec 10.3%,
book thins to 725–1497); **σ≈0.3 is the sweet spot**.

Curiosity worth keeping: with the prior type-decode on this checkpoint, market-type decodes went
to literally ZERO — the entire 7.9% execution flow arrives via Channel B (limit orders priced
across the spread). The market functions with no "market orders" at all; the flow-mix EXEC bucket
still lands on real's value because crossing limits log as executions. The type channel remains
per-checkpoint fragile (plain decode over-produces at 4.4% on dn0.5; prior kills it entirely) —
aggregate flow is right either way, but this axis is calibration-sensitive.

## Quantile depth reshape (dr): refuted — pins the marginal, breaks the joint

Every dr cell: B_crossing_limit ∈ {1, 2} despite the mechanism engaging fully (depth_applied
≈6–14k per run) and negative depths being produced. Two failure mechanisms, both visible in the
diagnostics:

1. **The far tail destabilizes a thin book.** The real LIMIT-depth marginal reaches +220 ticks —
   supportable in a dense real book, but remapped context-free into a thin simulated book those
   far-out orders become future best quotes when near-touch liquidity cancels away → the mid
   teleports → drift → conditioning blowup. Signature: dr mid drifted to $34.97 and
   cond_z[depth] max=1354; DPMpp+dr to $35.20 / max=1369 — the same instability class as
   CHURN/HYBRID, arrived at through a completely different mechanism.
2. **The negative mass is too rare and too small to cross.** ~0.9% of reshaped limits go
   negative (matching real's marginal) at mostly −1/−2 ticks — but −1 tick only crosses a
   1-tick spread, and the drifting/walling book rarely offers one. ≈120 attempts, ≈1 success.

Control cells seal it: **DDPM100+dr+sr is WORSE than plain DDPM** (10 mids vs ~23, walls 27–31k
vs none) — the reshape harms even a healthy sampler. The NFE ladder (DDIM5/20 + dr) stayed at
B≈1–2 throughout. Verdict: context-free marginal matching is the wrong tool for a closed-loop
system; what matters is SMALL, LOCAL variance near the touch — exactly what dn provides (σ=0.3
in z ≈ ±0.8 ticks) and dr's global remap does not.

Why dn works where everything else failed, in one line each:
- vs depth-temp: noise acts per-SAMPLE (splits the collapsed atom); temp slides the whole atom.
- vs CHURN/HYBRID: noise touches ONLY the decoded depth scalar, after sampling — it cannot feed
  back through the sampler or the other channels, so no instability.
- vs dr: noise is local (±0.8 ticks around the model's own output) — it fattens the near tail
  without importing a context-inappropriate far tail.
- vs unclamp retrain alone: the retrain fixed sign frequency but not magnitude-under-determinism;
  noise directly supplies the missing variance at the exact channel that needs it.

## Size-reshape (sr): not a freeze lever, but keep it

`DDIM10_sr` alone: B=0 — the size axis does nothing for the freeze (as predicted). But sr
eliminated the size_range drop-and-resample waste completely (extra_batches: thousands → 5–70)
and normalizes placed sizes; it costs nothing and speeds everything up. Keep it in the recipe.

## The open problem: long-horizon drift (ET1100)

09:30–11:00 (75 min generation) on the winning config: **alive but drifting** — 208 unique mids
(real window: 117), exec 11.2%, but the mid slid to $32.215 (−5% from open) and cond_z[depth]
hit −1259 late in the run. The asymmetry (ask-side mean 1493 vs bid 3495) suggests one-sided
liquidity consumption compounding. Short-horizon realism is in hand; long-horizon stability is
the honest remaining gap — candidate mitigations (untested): mild σ decay over the session,
drift-aware conditioning re-anchoring, or simply characterizing the usable horizon (~30–45 min).

## Also
- `DDIM10_dr_CLAMPED0.656` errored (log not yet inspected) — moot given dr lost on the merits.
- Recommended next: LOB-Bench score the winner vs DDPM-100 vs real (manifest auto-printed by the
  shell); update `trades_explainer.html` and the HSBC brief with the dn result.
