# TRADES sampler acceleration — diagnostic briefing

*State of play for the HSBC call. Covers what we diagnosed, how, what we tried, what we ran, and
what remains open. Code references given as `file` → `function` so specifics can be pulled up live.*

---

## 1. Objective and why it matters

TRADES is a transformer-diffusion generative "world model" for limit-order-book (LOB) simulation:
an agent inside ABIDES generates an entire trading session order-by-order, each order produced by
running the diffusion model's full reverse process (100 denoising steps) from random noise. That
per-order cost makes any downstream use (e.g. training an execution agent inside the simulator)
infeasible. **The work is to accelerate sampling** — replacing the paper's 100-step DDPM / DDIM
with faster ODE solvers (DPM-Solver, DPM-Solver++, UniPC) — and to **map the speed–realism
trade-off**.

## 2. The problem we hit

In *closed-loop* simulation (the model conditions on its own prior outputs for a full session),
the deterministic fast solvers do not degrade gracefully — they break the market:

- **Few-step deterministic** (DDIM η=0, DPM-Solver++ at low steps): the mid-price **freezes**.
- **Many-step deterministic** (DDIM 100 steps): the price **diverges** (crashes / runs away).

DDPM (stochastic, 100 steps) is the only sampler that produces a moving, realistic market.

## 3. How we diagnosed it — the instrumentation we built

So that every claim is backed by measurement rather than eyeballing plots:

- **Always-on diagnostics** in `ABIDES/agent/WorldAgent.py` → `kernelTerminating` prints a
  greppable block per run: pre-drop **type** histogram (`decoded_type_counts`), pre-drop **depth**
  histogram (`depth_hist`), per-reason **drop counters** (`drop_counts`), **resample** stats, and
  per-channel **conditioning z-score** min/mean/max (`cond_stats`).
- **`evaluation/quantitative_eval/flow_mix.py`** — order-type mix, unique mid-price count, and
  top-of-book volume for a generated CSV vs the real market-replay CSV.
- **`evaluation/diagnostics/open_loop_eval.py`** — samples the model on **real test-set windows**
  (no ABIDES feedback loop) to separate *model × sampler* effects from *simulator feedback*.
- **Staged, resumable, self-evaluating sweep harnesses**: `scripts/hypothesis_sweep.sh`,
  `decode_eta_test.sh`, `night_run.sh`, `depth_temp_sweep.sh`. Each runs a sim, then `flow_mix` +
  the diagnostics block, appending to a `summary.md`.

## 4. Root-cause mechanism (the core finding)

**Price movement reduces to one scalar.** An order moves the price only if it is *marketable* —
crosses the spread and executes. Depth is decoded in
`WorldAgent.py` → `_postprocess_generated_TRADES`:

```python
depth = round(z_depth * σ_d + μ_d)      # σ_d = 2.6777, μ_d = 1.3847  (normalization_stats.json)
```

Marketable ⇔ `depth < 0`. So the whole question is: *what fraction of the model's depth output
falls below the boundary?*

**The marketable tail is a sampling artifact, not a learned feature.** Preprocessing clamps depth
to ≥ 0, so the model is trained on a depth distribution piled at 0 (a spike at the best quote).
Marketable orders appear *only* when sampling noise spills the output below 0.

**Therefore deterministic few-step sampling freezes:** it contracts the depth output's variance
(pulls it onto the conditional mean), the spike stays at 0, no orders cross, passive orders
accumulate into immovable volume walls, and the mid-price never ticks. As a Gaussian
approximation, marketable fraction = Φ(−0.517/s):

| depth-output std s | marketable fraction | regime |
|---|---|---|
| 1.0 (well-calibrated) | 30% | ≈ Gaussian prior |
| 0.5 | 15% | — |
| 0.3 (few-step deterministic) | 4% | frozen |

**DDPM works because its noise is *learned and load-bearing*.** `WorldAgent`/`gaussian_diffusion.py`
→ `ddpm_single_step` uses the IDDPM variance
`log_var = f·log β_t + (1−f)·log β̃_t`, `f = ½(v_θ+1)` — a learned per-dimension interpolation up
to the upper bound. That keeps depth variance near its trained value (~24% marketable) → executions
→ movement.

**DDIM(η=1) ≠ DDPM.** In `gaussian_diffusion.py` → `ddim_single_step`, the injected-noise scale is
the *fixed lower bound* `η·√β̃_t`. Wherever the model wants more spread (`f>0`), DDPM injects
strictly more noise than DDIM can at any η ≤ 1, so DDIM under-samples the depth tail.

**Many-step deterministic DDIM diverges** because the probability-flow ODE is stiff for our
under-trained score field (observed: conditioning z reaching +265σ, price crashing 34%).

## 5. Hypotheses tested, and outcomes

All fixes are flag-gated (default off = original behaviour), wired via
`ABIDES/config/world_agent_sim.py` into the `WorldAgent` constructor.

| Hypothesis | Mechanism / code | Outcome |
|---|---|---|
| Type-decode geometry biases order type | `--type-decode prior`: Bayes prior-correct the nearest-anchor decode in `_postprocess_generated_TRADES` | Fixes the η=1 market-order blow-up and drift, but does **not** unfreeze (type ≠ depth) |
| Conditioning time channel frozen during generation | `--fix-time` (feed generated inter-arrivals back into `placed_orders`) | No effect on freeze; drove the time channel OOD (self-referential) |
| Cancels silently dropped when no exact match | `--fix-cancel-bind` (bind to nearest same-side order) | No effect on freeze |
| LOB padding inverted vs training (0 vs LOBSTER sentinels) | `--fix-lob-pad` in `_z_score_orderbook` | No effect (book rarely < 10 levels in this window) |
| Type-2 partial cancels in conditioning (absent in training) | `--drop-type2-cond` in `placeOrder` | Marginal |
| Order-flow imbalance builds walls | Order **TTL** (auto-cancel own resting orders) | Caused an O(n²) mass-cancel hang at the replay→gen boundary; fixed then **reverted** — didn't fix the freeze |
| OOD price conditioning (seed at −5.7σ vs training mean) | Mid-price-relative normalization | **Reverted** — DDPM works fine despite the same conditioning, so not the freeze cause |
| Freeze = sampler collapse; stochasticity needed | η sweep (0→1), sampler comparison | **Confirmed**: stochastic samplers move, deterministic freeze |
| The marketable tail is set in the **early** high-noise steps | `HYBRID_DDPM_PP` (stochastic DDPM head + deterministic DPM-Solver++ tail) in `gaussian_diffusion.py` | **Confirmed**: DDPM-head unfreezes (113 mids); the inverse (PP head + DDPM tail) stays frozen (5 mids). But it over-injects → drift |
| Restore the tail with a depth-channel temperature | `--depth-temp κ` scales `z_depth` in the decode | **Cliff, not a dial** (see §7) |
| Checkpoint conditional calibration drives it | run same samplers on ckpt 0.681 vs 0.719 | **Confirmed and important** (see §8) |

## 6. Robustness fixes found along the way (kept)

- **RecursionError** in `ABIDES/util/order/Order.py` → `generateOrderId` under high order volume →
  made iterative.
- **Quadratic slowdown**: history lists (`placed_orders`, `lob_snapshots`) grew unbounded → capped.
- **Unbounded resample loop** in `_generate_order` (each iteration is a full `sample()`) could hang
  on a degenerate checkpoint → capped at `max_attempts=100` with a retry wakeup; new
  `resample_exhausted` counter.
- **Config crash**: `GUIDANCE_SCALE` defaulted to `None` (dict pre-populated with None) →
  `None * tensor` on the first training batch → added explicit default + None-guard.

## 7. Key experiment: the depth-temperature sweep (`depth_temp_sweep.sh`)

Testing whether a decode-time knob can dial the marketable fraction to the target. It cannot — it
is a **switch, not a dial**:

| κ (`--depth-temp`) | marketable % | depth-0 pile % | unique mids | exec % | price drift |
|---|---|---|---|---|---|
| 1.0 (off) | 2 | 72 | 6 | 4.8 | \$0.02 (frozen) |
| 1.5 | 55 | 4 | 193 | 31.9 | \$2.02 |
| 2.0 | 54 | 2 | 164 | 35.7 | \$4.48 |
| 3.0 | 52 | 1 | 191 | 41.5 | \$7.85 |
| *real / DDPM target* | *~3 / ~24* | — | *69 / 23* | *7 / 18* | *~\$0.4 / \$0.1* |

**Why the cliff:** at κ=1.0 the collapsed depth output is a **spike** (72% at depth 0, z ∈
[−0.70, −0.33]) sitting right against the marketable boundary. κ *multiplies* z, so it slides the
whole spike; once κ≈1.4 its centre crosses and *all* of it flips marketable at once. You cannot peel
a realistic 3% tail off a spike with a scalar — the realistic depth profile has to come from the
model. (Additive noise on `z_depth` would widen rather than translate, but the real fix is model
calibration.)

## 8. Key finding: deterministic samplers are hypersensitive to checkpoint calibration

Same samplers, two checkpoints (`night_run.sh` Phase 1):

| checkpoint | DDIM_10 η=0 | DDPM_100 |
|---|---|---|
| val 0.681 (epoch 1, tighter conditional) | **frozen** — 6 mids, 0% marketable | works (~24% marketable) |
| val 0.719 (epoch 4, broader conditional) | **explodes** — 109 mids, 36% marketable, +\$1.3 drift | works (~24% marketable) |

The two checkpoints behave *oppositely* under deterministic DDIM (too tight → freeze; too broad →
explode), yet **DDPM lands ~24% marketable and works on both** — its injected noise dominates and
washes out the checkpoint's miscalibration. Note validation loss went the *wrong* way (0.681 →
0.719 got worse as training progressed) while behaviour flipped, so **val loss is not a reliable
selector** — checkpoints must be judged by simulation-level metrics.

## 9. Cross-check against the paper and released data

- The TRADES paper (arXiv:2502.07071) uses **DDPM, 100 steps** for its main results and released
  dataset. It presents single-step DDIM (η=0) as a disclosed **100× acceleration with significant
  degradation** (predictive score 1.213 → 3.146 on Tesla).
- We pulled their released **`TRADES-LOB`** dataset (`data/TRADES-LOB` in their repo) and analysed
  their INTC 2015-01-30 file: flow mix 46/36/17, depth std 2.02 with only 3.1% marketable spread
  across levels, lean book, 46 distinct mids — i.e. **DDPM-quality structured generation**, matching
  our own DDPM. We now use it as the **realism benchmark**.
- Nuance we verified in code (`ddim_single_step`, `nsteps=1` schedule): a single-step DDIM
  initialises at pure noise (t=99) and takes one step at the near-clean t=1 schedule entry
  (ᾱ₁=0.9983), so the model contributes **~1.7%** of the output and the rest is the input Gaussian
  noise. It is a **near-marginal regime**, not a graceful accelerator — the disclosed score collapse
  is the fingerprint of exactly that. (Framing point, not an accusation: the number *is* disclosed.)

Detailed maths in `analysis/deterministic_sampler_findings.md`.

## 10. Current state of play

**Established:**
1. DDPM works and replicates the authors' released benchmark; the base model is sound.
2. Deterministic fast samplers fail in closed loop via **depth-channel variance collapse**;
   stochasticity is load-bearing and must be injected **early** (high-noise steps).
3. A **decode-time scalar (depth-temp) cannot** reconstruct the realistic tail from a collapsed spike.
4. Deterministic samplers are **hypersensitive to checkpoint calibration**; DDPM is robust; val loss
   does not track simulation realism.

**In progress:** retraining a checkpoint with `CONDITIONAL_DROPOUT = 0.1`
(`configuration.py`) — this both aims for a better-calibrated conditional *and* activates the model's
existing **classifier-free guidance** path (`gaussian_diffusion.py` → `_forward_with_guidance`,
`--guidance-scale`), which is currently unused infrastructure.

**Open questions:**
- Does a well-calibrated (properly trained) checkpoint let moderate-step deterministic or hybrid
  sampling work, matching the DDPM benchmark at a fraction of the cost?
- Can classifier-free guidance tune realism / market impact directly?
- Best hybrid split / minimal stochastic-head steps for a stable fast sampler.

## 11. Findings / contributions

1. **Faithful accelerated sampling in closed-loop LOB simulation requires stochasticity** —
   deterministic ODE solvers collapse the marketable-order tail that drives price discovery. This
   reframes the "acceleration" problem: the target is a fast *stochastic* (or stochastic-head hybrid)
   sampler, not a deterministic ODE solver.
2. **Deterministic samplers are hypersensitive to conditional-variance calibration** where
   stochastic samplers are robust — with the practical corollary that validation loss is not a valid
   model-selection criterion for this task.
3. **Single-step DDIM is a degenerate near-marginal regime**, not a point on a graceful
   speed/fidelity curve — quantified from first principles.

## 12. Next steps

- Benchmark the retrained (dropout) checkpoint against `TRADES-LOB` via `flow_mix` + depth histogram.
- If better-calibrated: re-run the moderate-step DDIM / `HYBRID_DDPM_PP` / DPM-Solver++ comparison to
  find the fastest sampler that holds the benchmark.
- Sweep classifier-free guidance scale for realism/impact control.
- Then proceed to the downstream goal: the accelerated simulator as a responsive environment for
  execution-agent training (benchmarked vs Almgren–Chriss, VWAP, POV).
