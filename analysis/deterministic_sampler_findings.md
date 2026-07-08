# Why deterministic samplers collapse the market, and why the reference DDIM result is an artifact

*Finding writeup — INTC 2015, TRADES diffusion LOB simulator, closed-loop world-agent.*

## Executive summary

In closed-loop limit-order-book (LOB) simulation, the DDPM sampler reproduces a moving,
realistic market, but every *deterministic* accelerated sampler we tested fails in one of two
ways: few-step DDIM (η=0) **freezes** the mid-price, and full-step deterministic DDIM
**diverges** (price crash, out-of-distribution conditioning). We show the failure reduces to a
single scalar mechanism — **variance contraction in the order-depth channel** — and that it is
intrinsic to deterministic sampling, not to the trained model. We further show that the
reference implementation's apparent DDIM success is an artifact: its default configuration runs
DDIM with a *single* sampling step, which mathematically passes the initial Gaussian noise
through almost unmodified. Its realism is a property of the marginal noise statistics, not of
conditional generation.

---

## 1. From model output to a price-moving order

The model emits, per order, a vector in a normalized (z-scored) feature space. The channel that
determines whether an order can move the price is the **depth** (ticks from the best quote).
It is decoded ([WorldAgent.py](../ABIDES/agent/WorldAgent.py) `_postprocess_generated_TRADES`)
as

$$\text{depth} = \operatorname{round}\!\big(z_d \cdot \sigma_d + \mu_d\big), \qquad \mu_d = 1.3847,\ \sigma_d = 2.6777 .$$

A generated order is **marketable** — it crosses the spread, executes immediately, and therefore
moves the mid-price — exactly when `depth < 0`, i.e.

$$z_d < \frac{0-\mu_d}{\sigma_d} = -0.517 .$$

Passive orders (`depth ≥ 0`) rest in the book and never move the price on their own; they can
only accumulate. **Price movement in this simulator is driven by the fraction of generated
orders whose depth output falls below −0.517σ.** This is the single quantity that separates a
live market from a frozen one.

## 2. The marketable tail is a *sampling artifact*, not a learned feature

The training pipeline clamps depth to be non-negative (`depth = max(depth, 0)` in preprocessing;
the real next-event depth histogram likewise shows **0% negative depth**). The model therefore
never sees negative-depth targets and learns a depth distribution piled up at 0 with a floor.
Empirically the generated depth distribution confirms this: DDIM(η=0) produces **68.9% of orders
at depth 0 and 0% negative**.

Consequently the marketable (negative-depth) orders that a healthy market needs are **not**
produced by the model's conditional mean. They arise only when **stochastic sampling variance
spills the depth output below the depth-0 floor.** This single observation explains the entire
DDPM-vs-deterministic split.

## 3. The mechanism: variance contraction in one channel

At every reverse step both DDPM and DDIM compute the same posterior-mean estimate of the clean
sample,

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}} \approx \mathbb{E}[x_0 \mid x_t,\text{cond}] .$$

The samplers differ only in how much fresh noise they inject around it. Treating the generated
depth output as approximately Gaussian with standard deviation $s$ (centred near the floor), the
marketable fraction is $\Phi(-0.517/s)$:

| depth-output std $s$ | marketable fraction $\Phi(-0.517/s)$ | observed regime |
|---|---|---|
| $1.0$ (well-calibrated) | **30.3%** | ≈ DDPM (~23% neg) → **moves** |
| $0.5$ | 15.1% | — |
| $0.3$ (few-step deterministic) | **4.2%** | ≈ DDIM η=0 → **frozen** |

Deterministic few-step sampling contracts $s$: the coarse ODE discretisation pulls the output
onto the (floored) conditional mean, so almost nothing reaches the −0.517σ threshold. DDPM keeps
$s \approx 1$ by re-injecting noise each step, so ~1/4 of orders spill negative and execute. **The
freeze is variance contraction in the depth channel; nothing else is required to explain it.**

## 4. Why DDIM(η=1) is *not* equivalent to DDPM

A natural objection: DDIM with η=1 is supposed to recover DDPM. It does not here, and the reason
is the **learned variance**. The two updates share the mean but differ in the noise scale $s$:

- **DDIM(η):** $s = \eta\,\sqrt{\tilde\beta_t}$, where $\tilde\beta_t = \dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\big(1-\tfrac{\bar\alpha_t}{\bar\alpha_{t-1}}\big)$ is the *fixed* DDPM posterior-variance **lower bound**.
- **DDPM (IDDPM parameterisation, [gaussian_diffusion.py](../models/diffusers/gaussian_diffusion.py) `ddpm_single_step`):** $s = \sqrt{\exp(\text{log\_var})}$ with

$$\text{log\_var} = f\,\log\beta_t + (1-f)\,\log\tilde\beta_t, \qquad f = \tfrac{1}{2}(v_\theta+1)\in[0,1],$$

a **learned, per-dimension** interpolation between the lower bound $\tilde\beta_t$ and the upper
bound $\beta_t$, controlled by the network's second output $v_\theta$.

Wherever the model learned $f>0$ — i.e. it wants more spread than the floor, which is exactly what
a heavy-tailed channel like depth needs — **DDPM injects strictly more noise than DDIM can at any
η ≤ 1.** So even η=1 under-samples the depth tail. Empirically, DDIM(η=1) reaches "movement" not
through the depth tail at all but by the *type* channel wandering into the (geometrically
oversized) MARKET decision region — a different route that produces a buy-biased flood of market
orders and a **+11% price drift**, not genuine price discovery.

## 5. Why "just use more steps" also fails

If contraction is the problem, more deterministic steps should help. It does not: full-step
deterministic DDIM (`nsteps=100`, η=0) **diverges** — 70% of decodes become market orders, the
conditioning goes to +265σ, and the price crashes 34% to \$22. The probability-flow ODE is stiff
for this (under-trained) score field, so fine deterministic integration accumulates error rather
than resolving the conditional. Deterministic sampling thus has no good operating point here:
few steps freeze, many steps diverge.

## 6. The reference DDIM result is an artifact of single-step sampling

The reference implementation's documented simulation command
([README.md](../README.md)) passes neither `-type` nor `-nsteps`, so it runs the argparse
defaults: **`DDIM` with `nsteps = 1`** ([world_agent_sim.py](../ABIDES/config/world_agent_sim.py)).
A single-step DDIM is not a denoiser in any meaningful sense. Concretely:

1. The sample is initialised at the highest noise level, $t=99$, as essentially pure noise:
   $x_t = \sqrt{\bar\alpha_{99}}\cdot 0 + \sqrt{1-\bar\alpha_{99}}\,\epsilon \approx \epsilon$, since $\bar\alpha_{99}=2.4\times10^{-6}$.
2. The single reverse step uses the *nearly clean* $t{=}1$ schedule entry, $\bar\alpha_1 = 0.9983$.
   The posterior-mean estimate is then

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_1}\,\hat\varepsilon}{\sqrt{\bar\alpha_1}} = \underbrace{1.001}_{\text{coeff on }x_t}\,x_t \;-\; \underbrace{0.042}_{\text{weight on }\hat\varepsilon}\,\hat\varepsilon .$$

3. The final DDIM update ($\bar\alpha_0 = 0.9994$) leaves this essentially unchanged, so the
   decoded order vector is the **initial Gaussian noise, passed through with a ~4% model
   correction.** (There is also a train/inference mismatch: the network is queried at label
   $t{=}1$ while its input is fully-noised, so even that 4% is not meaningful denoising.)

The output therefore retains **unit variance in every channel**, including depth: $z_d \sim
\mathcal{N}(0,1)$, giving a marketable fraction of $\Phi(-0.517) = 30.3\%$ — a fully alive market.

**This is the crux: the reference DDIM "works" because it is not really running the model.** It
samples the standard normal prior and decodes it. Its realism is inherited from the *marginal*
statistics of the z-scored feature space (which are, by construction, well-behaved and balanced),
and marginal statistics are exactly what most stylized-fact metrics measure. It is not evidence
of good *conditional* generation, and it should not be read as a working accelerated sampler. If
the paper's reported DDIM baseline used this default configuration, its apparent success is an
artifact of single-step near-marginal sampling rather than a property of the diffusion model.

## 7. Empirical evidence (INTC, 09:30–10:00, checkpoint val_ema=0.681)

| sampler | market decode | executions | unique mids | mid drift | verdict |
|---|---|---|---|---|---|
| real target | 2.8% | 7.0% | 69 | \$0.37 | — |
| DDIM 10, η=0 | 2.7% | 4.8% | **6** | \$0.03 | frozen (depth collapse) |
| DDIM 10, η=1 | 24.1% | 15.7% | 136 | **\$1.23** | moves via market-order drift |
| DDIM 10, η=1 + prior decode | 1.7% | 2.8% | **3** | \$0.01 | drift removed → refreezes |
| DDIM 100, η=0 | 70.8% | 48.2% | 49 | **−\$11.5** | diverges (ODE stiff) |
| DDPM 100 | 7.9% | 18.8% | 23 | \$0.11 | **works** |
| DDIM 1 (reference default) | — | — | (moves) | — | near-marginal noise (§6) |

## 8. The trained model is not the cause

Critically, **DDPM recovers the marketable tail on the exact same checkpoint** — ~23% negative
depth, a lean book, and a moving price. The tail therefore *exists* in the learned distribution;
the model is not fundamentally broken. What fails is specifically the *deterministic extraction*
of a tail that requires stochasticity to reach. Under-training makes the score field stiffer
(worsening both the contraction and the divergence), but it is not the root cause: even a
perfectly trained model, sampled deterministically at low step count, would contract the same
scalar channel below the marketable threshold.

## 9. Caveats (for defensible claims)

- We were unable to run the original authors' released checkpoint (their distribution folder is
  empty), so the comparison against "the paper's model" is partly inferential.
- The claim that the reference DDIM baseline used `nsteps=1` is based on the repository's
  documented run command; the paper's internal experiments may have used a different setting. The
  *mechanism* (single-step DDIM = near-marginal sampling) is exact regardless.
- Validation loss does not track simulation quality here (a val=2.317 checkpoint reportedly
  simulates fine; our val=0.681 freezes under multi-step DDIM), so checkpoints must be judged by
  simulation-level metrics — flow mix, depth histogram, unique-mid count — not by val loss.

## 10. Implication

For faithful **closed-loop** LOB simulation, the marketable-order tail that drives price
discovery is produced by sampling variance breaching a training-imposed depth floor, not by the
conditional mean. Deterministic ODE samplers therefore cannot reproduce it: few steps collapse
it, many steps diverge. **Genuine accelerated sampling in this setting requires stochasticity**
(DDPM, an SDE solver, or a stochastic-head hybrid), or an explicit variance-restoring correction
in the depth channel. Reports of deterministic DDIM "working" should be checked for the
single-step near-marginal regime before being taken as evidence of conditional realism.
