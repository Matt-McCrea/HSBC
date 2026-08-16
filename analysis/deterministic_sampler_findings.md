# Deterministic samplers in closed-loop LOB simulation: why they collapse, and what single-step DDIM actually is

*Finding writeup — INTC 2015-01-30, TRADES diffusion LOB simulator, closed-loop world-agent.
Cross-checked against the TRADES paper (arXiv:2502.07071) and the authors' released
`TRADES-LOB` dataset.*

## Executive summary

In closed-loop limit-order-book (LOB) simulation, the DDPM sampler reproduces a moving,
realistic market, but every *deterministic* accelerated sampler we tested fails in one of two
ways: few-step DDIM (η=0) **freezes** the mid-price, and full-step deterministic DDIM
**diverges**. We show the failure reduces to a single scalar mechanism — **variance contraction
in the order-depth channel** — and that it is intrinsic to deterministic sampling on our
(under-trained) checkpoint, not evidence of a broken model: DDPM recovers the healthy behaviour
on the *same* weights.

We also examined the authors' own sampling choices. The paper's main results and released dataset
use **DDPM with 100 steps**; single-step DDIM appears only as a disclosed lossy acceleration
(predictive score 1.213 → 3.146 on Tesla). So there is no hidden result. However, the *framing* of
single-step DDIM as an "acceleration technique" is mechanically misleading: at one step the model
contributes ≈1.7% of the output and the sampler emits essentially the Gaussian prior. It is not a
reduced-fidelity version of conditional generation — it is the near-absence of it, and the
disclosed score collapse is the symptom.

---

## 1. From model output to a price-moving order

The model emits, per order, a vector in a normalized (z-scored) feature space. The channel that
decides whether an order can move the price is the **depth** (ticks from the best quote), decoded
([WorldAgent.py](../ABIDES/agent/WorldAgent.py) `_postprocess_generated_TRADES`) as

$$\text{depth} = \operatorname{round}\!\big(z_d \cdot \sigma_d + \mu_d\big), \qquad \mu_d = 1.3847,\ \sigma_d = 2.6777 .$$

An order is **marketable** — it crosses the spread, executes immediately, and moves the mid-price
— exactly when `depth < 0`, i.e.

$$z_d < \frac{0-\mu_d}{\sigma_d} = -0.517 .$$

Passive orders (`depth ≥ 0`) only rest and accumulate. **Price movement in this simulator is
driven by the fraction of generated orders whose depth output falls below −0.517σ.** That single
quantity separates a live market from a frozen one.

## 2. The marketable tail is a *sampling artifact*, not a learned feature

Preprocessing clamps depth to be non-negative, and the real next-event depth histogram shows **0%
negative depth**. The model therefore never sees negative-depth targets and learns a depth
distribution piled at 0 with a floor. Marketable orders arise only when **stochastic sampling
variance spills the depth output below that floor.** This one fact drives the entire
DDPM-vs-deterministic split.

## 3. The mechanism: variance contraction in one channel

At every reverse step both samplers compute the same posterior-mean estimate

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}} \approx \mathbb{E}[x_0 \mid x_t,\text{cond}] .$$

Treating the decoded depth as approximately Gaussian with std $s$ (near the floor), the marketable
fraction is $\Phi(-0.517/s)$:

| depth-output std $s$ | marketable fraction $\Phi(-0.517/s)$ | observed regime |
|---|---|---|
| $1.0$ (well-calibrated) | **30.3%** | ≈ the Gaussian prior (see §7) |
| $0.5$ | 15.1% | — |
| $0.3$ (few-step deterministic) | **4.2%** | ≈ our DDIM η=0 → **frozen** |

Deterministic few-step sampling contracts $s$ (the coarse ODE discretisation pulls the output onto
the floored conditional mean), so almost nothing reaches −0.517σ. DDPM keeps $s\approx$ its trained
value by re-injecting noise each step. **The freeze is variance contraction in the depth channel.**

## 4. Why DDIM(η=1) is *not* equivalent to DDPM

The two updates share the mean and differ only in the injected-noise scale $s$:

- **DDIM(η):** $s = \eta\sqrt{\tilde\beta_t}$, the *fixed* posterior-variance lower bound
  $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\big(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\big)$.
- **DDPM (IDDPM, [gaussian_diffusion.py](../models/diffusers/gaussian_diffusion.py) `ddpm_single_step`):**
  $s = \sqrt{\exp(f\log\beta_t + (1-f)\log\tilde\beta_t)}$, $f=\tfrac12(v_\theta+1)$ — a **learned**
  per-dimension interpolation between the lower bound $\tilde\beta_t$ and upper bound $\beta_t$.

Wherever the model learned $f>0$ (more spread than the floor — what a heavy-tailed channel like
depth needs), DDPM injects strictly more noise than DDIM can at any η≤1. Empirically our DDIM(η=1)
reaches "movement" not through depth but by the *type* channel wandering into the oversized MARKET
region — a buy-biased market-order flood that **drifts +11%**, not price discovery.

## 5. Why "just use more steps" also fails (for our checkpoint)

Full-step deterministic DDIM (`nsteps=100`, η=0) **diverges**: 70% of decodes become market orders,
conditioning reaches +265σ, price crashes 34%. The probability-flow ODE is stiff for our
under-trained score field, so fine deterministic integration accumulates error. Deterministic
sampling has no good operating point on this checkpoint: few steps freeze, many steps diverge.

## 6. What single-step DDIM actually computes

The repository's default simulation command ([README.md](../README.md)) passes neither `-type` nor
`-nsteps`, so it runs the argparse defaults **`DDIM`, `nsteps=1`**
([world_agent_sim.py](../ABIDES/config/world_agent_sim.py)). Trace it exactly:

1. The sample is initialised at $t=99$ as essentially pure noise:
   $x_t = \sqrt{\bar\alpha_{99}}\cdot 0 + \sqrt{1-\bar\alpha_{99}}\,\epsilon \approx \epsilon$
   ($\bar\alpha_{99}=2.4\times10^{-6}$).
2. The single step uses the nearly-clean $t{=}1$ entry, $\bar\alpha_1=0.9983$. In
   `ddim_single_step`, the x̂₀ line
   `pred_x0 = (x_t - sqrt_one_minus_alpha·noise_t)/alpha**0.5` has noise weight
   $\sqrt{1-\bar\alpha_1}/\sqrt{\bar\alpha_1} = \mathbf{0.042}$ and $x_t$-weight $1.001$.
3. The returned value `x_prev = √(ᾱ₀)·pred_x0 + √(1−ᾱ₀)·noise_t` (η=0, $\bar\alpha_0=0.9994$) adds
   part of the noise back, partially cancelling; the **net** output is

$$x_{\text{out}} = 1.0006\,x_t \;-\; \mathbf{0.017}\,\hat\varepsilon .$$

So end-to-end the model contributes **~1.7%** of the output; ~98% is the initialisation noise.
(There is also a train/inference mismatch: the network is queried at label $t{=}1$ while its input
is fully noised, so even that 1.7% is not meaningful denoising.) The output keeps **unit variance
in every channel**, so $z_d\sim\mathcal N(0,1)$ and the marketable fraction is $\Phi(-0.517)=30.3\%$
— an alive market produced by decoding the standard-normal prior, not by the diffusion model.

## 7. What the paper actually does (and the honest critique)

From the paper (arXiv:2502.07071) and the released data:

- **Main results and the `TRADES-LOB` dataset use DDPM, 100 steps.** Our analysis of their released
  INTC 2015-01-30 file confirms it: flow mix 46/36/17 (matching our DDPM 47/34/19), depth std 2.02
  with only 3.1% marketable spread smoothly across levels, a lean book (bid_size median 1,020), and
  46 distinct mid-prices over a \$0.23 range. This is structured conditional generation — **not** the
  ~30%-marketable signature of the §6 single-step regime. Their headline results are genuine.
- **Single-step DDIM is disclosed as a lossy acceleration**, with the predictive score worsening
  from 1.213 to 3.146 (Tesla) for a "100× efficiency" gain. Nothing is hidden.

**The defensible critique is one of characterisation, not concealment.** Presenting single-step
DDIM as an "acceleration technique" places it on the same fidelity–speed axis as multi-step DDPM,
implying a graceful trade-off. Mechanically (§6) it is not: at one step the sampler bypasses the
model (~1.7% contribution) and emits the Gaussian prior. The reported score collapse
(1.213 → 3.146) is exactly the fingerprint of that bypass. So the honest reading is: *the number is
disclosed, but framing single-step DDIM as reduced-fidelity acceleration understates that it is a
qualitatively different, near-marginal regime rather than a diminished version of the same
conditional generation.*

## 8. Empirical evidence (INTC, checkpoint val_ema=0.681)

| sampler | market decode | executions | unique mids | mid drift | verdict |
|---|---|---|---|---|---|
| real target | 2.8% | 7.0% | 69 | \$0.37 | — |
| **their DDPM (released `TRADES-LOB`)** | — | 17.3% | 46 | \$0.23 | **benchmark: structured, moving** |
| our DDPM 100 | 7.9% | 18.8% | 23 | \$0.11 | works (matches theirs) |
| our DDIM 10, η=0 | 2.7% | 4.8% | **6** | \$0.03 | frozen (depth collapse) |
| our DDIM 10, η=1 | 24.1% | 15.7% | 136 | **\$1.23** | moves via market-order drift |
| our DDIM 100, η=0 | 70.8% | 48.2% | 49 | **−\$11.5** | diverges (stiff ODE) |
| DDIM 1 (repo default) | — | — | (moves) | — | Gaussian prior (§6) |

## 9. The trained model is not the cause; ours is under-trained

DDPM recovers the marketable tail on the exact checkpoint that DDIM freezes on, so the tail exists
in the learned distribution. Our deterministic failures are (a) **aggressive acceleration** (few
steps collapse the depth channel; many steps hit a stiff ODE) on (b) an **under-trained checkpoint**
(epoch 1–4, validation loss plateaued). The authors' converged model has a genuinely wide
conditional depth distribution (their released std ≈ 2.0), which a proper sampler preserves. The
gap between "their DDIM/DDPM works" and "ours struggles" is **model quality × sampler
aggressiveness**, not a trick.

## 10. Caveats

- We could not run the authors' released checkpoint at the time of the first analysis (their Drive
  folder appeared empty); the README links it and it is worth re-attempting, which would allow a
  direct same-model comparison.
- Validation loss does not track simulation quality here (a val=2.317 checkpoint reportedly
  simulates fine; our val=0.681 freezes under multi-step DDIM). Judge checkpoints by
  simulation-level metrics — flow mix, depth histogram, unique-mid count — against the released
  `TRADES-LOB` benchmark, not by val loss.

## 11. Implication

For faithful **closed-loop** LOB simulation, the marketable-order tail that drives price discovery
is produced by sampling variance breaching a training-imposed depth floor, not by the conditional
mean. Deterministic ODE samplers cannot reproduce it on our checkpoint: few steps collapse it, many
steps diverge. **Genuine accelerated sampling here requires stochasticity** (DDPM, an SDE solver, or
a stochastic-head hybrid), or an explicit variance-restoring correction in the depth channel. And
single-step DDIM should be reported as a degenerate near-marginal regime, not as a point on a
graceful acceleration curve — the disclosed score collapse is precisely why.
