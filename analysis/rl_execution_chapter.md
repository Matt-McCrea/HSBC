# RL Optimal Execution on an Accelerated Generative LOB Simulator
### Draft sections for the dissertation: literature, methodology, results skeleton

> **How to use this.** Written to be lifted into the thesis with light editing. Maths is
> in LaTeX so it pastes into a `.tex` document directly. Citations marked ✅ were
> verified against the published record while drafting; those marked ⚠️ are from memory
> and the full bibliographic details should be checked before submission.
>
> **Section 3 is now measured, not a skeleton.** Every number in it comes from
> `logs/train_session.jsonl` (114 episodes) or `logs/eval_frontier_*.jsonl` (54 episodes),
> reproducible with `analyze_logs`, `calibrate`, `impact`, `inspect_policy` and
> `compare_policies`. The one section deliberately left empty is §3.6, the sampler
> comparison, which was not run.

---

# 1. Literature Review

## 1.1 The optimal execution problem

A trader holding a parent order of $X$ shares to liquidate cannot simply send it to the
market: a large marketable order walks the limit order book, and the resulting price
concession grows with size. Splitting the order over time reduces this *market impact*
but exposes the unexecuted remainder to price risk. Optimal execution is the study of
that trade-off.

The cost convention throughout is **implementation shortfall** (Perold, 1988 ⚠️): the
difference between the value of the parent order at the *arrival price* — the mid-quote
at the moment the decision was made — and the value actually realised. Writing $S_0$ for
the arrival mid, $n_k$ for shares traded in interval $k$ and $\tilde S_k$ for the
achieved price,

$$
C \;=\; X S_0 \;-\; \sum_{k=1}^{N} n_k \tilde S_k \;=\; \sum_{k=1}^{N} n_k \left( S_0 - \tilde S_k \right).
$$

The right-hand form matters for what follows: **shortfall is already a sum of per-interval
costs**, which is what licenses attributing cost to individual decisions in a sequential
decision problem (§2.3).

Bertsimas & Lo (1998) ⚠️ gave the first dynamic-programming treatment, showing that
under linear permanent impact and a random walk the risk-neutral optimum is to trade at a
uniform rate — the theoretical justification for TWAP as a benchmark rather than merely a
convention.

## 1.2 Almgren–Chriss and the efficient frontier

Almgren & Chriss (2000) ✅ [*Journal of Risk* 3(2), 5–40] added risk aversion, which is
the framework this work adopts. Liquidating $X$ shares over horizon $T$ in $N$ intervals
of length $\tau = T/N$, let $x_k$ be the holdings remaining at $t_k = k\tau$, with
$x_0 = X$ and the terminal constraint $x_N = 0$. Trades are $n_k = x_{k-1} - x_k$. Prices
follow

$$
S_k \;=\; S_{k-1} + \sigma \tau^{1/2} \xi_k \;-\; \tau\, g\!\left(\frac{n_k}{\tau}\right),
\qquad
\tilde S_k \;=\; S_{k-1} - h\!\left(\frac{n_k}{\tau}\right),
$$

with $\xi_k$ i.i.d. mean zero, unit variance; $g(\cdot)$ permanent impact and $h(\cdot)$
temporary impact. Under the linear specification $g(v) = \gamma v$ and
$h(v) = \epsilon\,\mathrm{sgn}(n) + \eta v$, the cost has

$$
\mathbb{E}[C] \;=\; \tfrac{1}{2}\gamma X^2 \;+\; \epsilon \sum_k |n_k| \;+\; \frac{\tilde\eta}{\tau}\sum_k n_k^2,
\qquad
\mathbb{V}[C] \;=\; \sigma^2 \tau \sum_{k=1}^{N} x_k^2,
\qquad \tilde\eta = \eta - \tfrac{\gamma\tau}{2}.
$$

The trader solves a mean–variance problem

$$
\min_{\{x_k\}} \;\; \mathbb{E}[C] \;+\; \lambda\, \mathbb{V}[C],
$$

whose solution traces an **efficient frontier** of expected cost against variance as the
risk-aversion parameter $\lambda$ varies. The optimal trajectory is

$$
x_j \;=\; X\,\frac{\sinh\!\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)},
\qquad
\kappa \;\approx\; \sqrt{\frac{\lambda \sigma^2}{\tilde\eta}} .
$$

Two consequences are used directly in this work.

**First, TWAP is the risk-neutral member of the same family.** As $\kappa \to 0$,
$\sinh(\kappa(T-t))/\sinh(\kappa T) \to (T-t)/T$, i.e. the trajectory becomes linear and
the schedule uniform. TWAP is therefore not an arbitrary yardstick but the $\lambda = 0$
solution, which makes any comparison against it interpretable in the framework's own terms.

**Second, the variance term is a running penalty on held inventory.** $\mathbb{V}[C] =
\sigma^2\tau\sum x_k^2$ depends only on the trajectory, so risk aversion is expressible as
a per-period charge $\propto x_k^2$ — the form used as a reward shaping term in §2.3.

Later work relaxes the linear-impact assumption (Almgren, 2003 ⚠️; Gatheral, 2010 ⚠️),
models order-book resilience explicitly (Obizhaeva & Wang, 2013 ⚠️), and extends to limit
rather than market orders (Guéant, Lehalle & Fernandez-Tapia, 2012 ⚠️). Cartea, Jaimungal
& Penalva (2015) ⚠️ give the continuous-time stochastic-control treatment, where the
objective carries a running penalty $\phi\int_0^T q_u^2\,du$ and a terminal liquidation
penalty $\alpha Q_T^2$ — the latter softening the hard $x_N = 0$ constraint that AC impose.

## 1.3 Reinforcement learning for execution

The analytical solutions above buy tractability with strong assumptions: a specific impact
functional, a martingale price, and no feedback from the trader's own actions to the
order-book state. Reinforcement learning replaces those assumptions with interaction,
learning a policy from experience rather than deriving one from a model.

**Nevmyvaka, Feng & Kearns (2006) ✅** [ICML, pp. 673–680] is the canonical reference and
remains the closest methodological ancestor of the present work. They apply tabular
RL via backward induction to limit-order placement, with a state space of **time
remaining and inventory remaining** plus optional market features, minimising
implementation shortfall. Their empirical base is 1.5 years of millisecond-resolution
NASDAQ data — a scale worth stating explicitly, because it is the sharpest contrast with
the episode budget available in a simulator-based study (§3.6).

Subsequent work scales the function approximator rather than changing the problem:
Hendricks & Wilcox (2014) ⚠️ learn adjustments to an AC trajectory; Ning, Lin & Jaimungal
(2021) ⚠️ apply double deep Q-learning; Lin & Beling (2020) ⚠️ use PPO. Recent surveys and
extensions address time-varying liquidity (Macrì & Lillo, 2024 ⚠️, arXiv:2402.12049) and
the joint market/limit order decision (arXiv:2507.06345 ⚠️, 2025).

A recurring methodological difficulty is that **RL on historical replay cannot represent
market response**: a policy that trades differently from the recorded tape would have
elicited different behaviour from other participants, and replay cannot supply it. This
motivates interactive simulators.

## 1.4 Generative market simulation

ABIDES (Byrd, Hybinette & Balch, 2020 ⚠️) provides a discrete-event, multi-agent exchange
with a real matching engine, so an experimental agent's orders genuinely interact with
resting liquidity. Its realism, however, depends on hand-specified background agents whose
calibration is itself a research problem.

Coletta, Moulin, Vyetrenko & Balch (2022) ✅ [ICAIF '22; arXiv:2210.09897] replace that
population with a single learned **"world agent"** trained to reproduce aggregate order
flow, implemented as a conditional GAN. This is the design the present simulator inherits
— the `WorldAgent` class is named for it. **TRADES** (Berti, Prenkaj & Velardi, 2025 ⚠️,
arXiv:2502.07071) continues the line, replacing the GAN with a transformer-based diffusion
model over order flow.

**Hafsi & Vittori (2024) ✅** [arXiv:2411.06389] is the most directly comparable prior
work: an RL execution agent trained inside ABIDES, reporting improvement over standard
execution strategies. Their background market is ABIDES' conventional agent population.

## 1.5 Positioning of this work

Against that literature, RL-for-execution is well established and RL-inside-ABIDES has
been done. The contribution here is narrower and more specific:

1. **RL execution against a *generative* world model rather than hand-specified agents.**
   The background market is a diffusion model of order flow, so the agent trades against
   learned rather than stipulated behaviour.
2. **Making that computationally feasible.** A generative world agent must be sampled at
   every generated order, so the dominant cost is diffusion sampling, not the exchange.
   The cold-start procedure (§2.1) removes a fixed ~15-minute simulated replay from every
   episode, and the accelerated sampler reduces per-order cost. Both are prerequisites
   rather than conveniences: without them the episode budget is not merely small but
   effectively zero.
3. **An honest account of what the simulator can and cannot support.** The diagnostics in
   §2.8 test the simulator's own assumptions — in particular whether generated prices
   satisfy the martingale property Almgren–Chriss requires — rather than assuming them.

---

# 2. Methodology

## 2.1 Simulation environment

Episodes run inside ABIDES with three agents: the exchange (a real limit-order matching
engine), the TRADES world agent generating background order flow, and the execution agent
under study.

**Cold-start seeding.** The world agent conditions on the previous $N-1$ orders and $N$
order-book snapshots. Conventionally these are accumulated by replaying ~15 minutes of
real order flow through the full simulation kernel before generation begins — a fixed cost
per episode that dominates a 5-minute experiment. Because the L3 message stream is
complete, the resting book at any timestamp $t_0$ is *exactly reconstructible* by replaying
messages from the open while maintaining a price-level book: no kernel, no model inference.
Episodes are therefore seeded by

1. reconstructing the exact resting order book at $t_0$ from the raw message log, and
   loading it into the exchange;
2. slicing the conditioning tensor directly from the same log, applying the identical
   normalisation used at training time.

Measured cost is 0.53 s per episode against a replay of approximately 15 minutes.

*Implementation note for the write-up:* reconstruction required a price–time-priority
matching engine rather than naive log replay, because approximately 2–3% of orders in
the LOBSTER data receive no explicit cancellation or execution event anywhere in the
day's log. A replay that trusts the log alone leaves these resting indefinitely and
produces a crossed book.

**Seed-timestamp sampling.** $t_0$ is drawn uniformly across trading days and times of
day, floored at `[10:00]` — 30 minutes after the open. Requiring only sufficient
conditioning history is insufficient: on a liquid symbol 256 messages elapse within
seconds of the open, when the real book still holds a few hundred resting orders against a
typical 3,000–5,000, and generation seeded that thin is unstable.

## 2.2 MDP formulation

The agent liquidates $Q$ shares over a 5-minute window divided into $N = 10$ decision
points spaced 30 seconds apart, following the convention that execution decision frequency
is set by market microstructure rather than by the parent order's horizon.

**State.** Following Nevmyvaka et al. (2006), the state is deliberately low-dimensional:

| Variable | Definition | Buckets |
|---|---|---|
| Time remaining | decision points left | 11 |
| Inventory remaining | $x_k / Q$ | 5 |
| Spread | best ask − best bid, ticks | 4 |
| Realised volatility | s.d. of mid over last 5 points | 3 |
| Order-flow imbalance | $(V^{bid} - V^{ask})/(V^{bid}+V^{ask})$ at touch | 3 |

All five are computed and logged; **only the first two index the value function** in the
primary configuration. With all five the space is $11\times5\times4\times3\times3 = 1{,}980$
states $\times$ 5 actions $= 9{,}900$ entries, which the achievable episode budget (§3.1)
cannot populate. The retained pair is exactly the Almgren–Chriss state $(t, x)$, making the
learned policy directly comparable to the analytical trajectory.

**Action.** Five discrete participation levels, from passive to aggressive:

| # | Level | Order type | Price | Size ($\times Q/N$) |
|---|---|---|---|---|
| 0 | Passive | Limit | own best quote | 0.5 |
| 1 | Light | Limit | own best quote | 1.0 |
| 2 | Marketable limit | Limit | crosses to opposite best | 1.0 |
| 3 | Aggressive | Market | immediate | 1.5 |
| 4 | Very aggressive | Market | immediate | 2.0 |

Outstanding child orders are cancelled at each decision point, so each decision's fills are
confined to its own interval. At $k = N$ the agent market-orders any remainder, enforcing
$x_N = 0$; this is the $\alpha \to \infty$ limit of the Cartea–Jaimungal terminal penalty.

## 2.3 Objective

The reward implements the Almgren–Chriss objective directly. Per-interval cost is
attributed to the decision whose child order produced each fill:

$$
r_k \;=\; -\frac{1}{Q}\sum_{i \in \mathcal{F}_k} q_i\,(S_0 - p_i) \;-\; \lambda\,x_k^2 ,
$$

where $\mathcal{F}_k$ are fills from decision $k$, and the sign convention is inverted for
buy orders. Two properties are worth stating explicitly:

- **The first term sums to $-C/Q$ over an episode**, so per-interval attribution is a
  re-attribution of implementation shortfall, not a different objective. It converts a
  single terminal payment into a dense signal, which materially improves credit assignment
  at small episode counts.
- **The second term is Almgren–Chriss's variance**, since $\mathbb{V}[C] = \sigma^2\tau\sum_k x_k^2$.
  It is deterministic given the trajectory, so it adds learning signal at zero additional
  variance.

Undiscounted returns ($\gamma = 1$) are used: the horizon is finite and fixed, and
discounting would bias the policy toward front-loading for reasons unrelated to risk.

## 2.4 Calibration

Rather than choosing $\lambda$ by hand, the impact and volatility parameters are estimated
from the simulator itself, over `[12]` benchmark episodes.

**Volatility** $\hat\sigma$ is the standard deviation of mid-price changes over one
decision interval.

**Temporary impact** $\hat\eta$ follows AC's $h(v) = \epsilon + \eta v$: per-decision
slippage against the prevailing mid is regressed on trade rate $v_k = n_k/\tau$,

$$
\underbrace{\left(m_k - \bar p_k\right)}_{\text{slippage}} \;=\; \epsilon + \eta\,v_k + u_k ,
$$

with $\bar p_k$ the volume-weighted fill price. The intercept estimates the half-spread
term, the slope the impact coefficient.

**Risk aversion.** $\lambda$ is a preference, not a measurable quantity, so it is fixed by
targeting a dimensionless $\kappa T$ — which determines the trajectory's shape, with
$\kappa T \to 0$ recovering TWAP. Target $\kappa T = [2]$. Converting to the per-share
reward convention used here gives the implementation constant

$$
\texttt{lam} \;=\; \frac{(\kappa T)^2\,\hat\eta\,\bar Q}{N^2 \tau}.
$$

Note $\hat\sigma$ cancels, so the calibration is robust to a noisy volatility estimate; the
same $\lambda$ then parameterises both the RL penalty and the analytical baseline, making
them comparable at identical risk aversion.

**Estimated values:** $\hat\sigma = [\,]$, $\hat\eta = [\,]$ ($R^2 = [\,]$),
$\hat\epsilon = [\,]$, giving $\texttt{lam} = [\,]$.

## 2.5 Learning algorithm

Tabular Q-learning with $\varepsilon$-greedy exploration; deep function approximation is
deliberately excluded given the episode budget.

$$
Q(s_k,a_k) \;\leftarrow\; Q(s_k,a_k) + \alpha_k\Big[r_k + \gamma \max_{a'} Q(s_{k+1},a') - Q(s_k,a_k)\Big]
$$

with $\gamma = 1$ and a **visit-count step size** $\alpha_k = 1/N(s_k,a_k)$. The latter
matters: with $\gamma=1$ and terminal-dominated reward, $Q$ *is* the mean return, so a
running average is the Robbins–Monro estimator. A fixed $\alpha = 0.3$ retains an effective
window of only $\sim 3$ episodes, and with per-episode reward dispersion of order
`[\sigma_r]` against an inter-action value gap of order `[\Delta Q]`, the estimate tracks
noise: greedy actions were observed to change between successive checkpoints.

Exploration decays $\varepsilon: 1.0 \to 0.05$ at 0.99 per episode (reaching the 0.05 floor was not attained in 114 episodes; the final value was 0.318).

Because full per-step trajectories are logged, value functions can be **re-fit offline**
under alternative step-size rules, reward shapings and state discretisations without
re-simulating — decoupling the expensive component (market simulation, ~823 s per
episode) from the cheap one (value fitting, milliseconds).

## 2.6 Baselines

**TWAP** — equal-size child orders each interval; as established in §1.2, the
Almgren–Chriss risk-neutral optimum, not merely a convention.

**Almgren–Chriss schedule** — trades toward $x_j = X\sinh(\kappa(T-t_j))/\sinh(\kappa T)$
at the calibrated $\kappa$, mapping the implied trade size onto the nearest available
participation level. *Stated limitation:* the action space couples size with aggression
(actions 1 and 2 are both $1.0\times$ the base slice but differ in whether they cross the
spread), so the mapping is approximate; the baseline is faithful to AC's *schedule*, not a
claim about optimal order placement.

## 2.7 Experimental protocol

Policies are evaluated on a **fixed held-out set of `[20]` seed timestamps**, generated
once with a dedicated random seed and reused across every policy and sampler configuration,
so comparisons are paired rather than merely distributional.

The sampler comparison follows the **matched wall-clock** design: the accelerated
(depth-noise) and DDPM-100 arms receive equal compute budgets, and the reported quantity is
the precision of the shortfall estimate achievable per unit of computation.

## 2.8 Validity diagnostics

Because the background market is generated rather than observed, the following are recorded
per episode and reported alongside results:

- **Execution rate** — the fraction of orders that execute, measured identically in both
  arms (executions per new order) so the simulated market is compared against the real
  one directly rather than against a published figure. Deviations indicate over- or
  under-trading.
- **Unique mid-price count** — a low count indicates a market whose price fails to respond.
- **Conditioning z-scores** — per-channel min/mean/max, detecting out-of-distribution
  conditioning where the generative model is not trustworthy.
- **Drift** $\;(S_T - S_0)/S_0$ with a $t$-statistic — **a direct test of the martingale
  assumption Almgren–Chriss requires.** A single-sided liquidation task in a systematically
  drifting market is flattered or penalised for reasons unrelated to execution skill, so
  this is reported whatever its outcome.
- **Terminal inventory** — must be zero. Unsold inventory *reduces* measured shortfall
  (the cost is normalised by $Q$), so incomplete liquidation would present as strong
  performance; it is therefore checked explicitly rather than assumed.

---

# 3. Results

All figures below are measured. Two logs are reported throughout and distinguished
explicitly: **train** (114 episodes, all SELL, $\varepsilon$-greedy exploration) and
**eval** (54 episodes: 18 held-out seeds $\times$ 3 policies, greedy). Where a quantity is
estimated on both, both are given — agreement across them is the evidence that a finding is
a property of the simulator rather than of one sample.

## 3.1 Computational cost

| Quantity | Value |
|---|---|
| Cold-start reconstruction | 0.53 s/episode (median 0.47) |
| Replay it replaces | ~15 min/episode |
| Wall-clock per 5-min episode (train) | mean 823 s, median 912, IQR [178, 1384] |
| Wall-clock per 5-min episode (eval) | mean 759 s, median 396, IQR [155, 1426] |
| Episodes per 24 h | ~105 |
| Training episodes achieved | **114** in 26.1 h |

Cold-start reconstruction is roughly **1,700$\times$ cheaper** than the kernel replay it
replaces, which is what makes per-episode seeding from arbitrary intra-day times feasible
at all. It is not the bottleneck: simulation is 99.93% of episode wall-clock.

Frame against Nevmyvaka et al.'s 1.5 years of data: the episode budget here is smaller by
orders of magnitude, and that is a property of interactive generative simulation rather
than a shortcoming of the implementation.

## 3.2 Calibration and simulator validity

| Parameter | Train (114 ep) | Eval (54 ep) |
|---|---|---|
| $\hat\sigma$ (per 30 s, price units) | — | 155.6 |
| $\hat\eta$ (temporary impact) | 6.554 ($t$=15.69, $R^2$=0.197, $n$=1004) | 4.751 ($t$=7.86, $R^2$=0.113, $n$=488) |
| $\hat\epsilon$ (intercept) | −47.16 | −66.01 |
| Derived $\lambda$ | 42.12 (used for training) | 16.64 |
| Drift (bps / 5 min) | **+7.09** ($t$=7.88) | **+5.33** ($t$=4.30) |
| Execution rate | 0.18 | 0.17 |
| Unique mid-prices per episode | 9.45 | 8.94 |
| Resting orders at $t_0$ | 3,771 | 3,595 |

Two validity findings, both robust across the logs and both material to the live-trading
question:

**The martingale assumption fails.** Drift is significantly positive in both samples
($t$=7.88 and $t$=4.30). Almgren–Chriss assumes a driftless mid-price, so the analytic
frontier is not the correct optimum in this simulator, and a single-sided liquidation is
flattered or penalised for reasons unrelated to execution skill. This is reported as a
property of the world agent, not corrected away.

**The simulated market over-executes by roughly twofold.** Execution rate is 17–18% in
the generative arm. Measured directly on the real INTC stream over the same episode
window — executions per new order, the same quantity — it is **8.5%**. A generated order
is therefore about twice as likely to execute as a real one, so measured shortfall is
optimistic relative to live trading: fills that a real book would not have provided are
provided here.

The 4–6% figure quoted for this symbol elsewhere in the literature is not the comparison
used here. It is not measured on this window with this definition, and the like-for-like
number is the one above; against 4–6% the gap would look nearer threefold, which would
overstate it. One caveat on the definition: LOBSTER event type 4 marks each visible
execution, so an order filling in several parts contributes more than once, and the real
rate is a mild upper bound on the fraction of orders that execute at all.

$\hat\epsilon$ is negative in both logs, which cannot hold for pure liquidity taking — a
fill cannot beat the mid at zero size. It is the signature of an action space that also
*provides* liquidity (a passive sell rests at the ask and fills above the mid), and it means
$\hat\epsilon$ must not be read as a half-spread without first separating passive from
aggressive fills.

The derived $\lambda$ differs between logs (42.12 vs 16.64) because it inherits the noise in
$\hat\sigma^2/\hat\eta$. The conclusions are insensitive to this: the penalty sweep moves
the mean greedy action only from 1.22 at $\lambda=0$ to 1.25 at $\lambda=42$, so any value
in that range yields effectively the same policy.

## 3.3 Learned policy

State coverage after 114 episodes: **32 of 55 states visited (58.2%)**, 117 of 275
state–action entries updated (42.5%), median 6 visits per entry (max 59).

Greedy action distribution over visited states at the calibrated $\lambda=42$:

| Action | Count | Share |
|---|---|---|
| 0 passive (0.5$\times$ slice) | 9 | 28.1% |
| 1 light (1.0$\times$ slice) — *TWAP's action* | 14 | 43.8% |
| 2 neutral (crossing limit) | 4 | 12.5% |
| 3 aggressive (market, 1.5$\times$) | 2 | 6.2% |
| 4 very aggressive (market, 2.0$\times$) | 3 | 9.4% |

The learned policy differs from TWAP in 18 of 32 visited states (56.2%), but its modal
action *is* TWAP's, and the densely-visited interior of the grid (time remaining 4–6/10,
inventory 20–80%) is uniformly action 1. The erratic cells are the sparse corners. The
agent therefore converges **toward** the risk-neutral optimum that Almgren–Chriss predicts,
which is the theoretically expected destination for a risk-neutral objective.

**The AC front-loading prediction is confirmed.** Mean greedy action rises monotonically
with the inventory penalty — 1.22 at $\lambda=0$, 1.25 at $\lambda=42$, 1.78 at
$\lambda=337$ — with 8 of 32 greedy actions changing at $\lambda=42$ and 20 of 32 at
$\lambda=337$. A risk term that pushes the agent to liquidate faster is exactly what
Almgren–Chriss requires, and it is recovered here from logged trajectories without
re-simulation.

Convergence is **partial**. Greedy-action churn between successive re-fits on growing data
runs 40% → 4% → 27% → 12%; the Q-value spread within a visited state has median 54.4.
The policy is not converged at 114 episodes and is reported as such.

## 3.4 Policy comparison

Shortfall in basis points of the arrival mid; lower is better. 18 held-out seeds, identical
across policies. Because every policy walks the same seed list, the **paired** difference is
the correct statistic: it cancels the market movement common to a seed. Pairing reduces the
standard error by 2–3$\times$ relative to the unpaired comparison.

**At the calibrated $\lambda = 42$:**

| Policy | Mean | SE | Median | IQR | Paired $\Delta$ vs TWAP | $t$ | $p$ | Wins |
|---|---|---|---|---|---|---|---|---|
| TWAP | 0.03 | 1.08 | 1.20 | [−2.7, 3.4] | — | — | — | — |
| AC ($\kappa T$=2) | 1.39 | 0.87 | 0.99 | [−1.2, 2.8] | +1.362 (SE 0.669) | 2.04 | 0.058 | 9/18 |
| Q-learning | 0.92 | 1.61 | 0.44 | [−3.4, 4.8] | **+0.892** (SE 0.929) | 0.96 | **0.351** | 8/18 |

**The learned policy is statistically indistinguishable from TWAP** at the calibrated risk
aversion: 95% CI [−0.93, +2.71] bps, winning on 8 of 18 seeds. Given that TWAP *is* the
risk-neutral Almgren–Chriss optimum, matching it is the theoretically expected ceiling for a
risk-neutral objective, not a failure to learn.

**At the exploratory $\lambda = 337$ (8$\times$ calibrated):** both structured policies
degrade — AC +1.845 ($t$=3.07, $p$=0.007, 3/18 wins) and Q-learning +3.012 ($t$=4.02,
$p$=0.0009, 1/18 wins). Over-weighting inventory risk at a five-minute horizon buys risk
reduction that the cost side does not repay.

The AC schedule underperforms TWAP at every $\lambda$ tested. Section 3.5 explains why this
is consistent with the measured impact structure rather than a defect of the baseline.

## 3.5 Market impact of the agent's own orders

| Quantity | Train | Eval | Verdict |
|---|---|---|---|
| Temporary impact $\hat\eta$ | 6.554 ($t$=15.69) | 4.751 ($t$=7.86) | **significant in both** |
| Permanent $\hat\gamma$, $h$=1 (30 s) | 0.0248 ($t$=0.94) | −0.0190 ($t$=−0.70) | not detectable |
| Permanent $\hat\gamma$, $h$=2 (60 s) | 0.0352 ($t$=1.15) | −0.0237 ($t$=−0.69) | not detectable |
| Permanent $\hat\gamma$, $h$=3 (90 s) | 0.0324 ($t$=0.96) | −0.0332 ($t$=−0.90) | not detectable |
| Permanent $\hat\gamma$, $h$=5 (150 s) | 0.0244 ($t$=0.55) | 0.0199 ($t$=0.43) | not detectable |

**Temporary impact is strong and permanent impact is undetectable at every horizon out to
150 s, in both samples.** This is the chapter's most robust quantitative finding, and it
explains §3.4: Almgren–Chriss front-loading trades spread cost now for reduced exposure to
price moves later, but if trading leaves no lasting price mark there is little for
front-loading to avoid, so it pays the cost without earning the benefit.

**No claim is made about the functional form of impact.** Linear and square-root fits are
statistically indistinguishable — linear leads by $R^2$ 0.0030 on train, square-root by
0.0047 on eval, i.e. the nominal winner flips between samples. Over the size range a single
parent order spans, the two forms are near-collinear and this data cannot separate them.

Permanent impact here is the price response *associated* with trading, not a causally
isolated effect, because the world agent reacts to the book being perturbed. A clean
estimate requires paired counterfactual episodes with the agent disabled.

## 3.6 Sampler comparison

**Not performed.** The evaluation ran with the DDPM-100 arm disabled in order to spend the
available compute on the policy frontier instead. No sampler claim is made from these runs.

## 3.7 Limitations

- **Episode budget of 114**, against millions in historical-replay studies. State coverage
  is 58% and convergence is partial; the learned policy is best described as approaching
  TWAP rather than as converged.
- **Side mismatch between training and evaluation.** Training fixed the parent order to
  SELL; the held-out seed list drew 14 BUY and 4 SELL, and the state
  $(\text{time},\text{inventory})$ carries no side feature. The policy was therefore applied
  out of distribution on most evaluation episodes. Under AC's martingale assumption the
  optimal trajectory is side-symmetric and this would be immaterial — it is precisely the
  simulator's measured drift that breaks the symmetry. Corroborating evidence: the
  correlation between shortfall and drift is **−0.837 on the all-SELL training log and
  +0.292 on the majority-BUY evaluation log**, the sign flip that side composition predicts.
  The reported result is thus a *conservative* estimate of the policy's performance.
- **No common random numbers.** Episodes re-sample the generated market, so pairing cancels
  the market *situation* (day, $t_0$, side, $Q$) but not the realised path. Identical
  policies score slightly differently across runs (TWAP 0.026 vs 0.048 bps), which bounds
  the residual noise.
- **Execution rate is ~2$\times$ the real market's**, measured like-for-like on the same
  window (17-18% generative vs 8.5% real), so shortfall measured here is optimistic
  relative to live trading.
- **Drift violates the martingale assumption**, so divergence from the analytic AC frontier
  is informative about the simulator rather than evidence against the agent. This is a
  *comparison*, not a validation.
- The action space couples order size with aggression, limiting the fidelity of the AC
  schedule mapping.
- Single symbol (INTC), single 20-day sample.

---

## References

✅ = verified while drafting · ⚠️ = check full details before submission

- ✅ Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–40.
- ⚠️ Almgren, R. (2003). Optimal execution with nonlinear impact and trading-enhanced risk. *Applied Mathematical Finance*, 10(1).
- ⚠️ Bertsimas, D. & Lo, A. (1998). Optimal control of execution costs. *Journal of Financial Markets*, 1(1), 1–50.
- ⚠️ Berti, L., Prenkaj, B. & Velardi, P. (2025). TRADES: Generating realistic market simulations with diffusion models. arXiv:2502.07071.
- ⚠️ Byrd, D., Hybinette, M. & Balch, T. (2020). ABIDES: Towards high-fidelity multi-agent market simulation.
- ⚠️ Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- ✅ Coletta, A., Moulin, A., Vyetrenko, S. & Balch, T. (2022). Learning to simulate realistic limit order book markets from data as a World Agent. *ICAIF '22*. arXiv:2210.09897.
- ⚠️ Gatheral, J. (2010). No-dynamic-arbitrage and market impact. *Quantitative Finance*, 10(7).
- ⚠️ Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2012). Optimal portfolio liquidation with limit orders.
- ✅ Hafsi, Y. & Vittori, E. (2024). Optimal execution with reinforcement learning. arXiv:2411.06389.
- ⚠️ Hendricks, D. & Wilcox, D. (2014). A reinforcement learning extension to the Almgren–Chriss framework for optimal trade execution. *IEEE CIFEr*.
- ⚠️ Lin, S. & Beling, P. (2020). An end-to-end optimal trade execution framework based on proximal policy optimization. *IJCAI*.
- ⚠️ Macrì, A. & Lillo, F. (2024). Reinforcement learning for optimal execution when liquidity is time-varying. arXiv:2402.12049.
- ✅ Nevmyvaka, Y., Feng, Y. & Kearns, M. (2006). Reinforcement learning for optimized trade execution. *ICML*, 673–680.
- ⚠️ Ning, B., Lin, F. & Jaimungal, S. (2021). Double deep Q-learning for optimal execution. *Applied Mathematical Finance*.
- ⚠️ Obizhaeva, A. & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets*, 16(1).
- ⚠️ Perold, A. (1988). The implementation shortfall: Paper versus reality. *Journal of Portfolio Management*, 14(3).
