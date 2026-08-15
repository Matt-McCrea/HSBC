# Evaluation methodology — handoff for the rewrite

*2026-08-09. Everything needed to rewrite the evaluation-methodology section: the metrics actually
used, what each can and cannot show, and the numbers already in hand for the six diagnosis claims.*

**Read the second half first if you want the numbers.** §1–§6 define the metrics; §7 maps each of
your six questions to concrete figures and their source file.

---

## The short version

Five metric families were used, and one metric of the original paper was **cited but not
reproduced**. Being explicit about that last point is worth a sentence in the section — it is the
kind of gap an examiner will otherwise assume you overlooked.

| # | Metric | Purpose | Computed from |
|---|---|---|---|
| 1 | LOB-Bench (6 Wasserstein) | headline distributional fidelity | `processed_orders.csv` + real replay |
| 2 | Price-path diagnostics | did the market move, and sensibly | `processed_orders.csv` |
| 3 | Flow composition | limit/cancel/execute balance | `processed_orders.csv` (TYPE) |
| 4 | Decode-stage diagnostics | **the variance-contraction evidence** | run **stdout** (not the CSV) |
| 5 | Variance ratio VR(q) | persistence vs jitter | `processed_orders.csv` |
| — | Predictive score (MAE) | their headline metric | **cited only, not run** |

---

## 1. LOB-Bench

Nagy et al. (2025). Six metrics, each a Wasserstein-1 distance between generated and real
distributions; lower is more realistic.

`spread`, `log_inter_arrival`, `orderbook_imbalance`, `orderflow_imbalance`, `limit_depth_ask`,
`cancel_depth_ask`

Reported as a per-metric table plus a grand mean. Two conventions to state explicitly, because both
change the number:

- **Inter-arrival is excluded when scoring against TRADES's released output.** Their released CSVs
  carry only 0.1 s timestamp resolution, which inflates that one metric. It is a file-format
  artefact, not model behaviour. Say this in the text — otherwise it looks like a dropped metric.
- **A five-metric mean and a six-metric mean are not comparable.** Our full-month figure of
  **0.468** includes inter-arrival; excluding it, the same month is **~0.437**. Quoting 0.468 against
  their 0.798 compares six metrics against five.

Runs **locally only** (jax needs Python ≥3.10; the remote box cannot run it). Scoring therefore
happens after pulling CSVs, never on the GPU machine.

**What it cannot do.** It scores *distributional* fidelity, not the validity of the price path. Our
single-step run that diverges \$1.90 from real scores **0.673**, better than the released model's
**0.855** on the same day. That blind spot is the reason metrics 2 and 5 exist and are reported
alongside it throughout.

## 2. Price-path diagnostics

From `evaluation/quantitative_eval/flow_mix.py` and the `movemetric` helper embedded in the sweep
scripts. All computed post-warm-up (09:45 cutoff for a 09:30 start; 15 min for long runs).

| Quantity | Definition | Why |
|---|---|---|
| `uniq_mid` | count of distinct mid-prices, rounded to 0.001 | the freeze detector — a frozen book has 3–6 |
| `ret1s_std` | std of log returns on 1-second bars, in bp | volatility, comparable to real |
| `mid_range_tk` | (max − min) mid, in ticks | how far the price actually travelled |
| `bid_size_1` / `ask_size_1` | mean and max touch size | wall formation |

`uniq_mid` is the primary freeze diagnostic and the one to lead with. **Its known weakness**: it
cannot distinguish movement that accumulates from movement that cancels — which is what motivated
metric 5. Do not use it alone to argue a market is realistic.

## 3. Flow composition

Percentage split of processed events into `LIMIT_ORDER` / `ORDER_CANCELLED` / `ORDER_EXECUTED`,
from the TYPE column post-warm-up. Real INTC reference (2015-01-30, 09:45–10:00): **49.2 / 43.8 /
7.0**.

Note the definitional caveat you already use: "executed" covers both market orders and marketable
limit orders, which the released TRADES format does not separate.

## 4. Decode-stage diagnostics — the variance-contraction evidence

**These are printed to run stdout and are not in `processed_orders.csv`.** A CSV records what was
*placed*; the collapse argument depends on what the model *decoded before drops*. That gap is the
evidence. Emitted by `ABIDES/agent/WorldAgent.py:222-274`:

| DIAG line | Contents | What it shows |
|---|---|---|
| `decoded_pre_drop` | limit / cancel / market counts | the **type collapse** — market share before any filtering |
| `placed` | same, after drops | how much survives |
| `drops` | `size_range`, `limit_out_of_depth`, `cancel_no_best` | **size collapse** — negative sizes discarded |
| `execution_channels` | `A_market_order`, `B_crossing_limit` | separates true market orders from marketable limits; **B is the only channel depth fixes affect** |
| `depth_pre_drop` | histogram: neg / 0 / 1-2 / 3-5 / 6+ | the **depth collapse** — `neg` is the marketable tail |
| `size_pre_drop[type]` | histogram + mean/std per type | size distribution before filtering |
| `resample` | `total_batches`, `extra_batches` | decode waste from invalid samples |
| `cond_z[channel]` | min / mean / max z-scores | conditioning drifting out of training support |

**`B_crossing_limit` is the single number the whole depth argument turns on** — it counts decoded
limit orders whose price crossed the book. Under a frozen sampler it is 0.

⚠️ **The original logs are gone.** Nothing under any local `*.txt`/`*.log` still contains a `DIAG`
line. They survive only where they were pasted into markdown during the runs — see §7 for which file
holds which. Any *new* diagnostic run must have its stdout captured deliberately.

## 5. Variance ratio

```
VR(q) = Var(q-period return) / (q × Var(1-period return))
```

≈1 random walk · <1 mean-reverting · >1 trending. Reported at q = 10 s, 60 s, 300 s on 1-second bars
after a 15-minute warm-up.

Added late, to separate *how much* the price moves from *whether the movement persists* — a
distinction `uniq_mid` and `ret1s_std` cannot make. **Estimation caveat to state**: VR(300 s) on a
30-minute session rests on ~5 non-overlapping blocks and is unreliable (real itself reads 1.62 there
against its true ~1.02). Trust the two-hour numbers; treat 30-minute VR(300 s) as indicative.

## 6. Predictive score (MAE) — cited, not reproduced

TRADES's headline metric. Their reported Intel figures: **0.307** at 100 steps, **0.486** at one
step, which we cite as their result.

**We did not compute it.** Infrastructure exists (`evaluation/quantitative_eval/predictive_score.py`,
`predictive_lstm.py`, `predictive_batch.py`) and was smoke-tested, but was never run on a protocol
comparable to theirs. Reasons worth stating: the LSTM configuration (lookback, hidden size, layers)
is not fully specified in the paper, and their published figures are TSLA while this work is INTC,
so the numbers would not have been directly comparable in any case.

Write this as a deliberate scoping decision with its justification, not as an omission. One or two
sentences, in the methodology, is the right weight.

## Supporting instrumentation (not fidelity metrics, but reported)

- **Stylised-fact battery** — `evaluation/stylized_custom/paper_style_stylized_facts.py`. Convention
  fixed late and worth stating: **minute bars for all lag-based panels** (matching the paper's
  0–30 minute axis), **1-second bars for the mid-price trace only**.
- **Timing** — `timing_summary.txt` per run: orders generated, ms/order (augmenter + network split),
  total NN wall-clock, throughput. ⚠️ **Not comparable across machines** — the same 100-step sampler
  measures 251 ms/order on the old box and 142 ms/order on the RTX 4070. Any speed claim needs a
  same-hardware pair.
- **KL / JS divergence** — `evaluation/quantitative_eval/kl_divergence.py`, on SIZE / PRICE / TYPE
  marginals. Used in the early July sweeps only, superseded by LOB-Bench. Mention only if you cite a
  July number; otherwise omit.

---

## 7. Evidence map — your six questions

Windows are 09:30–10:00 on INTC unless stated; real reference is the matched replay. As you say,
day and checkpoint are illustrative here rather than load-bearing.

### Q1 — Multi-step DDIM freezes the mid-price
**Source:** `hypothesis_results.md` §`A_DDIM_10_eta0` (ckpt 0.681, 2015-01-30, DDIM-10 η=0, no fixes)

| | Real | DDIM-10 η=0 |
|---|---|---|
| unique mid-prices | **69** | **6** |
| mid range | 33.605–33.975 (37 tk) | 33.935–33.960 (**2.5 tk**) |
| flow limit/cancel/exec | 49.2 / 43.8 / 7.0 | 59.4 / 35.8 / **4.8** |
| bid_size_1 mean | 3,899 | **115,158** |
| ask_size_1 mean | 2,117 | **103,431** |

The touch-size rows are the strongest single illustration: liquidity piles up ~30× real because
nothing consumes it.

### Q2 — Variance contraction across type, depth and size
**Source:** same block, plus `analysis/cancel_sweep_table.md` for the depth histogram.

From `A_DDIM_10_eta0`:
- `decoded_pre_drop: limit=16214 cancel=10107 market=729` → market **2.7%** of decodes
- `drops: size_range=6527` → **24%** of decoded orders discarded for invalid size
- `resample: total_batches=27050 extra_batches=6653`
- `cond_z[price]: min=-3.81 mean=-3.36 max=-3.12` → conditioning pinned ~3.4σ below the training
  mean for the whole run, which is the observation that motivates `PRICE_REANCHOR`

⚠️ **`depth_pre_drop` and `size_pre_drop` are absent from this block** — those DIAG lines were added
after this sweep ran. For the depth histogram use `analysis/cancel_sweep_table.md`, e.g.
`depth_pre_drop: neg=1694 0=6515 1-2=3941 3-5=1905 6+=1138` with
`execution_channels: A_market_order=0 B_crossing_limit=1412`. Different run, but illustrative —
state which run each figure comes from.

### Q3 — Restoring η does not restore DDPM
**Source:** `hypothesis_results.md` §`A_DDIM_10_eta1`

| | η=0 | η=1 | Real |
|---|---|---|---|
| decoded market | 729 (2.7%) | **5741 (24.1%)** | ~3% |
| executed % | 4.8 | **15.7** | 7.0 |
| unique mids | 6 | 136 | 69 |
| mid range | 33.935–33.960 | 33.935–**35.160** | 33.605–33.975 |
| `drops: size_range` | 6,527 | **14,795** | — |

**The argument in one line:** movement returns, but via the *type* channel flooding to market orders
(2.7% → 24.1%), not via depth-driven price discovery — and the price drifts **+\$1.22** out of the
real envelope. Pair with the IDDPM learned-variance argument you already have.

### Q4 — Noise placement, not noise quantity
**Source:** `analysis/churn_results.md`, closed-loop table (ckpt 0.681 row); cross-checked against
`hypothesis_results.md` §`A_HYBRID_PP_DDPM_10`

**The 113/5 pair is verified and is a same-checkpoint comparison** — ckpt 0.681, INTC 2015-01-30,
09:30–10:00, real = 69 unique mids:

| sampler | unique mids | neg-depth % |
|---|---|---|
| DDPM-100 | 23 | 23.7 |
| DDIM-10 η=0 (control) | **6** | 2.2 |
| **HYBRID_DDPM_PP** (DDPM head, ODE tail) | **113** | 35.8 |
| **HYBRID_PP_DDPM** (ODE head, DDPM tail) | **5** | — |

Two qualifications that make the claim stronger rather than weaker:

- **Checkpoint-dependent.** The same DDPM-head hybrid gives only **12** unique mids on ckpt `0.656`
  (and 14 in the `HYBRID_DDPM_PP_8+2` variant of `new_ckpt.md`). So placement decides the outcome on
  0.681 but not universally — consistent with Finding 1 of `churn_results.md`.
- **113 overshoots real (69) rather than matching it**, at 35.8% negative depth against DDPM's ~24%.
  The defensible claim is therefore *where the noise goes determines which failure mode you get* —
  ODE head freezes, DDPM head diverges, neither lands on real — not that the DDPM head works.

⚠️ Do not source this from `hypothesis_results.md` alone: `A_DDIM_20_eta1` in that file reads **112**
unique mids and is easily mistaken for the 113. It is a twenty-step η=1 run, not a hybrid, and it
runs away to \$35.74.

### Q5 — Adjusting DDIM configuration is insufficient
**Step count** — `ETA_SUMMARY.md` has the ladder on ckpt 0.681: `DDIM10_eta0` (1410 s),
`DDIM100_eta0` (5905 s), plus `DDPM100_default` (3819 s) and `DDPM100_prior` (3824 s).
`hypothesis_results.md` adds `A_DDIM_20_eta1`.

**Scalar temperature** — ⚠️ **Gap.** The κ sweep (1 → 2%, 1.5 → 55%, 2 → 54%, 3 → 52%) is quoted in
your skeleton but is **not** in any retained markdown I can find; `analysis/reshape_results.md` and
`appendix_lobbench_and_refutations.md` describe the mechanism ("temp slides the whole atom") without
the figures. Find the source or re-run a three-point sweep — it is cheap and the switch-not-dial
contrast is worth having properly sourced.

### Q6 — Validation loss does not select a usable sampler
**Sources:** `ETA_SUMMARY.md` (ckpt 0.681 and 0.719 sections); `analysis/churn_results.md`
Finding 2 and Finding 3

The comparison: `0.681` froze under DDIM η=0 while working under DDPM; `0.719` — the *better*
validation loss — exploded under the same sampler. Loss moved the wrong way against sampler
behaviour. `churn_results.md` Finding 2 states 0.719 is intrinsically unstable independent of churn,
which is the cleanest phrasing of the point.

This is what makes the behavioural search of §5.4 principled rather than improvised, so give it the
forward link.

---

## 8. What to pull from the remote

**Most of what you need is already local.** The retained markdown carries the numbers; the CSVs are
only needed if you want to regenerate figures.

Worth pulling for illustrative figures — the vanilla (no `_tdprior_sr_dn0.3` suffix) runs, which are
the like-for-like freeze comparison:

```
world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.69_     # vanilla DDIM-10, alive
world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.724     # vanilla DDIM-10, alive
world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_1_val_ema=0.69_      # vanilla DDIM-1
world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_1_val_ema=0.724      # vanilla DDIM-1
world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763      # the 73-min collapse
world_agent_INTC_2015-01-30_09-47-00_30_DDIM_0.0_10_val_ema=0.682     # vanilla, 0130
market_replay_INTC_2015-01-29_10-00-00_30                             # real reference
market_replay_INTC_2015-01-30_10-00-00_30
market_replay_INTC_2015-01-29_12-00-00_30
```

```bash
cd /cs/student/project_msc/2025/cf/mmccrea/HSBC/HSBC/ABIDES/log
tar czf ~/eval_method_csvs.tgz \
  world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.69_ \
  world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_10_val_ema=0.724 \
  world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_1_val_ema=0.69_ \
  world_agent_INTC_2015-01-29_10-00-00_30_DDIM_0.0_1_val_ema=0.724 \
  world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_1_val_ema=0.763 \
  world_agent_INTC_2015-01-30_09-47-00_30_DDIM_0.0_10_val_ema=0.682 \
  market_replay_INTC_2015-01-29_10-00-00_30 \
  market_replay_INTC_2015-01-30_10-00-00_30 \
  market_replay_INTC_2015-01-29_12-00-00_30
```

Note these vanilla runs are on the **current** checkpoints, where the freeze does not occur — they
support the checkpoint-dependence subsection (§5.2.7 in the results plan), not Q1. The original
0.681 freeze CSV does not appear in the remote listing; its numbers survive in
`hypothesis_results.md`, which is sufficient for a table.

**Not worth pulling:** the `rl_execution_*` directories (a different workstream) and the
`_tdprior_sr_dn0.3` runs (post-fix configurations, already covered elsewhere).

---

## 9. The three gaps, ranked

1. **Depth-temperature kappa sweep figures** — quoted in the results skeleton, source not found. Cheap
   to re-run (three 30-min cells) and the switch-not-dial contrast deserves a real citation.
2. ~~Head/tail 113-vs-5 pair~~ — **RESOLVED 2026-08-10.** Verified in `churn_results.md`: same
   checkpoint (0.681), same day, 113 vs 5. See Q4 above for the two qualifications.
3. **`depth_pre_drop` / `size_pre_drop` for the original freeze run** — not emitted at the time.
   Substitute the `cancel_sweep_table.md` histogram and say which run it came from, or re-run
   DDIM-10 η=0 on 0.681 with stdout captured.

None blocks the rewrite. Gaps 1 and 2 need resolving before those two specific subsections are
finalised.
