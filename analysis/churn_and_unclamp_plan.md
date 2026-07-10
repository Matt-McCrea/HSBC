# Churn sampling (tonight) + depth unclamp (tomorrow)

Two experiments attacking the frozen-mid-price from opposite ends: a **sampler-side** fix that needs
no retrain (tonight), and the **root-cause training fix** (tomorrow).

## Where we are

The dropout retrain + classifier-free guidance did **not** fix the collapse (`new_ckpt.md`): every
deterministic / near-deterministic config still lands at 9–17 unique mids (real: 69) and builds
40k–140k-share volume walls (real ~2–4k). Depth-0 concentration is 78–84%, if anything worse than the
old checkpoint's ~72%.

CFG was never the right tool: guidance reweights toward the conditional **mean**
(`eps_uncond + w·(eps_cond − eps_uncond)`); it adds **no variance**. The freeze is a *variance*
collapse — few-step ODE integration drives depth → its conditional mean → 0 → no marketable orders →
frozen mid. You cannot sharpen your way out of a variance collapse, which is exactly why the g-sweep
moved the needle only slightly and non-monotonically. Book closed on guidance-scale as a freeze fix.

The one lever we *proved* works is **early-step stochasticity** (`HYBRID_DDPM_PP`: a stochastic head
unfroze the old checkpoint). Both experiments below build on that finding.

---

## Tonight — CHURN sampler (no retrain)

**Idea.** A DPM-Solver++ backbone (accurate, few-step) with **EDM-style stochastic churn on the first
`churn_steps` (high-noise) steps** — the steps our `HYBRID_DDPM_PP` result showed set the
marketable-order diversity. Each churned step: renoise `x_t` to a slightly noisier level, one denoiser
eval, one 1st-order step; the clean tail runs standard 2nd-order DPM++. It's a **continuous dial**
between deterministic DPM++ (κ=0) and a stochastic head, concentrated where entropy matters, at
few-step cost. Unlike DDIM-η it rides an accurate 2nd-order backbone and can push κ>0 to actively
*re-inflate* the collapsed depth variance.

Renoise (EDM restart in the VP / ᾱ parametrisation), κ = `churn_strength`:
```
ᾱ̂ = ᾱ_t·(1−κ);   x̂ = √(ᾱ̂/ᾱ_t)·x_t + √(1−ᾱ̂/ᾱ_t)·z,  z~N(0,I)
```
signal-preserving, adds variance. Then denoise from ᾱ̂ down to ᾱ_{t−1}.

**Knobs** (`--churn-steps N --churn-strength κ`, κ clamped to [0, 0.9]):
- `churn_steps` — how many early steps get renoised (default 3 of 10).
- `churn_strength κ` — how much entropy per churned step (default 0.3). κ=0 ⇒ pure DPM++.

**Code:** `models/diffusers/gaussian_diffusion.py::churn_sample` (+ `-type CHURN` dispatch, CLI
`--churn-steps/--churn-strength` in `world_agent_sim.py` and `open_loop_eval.py`).

**Run:**
```sh
# open-loop (minutes): included as two cells in the sweep
bash scripts/open_loop_sweep.sh
# closed-loop: two CHURN cells now in Stage 1 of eval_new_checkpoint.sh
```

**What to watch:** `depth_pre_drop` negative bucket > 0 and unique mids climbing toward real, WITHOUT
the DDPM-style overshoot into drift. The hoped-for result: DDPM-quality realism at DDIM speed — which
is the thesis's accelerated-sampler contribution rather than another diagnosis. If even κ=0.5 stays
frozen, it confirms the entropy the model needs simply isn't in the weights (→ the unclamp fix).

---

## Tonight — the overnight sensitivity sweep

Two instruments, cheap-first:

```sh
# 1. CHEAP breadth (~minutes/cell): 7 checkpoints × 6 samplers, open-loop (no ABIDES). Prints a
#    market%/depth-0% table. This is the sensitivity map.
bash scripts/open_loop_sweep.sh

# 2. EXPENSIVE depth (~90 min/ckpt): the Stage-1 battery (DDPM, DDIM, HYBRID_DDPM_PP, 2×CHURN) on a
#    curated bracket best→mid→old→exploder→untrained.
bash scripts/eval_new_checkpoint.sh \
  --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv \
  --ids "0.656 0.671 0.681 0.719 2.869"
```

Including `2.869` (epoch 0, near-untrained) is a sharp test: an untrained model outputs near-marginal
noise ≈ balanced flow, so it may look *less* collapsed under a deterministic sampler than the trained
checkpoints — strong evidence the collapse is about the sharpness of a *trained* conditional mean, not
a bug. The key question the matrix answers: does the freeze depend on checkpoint quality at all, or is
it sampler-intrinsic across every checkpoint?

---

## Tomorrow — depth unclamp (root cause, needs retrain)

**The bug.** `utils_data.py:324` clamps `depth ≥ 0` in preprocessing, and `WorldAgent.py:796/802`
does the same to the simulation conditioning. A marketable, spread-crossing order has `depth < 0` — so
it is **never a training target**. The model can only ever emit aggression as *sampling variance*,
never as a learned signal. That is the mechanistic root of why killing the variance (few-step
deterministic) freezes the market.

**The fix (already implemented, flag-gated OFF by default).** `constants.UNCLAMP_DEPTH` reads env var
`UNCLAMP_DEPTH`; when set, both the preprocessing clamp and the conditioning clamp keep signed depth.
Default unset ⇒ original clamped behaviour, so nothing changes until you opt in.
- `constants.py` — `UNCLAMP_DEPTH = os.environ.get("UNCLAMP_DEPTH","0")=="1"`
- `utils/utils_data.py:324` — `depths[j] = depth if cst.UNCLAMP_DEPTH else max(depth,0)`
- `ABIDES/agent/WorldAgent.py:796/802` — `if depth < 0 and not cst.UNCLAMP_DEPTH: depth = 0`

Unclamping shifts μ_depth down and widens σ_depth, so it **requires reprocessing** (regenerates the
`.npy` + normalization stats). The retrain shell handles backup + reprocess + train:

```sh
# ensure configuration.py has IS_DATA_PREPROCESSED = False first
bash scripts/unclamp_retrain.sh          # exports UNCLAMP_DEPTH=1, backs up clamped data, reprocess+train
# ...then evaluate with the SAME env var so conditioning matches training:
export UNCLAMP_DEPTH=1
bash scripts/open_loop_sweep.sh      --ids "<new-val-loss-id>"
bash scripts/eval_new_checkpoint.sh  --real <replay csv> --ids "<new-val-loss-id>"
bash scripts/unclamp_retrain.sh --restore   # roll back to the clamped baseline data if needed
```

> ⚠️ `UNCLAMP_DEPTH=1` must be exported at **simulation** time too, not just training — otherwise the
> conditioning depth is clamped while the model expects signed depth (train/sim mismatch).

**Expected outcome / success criteria.** After the unclamp retrain, `depth_pre_drop` should show a
genuine **negative bucket** (marketable orders the model learned, not sampled by accident), and the
*deterministic* samplers (DDIM / DPM++ / CHURN) should move the mid and drop the volume walls — because
aggression is now a signal they can reproduce without stochasticity. If deterministic samplers move
after unclamping where they froze before, that's the decisive confirmation of the root cause and the
cleanest result for the thesis: *the freeze was a data-clamp artifact, not a diffusion-sampler
limitation.*
