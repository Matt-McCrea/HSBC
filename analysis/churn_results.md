# CHURN sweep results — 2026-07-10/11 overnight run

Full results: `open_loop_sweep/20260710_230511/` (open-loop, 7 checkpoints × 6 samplers) and
`eval_new_ckpt/20260710_231106/` (closed-loop, 5 checkpoints × 5 samplers). Real window: INTC
2015-01-30, 09:30–10:00 (real unique mids = 69).

## Open-loop (per-order, real conditioning, no feedback) — sanity check

| ckpt | market% (real 2.8%) | depth-0% (real 58.6%) |
|---|---|---|
| 0.656–0.681 (trained, DDIM/DPM++) | 1.8–3.2% ✓ | 63–74% (slightly high) |
| DDPM (all ckpts) | 7.6–8.4% (~3× real) | 12–15% (way under-concentrated) |
| 2.869 (untrained) | 33–36% | 1–2% |

**Per-order, the deterministic samplers on trained checkpoints match real marginals *better* than
DDPM does** — DDPM over-produces market-type orders 3x and massively over-spreads depth. This
confirms the freeze is not a per-order defect of the deterministic sampler; fed healthy history it
behaves realistically. The freeze is a closed-loop / feedback phenomenon: once a deterministic
market goes quiet it has no noise to restart itself and spirals into the frozen attractor, whereas
DDPM's over-activity keeps it out of that basin. (Caveat below on what "DDPM works" means.)

## Closed-loop — unique mid-prices / negative-depth-fraction (real = 69 mids)

| checkpoint | DDPM_100 | DDIM10 η=0 | HYBRID_DDPM_PP | CHURN s3/κ0.3 | CHURN s4/κ0.5 |
|---|---|---|---|---|---|
| 0.656 (best val) | 22 / 23.8% | 9 / 0.2% | 12 / 0.1% | 11 / 0.2% | 20 / 0.4% |
| 0.671 | 23 / 23.5% | 10 / 0.2% | 32 / 0.1% | 12 / 1.8% | **100 / 41.6%** |
| 0.681 | 23 / 23.7% | 6 / 2.2% | **113 / 35.8%** | **158 / 35.9%** | **145 / 39.9%** |
| 0.719 | 24 / 24.5% | **109 / 36.3%** | 151 / 44.2% | 131 / 44.6% | 83 / 46.2% |
| 2.869 (untrained) | 32 / 24.3% | 96 / 46.2% | 84 / 46.5% | 80 / 47.0% | 56 / 47.0% |

(`neg%` = share of the depth histogram's negative bucket, i.e. genuinely marketable orders —
`DIAG depth_pre_drop` in each run's log.)

## Finding 1 — CHURN does not dial; it is a checkpoint-dependent cliff, same shape as depth-temp

Down every column, behaviour is binary: **frozen** (neg% <2%, single digits to low-teens mids) or
**diverging/overshooting** (neg% 35–47%, 80–160+ mids, far past real's 69). No checkpoint × churn
setting lands near real's numbers (69 mids, ~24%-ish neg matching DDPM). Turning the churn strength
up (κ 0.3→0.5) does not smoothly increase realism — on checkpoint 0.671 it is the entire difference
between frozen (12 mids) and diverging (100 mids), with nothing in between. This is the same
qualitative shape as the `--depth-temp` cliff found in `new_ckpt.md`: a spike sitting on a decision
boundary that either doesn't move or slides across all at once, regardless of which knob pushes it.

**CHURN, HYBRID_DDPM_PP, classifier-free guidance, and depth-temp are now four independent
sampler-side interventions that all hit the identical frozen/diverging wall.** None gives a stable,
controllable middle ground. This is a genuine negative result for the "no-retrain" hypothesis.

## Finding 2 — checkpoint 0.719 is intrinsically unstable, independent of churn

`0.719 / DDIM10 η=0` (no churn, no hybrid) already overshoots — 109 mids, 36.3% neg — with **zero**
extra stochasticity added. This reconfirms the earlier checkpoint-calibration-sensitivity finding
from prior sessions; it is not a new effect of tonight's samplers.

## Finding 3 — DDPM's neg% is suspiciously flat across checkpoint quality

DDPM's negative-depth fraction: 23.8%, 23.5%, 23.7%, 24.5%, 24.3% across checkpoints 0.656 → 2.869
(untrained). An **untrained** checkpoint produces almost the same marketable-order rate under DDPM
as the best-trained one. This suggests DDPM's "healthy" behaviour is driven substantially by its
own injected per-step noise rather than by anything the model specifically learned — a caveat on
using DDPM as the realism benchmark, not just on the fast samplers' failure.

## Conclusion

Sampler-side fixes (guidance scale, depth-temp, sampler hybrids, and now churn) have been
exhausted as an avenue and all converge on the same result: a binary frozen/diverging outcome with
no stable realistic middle. This points at the training data itself, not the sampling procedure —
directly motivating the depth-unclamp retrain (`scripts/unclamp_retrain.sh`), now launched. See
`analysis/churn_and_unclamp_plan.md` for the mechanism and `analysis/trades_explainer.html` §4.4(B)
for the plain-language version.
