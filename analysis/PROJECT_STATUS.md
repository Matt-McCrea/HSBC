# TRADES acceleration — project status & next-session handoff
*Last updated: 2026-08-03. Read this first at the start of a session.*

## TL;DR
The core thesis result is **done and quantified**. A 10-step deterministic sampler (DDIM) with a
decode-time variance fix now produces a live, realistic, boundary-safe market at ~2.7× the
wall-clock speed of 100-step DDPM — and on the standard **LOB-Bench** benchmark it is *more*
realistic than DDPM, not a reduced-fidelity approximation. **Cross-day robustness is also now
resolved**: `val_ema=0.724_epoch=0` is confirmed stable (zero timeouts) across all 20 trading days
in the month, found via a systematic elimination search over a recovered checkpoint family — see the
2026-08-03 handoff below. This is the FINAL model. A scheduled-sampling retrain was attempted to push
its activity level closer to real and was refuted (see below); the pre-retrain checkpoint stands.
Project is now moving to the downstream reinforcement-learning task (execution agent inside the
simulator) on this checkpoint.

## SESSION HANDOFF — 2026-08-03 (READ FIRST when you return)
**FINAL MODEL: `val_ema=0.724_epoch=0`.** Confirmed stable across all 20 trading days in Jan 2015
(zero timeouts, `ret1s_std` inside the real 1.5-2.5bp band everywhere) via `scripts/
adaptive_ckpt_search.sh` — a systematic elimination search that tested 6 checkpoints recovered after
a storage-loss incident, abandoning each at its first per-day timeout. 5 of 6 were eliminated this
way; only `0.724_epoch=0` cleared the full month. Full-month LOB-Bench: grand mean Wasserstein
**0.4683** (`lob_bench_0724_full_month/SUMMARY_mean_wasserstein.csv`; cancel/limit depth strongest,
spread/flow-imbalance weakest). Same-day (2015-01-30) comparison vs last week's tuned single-day best
(0.298) and DDPM (0.507): this checkpoint scores 0.447 — beats DDPM, doesn't beat the old
heavily-day-tuned config, which is the honest trade-off of generalising vs over-fitting to one day.
Full writeup + figures: HSBC progress page (artifact URL in memory), `lob_bench_0724_full_month/`.

**SCHEDULED-SAMPLING RETRAIN: TRIED, REFUTED, KILLED.** Goal was to see if training the model on its
own rollouts could push its activity level (uniq_mid) closer to the real 27-66 range without
reintroducing drift — `0.724_epoch=0` is stable but somewhat under-active (uniq_mid 11-25 vs real).
Two bugs fixed en route: `SAMPLING_TYPE` defaulted to `"DDPM"` in `configuration.py`, making the SS
rollout ~10x more expensive than necessary (should be `"DDIM"`, `DDIM_NSTEPS=10` was already
configured but unused); and `SS_RAMP_FRAC` cut to 0.0 (no ramp) since `current_epoch` was observed to
reset/relabel on every training resume. Ran a pure-DDIM lineage from `0.724_epoch=0` through epoch 5.
**Result: uniq_mid did NOT climb toward real levels — flat to slightly declining across epochs
(median ~14→13→11→13 for epochs 0/2/3/4), and epoch 5 introduced a new stability regression (timed
out on a day epoch 4 handled fine).** Killed 2026-08-03. `0.724_epoch=0` remains the final model,
unchanged. TODO before writing this up: LOB-Bench the epoch 2/3/4 CSVs against `0.724_epoch=0` for
the full 6-metric picture (uniq_mid was only ever a triage proxy) — `scripts/lob_bench_multiday.sh`.
**Write-up framing**: legitimate negative result, same standard as every other refuted lever in this
project (the controller, flow-balance, cancel-boost, cond-clip below) — state the hypothesis, the fix
attempts, the measured non-result, and the decision to retain the pre-retrain checkpoint.

**NEXT: reinforcement-learning task** — place an execution/trading agent inside the simulator, using
`0.724_epoch=0` as the backbone (deliberately the most rigorously-validated checkpoint, to avoid the
RL training absorbing confounds from an unstable environment).

## SESSION HANDOFF — 2026-07-21 (historical — superseded by the 2026-08-03 entry above for the
cross-day-robustness question, kept for the long-horizon/generalisation detail which still stands)
**Long-horizon stability is RESOLVED — decision tree landed on branch (b).** The away_run Phase 1 and 2
summaries are in and scored on LOB-Bench locally. Findings:
- **The plain 90-min config DOES diverge** (branch a ruled out). `LH_BASE` (dn0.3 + controller, no
  levers) produces one-sided ask walls up to ask1≈52168 while bid1 stays ~1150-5350; the exec
  controller alone does not stop it. `--depth-drift` was never in this test, so it was not the cause.
- **`--book-target-thick 2.0 --book-cancel-rate 0.5` bounds it** (branch b). Every `bt`-active cell is
  clean across all 18 five-minute buckets (bid1/ask1 ~3300-4800, non-lopsided). `--cond-clip` alone
  does nothing (clip-only cells still wall). We ADOPT `bt2.0_r0.5` for long horizons; we DROP
  `--cond-clip` (dead weight).
- **LOB-Bench confirms the fix is a fidelity GAIN, not a trade-off** (`lob_bench_stability_30min/`,
  `lob_bench_stability_90min/` on the Mac). 30-min: lever is neutral (mean Wasserstein 0.298 base →
  0.296) so adopting it costs nothing short-horizon. 90-min: lever WINS on every metric (mean 0.337 →
  0.255), driven by orderbook_imbalance 0.575 → 0.321 (the wall pathology, now quantified). With the
  lever the 90-min sim (0.255) is MORE realistic on aggregate than the 30-min base (0.296).
- **Tier-3 scheduled-sampling retrain is NOT needed** (branch c not triggered).

**Generalisation COMPLETE (2026-07-22) — the base config does NOT cleanly generalise, and the
σ-controller does NOT rescue it.** Full month (20 Jan-2015 days × 2 seeds, 30-min) in
`generalisation_summary.md` (local on Mac). Findings:
- **Universal type-mix signature holds across the month**: over-execution (gen exec median 14.9%,
  range 12-21%, vs real median 5.2%) and under-cancellation (gen ~31.7% vs real 45%) on EVERY day.
  The mechanism/characterisation generalises even though the calibration does not.
- **Bimodal drift**: ~8 of 20 days DRIFT (gen unique mids 90-187 vs real 27-66, price walks out of the
  real envelope), ~12 pinned. The sim's activity does NOT track the day: corr(real uniq_mid, gen
  uniq_mid) = -0.35. The drift is the fixed-σ over-execution tipping the closed loop.
- **σ-controller re-run REFUTED (`scripts/rerun_drift_controller.sh`, killed early — 4 days × 2 seeds
  gave a consistent answer).** Pinning `--dn-target-exec` to each day's real exec share HALVES
  over-execution (~15% → ~9-12%) but does NOT reach target and does NOT remove the drift; on some days
  the mid walks FURTHER out of envelope (0107 uniq_mid 176 → 295). Cause: σ-floor saturation (controller
  clips at 0.25× = σ≈0.075, cannot go lower) AND the drift is a closed-loop VARIANCE instability, not a
  σ-magnitude fault, so lowering σ cannot cure it. Do NOT re-attempt the controller as the drift fix.
  MEASURED (2026-07-23, local): the drift is EXCESS VARIANCE not a directional bias — generated
  buy/sell aggression is balanced everywhere measurable (Jan-30 30-min +0.004, 90-min wall run -0.002),
  the gen direction does NOT track the real day (Jan-30 real -38tk while gen ~flat), and the real month
  is 10up/10down so there is no learned downtrend (drift days' real mean +3.4tk). The "learned downward
  bias" idea was tested and refuted. Details + tables in `analysis/capabilities_summary.md`.
- **Drift also destroys the speed advantage**: drift days emit 90k-152k events (3-4× real) → the sim
  runs 1.3-3× SLOWER than real-time (2400-5800s for a 30-min window). A clean day (0130, ~17k events)
  runs ~680s = ~2.6× FASTER than real-time. Triple penalty: worse fidelity, off-envelope, and slow.
  The "2.7× vs DDPM" claim is relative/same-day and still holds; "faster than real-time" only holds on
  non-drift days.
- **Real fix = Tier-3 type-channel / scheduled-sampling retrain (still NOT built).** The controller and
  book-balance are decode-time band-aids; the directional/type-mix drift needs the retrain. Frame
  generalisation as a characterised limitation for now: present current-best on Jan-30 + well-behaved
  days; the sim is realistic-and-fast on the ~12/20 clean days, drifts on the rest.

**HSBC figures built (local, no remote needed):** `analysis/plots/hsbc_20260722/` — LOB-Bench fidelity
(DDIM-10 0.298 vs DDPM-100 0.507), the 90-min wall before/after (`2_walls_before_after.png`), stability
LOB-Bench (0.337 → 0.255), month-wide exec bars, plus `tables.md`. Built by `scripts/make_hsbc_figs.py`.

**Earlier away_run stability result (still valid, 2026-07-21):** long-horizon 90-min divergence on
Jan-30 is one-sided depth walls from under-cancellation, fixed by `--book-target-thick 2.0
--book-cancel-rate 0.5` (LOB-Bench 0.337 → 0.255, neutral at 30-min). `--cond-clip` DROPPED. That is a
SEPARATE failure mode from the cross-day over-execution drift above (depth walls vs price walk).

Launched on the remote with `nohup bash scripts/away_run.sh > away_run.out 2>&1 &`
(resume: `bash scripts/away_run.sh --root <dir>`). All decode-time, no training, fully resumable.
Phase outputs: `<root>/1_diagnostics/summary.md`, `<root>/2_stability/summary.md`,
`<root>/generalisation_summary.md` + `<root>/lobbench_manifest.txt`.

**NEW code this session (pushed to main, commits 950dbc3 + c8aadb6):**
- Two decode-time, default-off levers in `ABIDES/agent/WorldAgent.py` + `world_agent_sim.py`:
  `--book-target-thick T --book-cancel-rate r` (book-balancing spontaneous cancel: cancels own resting
  touch when a side > T x real mean level size; targets under-cancel AND divergence) and `--cond-clip C`
  (clip z-scored book SIZE conditioning to [-C,C]). Both add a `DIAG stability_levers:` line; 30-min
  winner unchanged when off. Revert via `git checkout -- ABIDES/`.
- Shells: `scripts/away_run.sh` (master), `scripts/long_session.sh`, `scripts/long_session_stability.sh`,
  `scripts/showcase_today.sh` (present-ready 3-cell + battery).
- `evaluation/stylized_custom/battery_reanchored.py` now parametrised (`--series`/`--date`/`--lob-dir`/
  `--out`), resolves gen paths as processed_orders.csv / run dir / flat CSV, and its default runs
  resolve by suffix (`tdprior_sr[_dn0.3/0.5/0.6][_te0.045]`). NOTE: the battery PLOTS look poor —
  cosmetic cleanup still TODO (user deprioritised).

## The winning configuration
```
DDIM -nsteps 10 -eta 0.0 --depth-noise 0.3 --size-reshape --type-decode prior     # 30-min
  + --depth-drift 0.2                                                              # optional volatility lever (see Direction A)
  + --dn-target-exec 0.045                                                         # long horizons (75-min+)
  + --book-target-thick 2.0 --book-cancel-rate 0.5                                 # long horizons — ADOPTED, bounds the depth divergence (see below)
```
on checkpoint **val_ema=0.627, epoch 15** (`data/checkpoints/TRADES/` on the remote) — the
UNCLAMP + PRICE_REANCHOR retrain. Both flag files (`UNCLAMP_DEPTH_FLAG`, `PRICE_REANCHOR_FLAG`)
must be present at sim time. `--depth-drift 0.2` is a kept-but-modest add-on (below); the core
result stands without it. `--book-target-thick 2.0 --book-cancel-rate 0.5` is ADOPTED for long
horizons (75-min+): LOB-Bench-validated to bound the 90-min depth divergence and improve fidelity
(mean Wasserstein 0.337 → 0.255) while being neutral at 30-min (0.298 → 0.296). `--cond-clip` was
tested alongside and DROPPED (no effect on its own or in combination).

## What's DONE (with evidence)
- **Freeze solved.** Diagnosed as depth-channel variance collapse under deterministic sampling
  (law-of-total-variance argument); fixed by per-sample decode-time depth-noise `--depth-noise σ`.
  The freeze control holds even on the converged checkpoint (`DDIM10_prior_raw`: B=2, exec 0%, 3 mids).
- **Two training-data bugs fixed** (depth clamp + self-referential indexing) → 0.00%→0.91% signed
  marketable targets. This is a *separate* fidelity fix; it did NOT unfreeze on its own (the control
  that proves the collapse is a sampler property, not a data one).
- **Price-OOD cliff removed** via `PRICE_REANCHOR` (subtract day-open mid). cond_z[price] went
  −3.4 → ~+0.5, stays in support even over 75 min. No more z≈−4 degeneration.
- **σ re-tuned on the new checkpoint** (`scripts/exec_bracket.sh`): σ=0.16 pins exec to real 7%.
- **LOB-Bench run** (results in `lob_bench_reanchored/` and `lob_bench_reanchored_75min/` on the Mac):
  - **dn0.3 = headline**: mean Wasserstein 0.298, seed-robust (0.276/0.286).
  - **Beats DDPM** (0.508, the worst config on aggregate — over-executes at 17% on this ckpt).
  - σ=0.16 (exec-matched) is only middling (0.387): matching the exec *rate* ≠ matching the
    *distributions*. Per-metric trade-off, no dominant σ.
  - **75-min fidelity holds** (0.326 vs 0.298 at 30-min; controller marginally better at 0.322).
- **Methodology written**: `analysis/methodology_draft.tex` — motivation, metrics, diagnosis
  (variance collapse + data-bug control + OOD boundary), intervention (4 fixes + 2 rejected
  alternatives), algorithms, and the results section w/ the LOB-Bench table. Compiles clean
  (needs amsmath, amssymb, booktabs, xcolor, natbib; cite LOB-Bench = Nagy et al. 2025).
- **Decision: type-reshape DROPPED.** The over-limit/under-cancel/thick-book gap does NOT show up
  as a LOB-Bench penalty (orderbook_imbalance flat ~0.44 across all incl. DDPM). Cosmetic; not worth building.

## Direction A sweep (2026-07-20) — impact/volatility levers
Full log: `analysis/cancel_sweep_table.md` (ckpt 0.627, DDIM10, dn0.3 base). Real targets:
ret1s_std 1.53bp, lag1 -0.09, cancel 43.8%, exec 7.0%, uniq_mid 69.
- **`--cancel-boost` REFUTED.** Meant to raise cancel share toward 43.8%; instead nudged it *down*
  (32.4 → 31.7% at cb2.0) while raising over-execution (12.4 → 13.2%). Extra decoded cancels are
  swamped by more crossing limits. A decode-time bias on the type score does NOT close the cancel
  gap. Removed from the config. See `analysis/refuted_approaches.md`.
- **`--depth-drift` KEPT (modest).** AR(1) directional bias on limit depth. dd0.2 lifts ret1s_std
  1.36 → 1.60bp (onto real 1.53) and holds mid coverage (21 mids); dd0.3 best repairs over-mean-
  reversion (lag1 -0.178 → -0.124) but drops coverage to 14. Genuine volatility lever, but it does
  NOT fix the cancel mix or the mid *range*, and its overall lift over plain dn0.3 is small.
- **Over-execution is channel B, not A.** `A_market_order=0` in every cell (the type decode never
  emits a market order); all executions are crossing limits. So the 12-14% exec overshoot is σ making
  too many limits cross, a σ-tuning matter, NOT a market-order flood. (Correct the methodology hedge.)
- **Brute-force σ is a false friend.** dn0.5/0.6 raise std (2.48/2.92) but wreck cancel (29%),
  exec (16%) and lag1 (-0.22/-0.26). Movement bought at the cost of every other metric.

## Open issues / known gaps
**Accepted as scope (frame as limitations, don't chase):**
- **No directional trend / low unique-mids.** The sim is range-bound (~15-21 mids, ~10-tk range) where
  real trended down ~37tk on sustained one-sided flow (real B-S -800, limOFI -690; generated B-S ~0).
  Symmetric depth-noise gives micro-volatility, not macro drift. **Reframe (2026-07-20): stop chasing
  uniq_mid≈69 — that range is largely the day's trend, which we explicitly do NOT want to reproduce.**
  The legitimate variance target is ret1s_std (real 1.53bp), which dd0.2 hits (1.60). Evaluate on
  stylized facts / return moments / ensembles, not trajectory or range reproduction.
- **Type mix**: over-limit (~55 vs 49), under-cancel (~32 vs 44), book ~2× thick. Cosmetic on
  LOB-Bench → not fixing. Confirmed 2026-07-20 that no decode-time lever (`--cancel-boost`) shifts it.

**Genuine open items:**
- **Long-horizon book-depth divergence — RESOLVED 2026-07-21.** Over 90 min the plain config (dn0.3 +
  σ-controller, no levers) accumulates one-sided touch walls (ask1 up to ~52168 while bid1 stays
  ~1150-5350); the σ-controller holds the execution rate but does NOT stop the lopsided depth
  accumulation, exactly the closed-loop amplification the mechanism predicts. FIXED by the adopted
  book-balancing cancel `--book-target-thick 2.0 --book-cancel-rate 0.5`: bounds bid1/ask1 to ~3300-4800
  non-lopsided across all buckets, and LOB-Bench-validated as a fidelity gain (90-min mean Wasserstein
  0.337 → 0.255, orderbook_imbalance 0.575 → 0.321) that is neutral at 30-min (0.298 → 0.296).
  `--cond-clip` was refuted for this (no effect alone or combined). No Tier-3 retrain needed. Scores in
  `lob_bench_stability_30min/` and `lob_bench_stability_90min/`.
- **Inter-arrival timing** is the weakest LOB-Bench axis for *every* sampler incl. DDPM (best ~0.48).
  Event timing is hard; no fix yet. Could explore a time-channel treatment, or characterise & accept.
- **DDPM baseline caveat**: our DDPM over-executes on this checkpoint (not the authors' tuned DDPM),
  so "beats DDPM" is a same-checkpoint controlled claim. A cleaner/tuned DDPM (or the authors'
  checkpoint) would harden it.
- **Generalisation**: everything is INTC, 2015-01-30 — one stock, one day. Not yet shown to transfer.
- **Training infra**: the retrain HUNG at epoch 15 (dataloader deadlock; no resume). We recovered the
  epoch-15 checkpoint, but a future clean converged run needs the dataloader hardened
  (`num_workers=0` or a timeout) to avoid the stall.

## Next things to run/test (prioritised for next GPU session)
1. **Generalise the winning config** — run `DDIM10 dn0.3 sr prior` (+ controller for long) on 1–2
   OTHER trading days (and TSLA if the LOBSTER data is available), score with LOB-Bench. This is the
   highest-value rigor step: shows the result isn't INTC-Jan-30-specific.
2. **Downstream use (the thesis "why")** — train/evaluate an execution agent inside the accelerated
   simulator. This is the stated motivation for a fast realistic sim; demonstrating it closes the arc.
3. **Cleaner DDPM baseline** — tune DDPM's exec rate on this checkpoint (or retry the authors'
   checkpoint) so the "beats DDPM" claim is airtight rather than caveated.
4. **Stylized-facts figures** — generate real-vs-generated overlays (`paper_style_stylized_facts.py` /
   `combined_stylized_facts_overlay.py`) for the winning config → thesis figures. CPU, no GPU needed.
5. **Fill methodology XXX placeholders** — the concept-draft diagnosis has XXX for a few diagnostic
   numbers (frozen/DDPM open-loop marketable %, size blow-up). Either use the 0.627 sweep numbers or a
   short diagnostic run; then build the two no-GPU figures (Φ(z*/s) curve; size blow-up bars).
6. *(Low priority)* LOB-Bench the remaining σ cells (0.125/0.15/0.17/0.18, te0.03/0.06) for a complete
   σ-vs-fidelity curve.

## Where things live
- **Checkpoint**: remote `data/checkpoints/TRADES/val_ema=0.627_epoch=15_*.ckpt`.
- **Generated CSVs** (18, ckpt 0.627): Mac `reanchored_csvs/`; also on remote branch `csv-xfer`.
- **LOB-Bench results**: Mac `lob_bench_reanchored/` (30-min), `lob_bench_reanchored_75min/`.
- **Scripts** (tracked, on remote via git): `scripts/away_run.sh` (master unattended driver),
  `scripts/long_session.sh`, `scripts/long_session_stability.sh`, `scripts/showcase_today.sh`,
  `scripts/direction_a_sweep.sh` (the cancel-boost/depth-drift sweep behind
  `analysis/cancel_sweep_table.md`), `scripts/sweep_reanchored.sh`, `scripts/exec_bracket.sh`,
  `scripts/reanchor_retrain.sh`, `scripts/tonight_retrain.sh`.
- **Docs**: `analysis/methodology_draft.tex` (the write-up), `analysis/dissertation_source_map.md`
  (reading guide for the whole project), this file.
- **Transfer note**: remote is ThinLinc — no scp/ssh from Mac, and no GitHub *push* auth on the remote
  (pull only). Move files by emailing a tarball to yourself (bundle → `~`/Desktop → webmail attach).
