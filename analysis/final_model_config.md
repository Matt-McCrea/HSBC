# Exact configuration — the two-hour headline run

*Reference for the methods section. Every intervention, where it lives, and what it does.
Reconstructed from the run artefacts and git history on 2026-08-08, not from recollection.*

**The run:** `ABIDES/log/paper_runs_downloaded/world_agent_INTC_2015-01-29_12-00-00_30_DDIM_0.0_10_val_ema=0.69__tdprior_sr_dn0.3`
INTC, 2015-01-29, 10:00–12:00, seed 30. LOB-Bench grand mean **0.257**; end 34.09, range 41 tk,
ret1s_std 1.65 bp, 64 unique mids (real: 33.99 / 56 tk / 1.26 bp / 92).

---

## The command

```bash
# three interventions are FILE-GATED, not CLI flags — see §3. They must exist in the repo root
# at both training and simulation time.
touch UNCLAMP_DEPTH_FLAG PRICE_REANCHOR_FLAG

python ABIDES/abides.py -c world_agent_sim -t INTC -date 20150129 \
  -st 10:00:00 -et 12:00:00 -d True -m TRADES \
  -type DDIM -nsteps 10 -eta 0.0 \
  --ckpt-path data/checkpoints/TRADES/val_ema=0.69_epoch=4_INTC_se_256_au_64_CD_8_seed_30.ckpt \
  -seed 30 \
  --depth-noise 0.3 --size-reshape --type-decode prior
```

---

## 1. Architecture — unchanged from TRADES

No architectural modification was made. Berti, Prenkaj & Velardi (2025), arXiv:2502.07071.

| | value | checkpoint-name field |
|---|---|---|
| sequence length | 256 | `se_256` |
| augmenter dim (MLP) | 64 | `au_64` |
| conditional denoising transformer depth | 8 | `CD_8` |
| CDT heads / MLP ratio | 1 / 4 | — |
| conditioning | concatenation, `full` (LOB levels) | — |
| masked seq size (events generated at a time) | 1 | — |
| seed | 30 | `seed_30` |

## 2. Training

| | value |
|---|---|
| learning rate | 2.5e-4 |
| batch size | 256 |
| conditional dropout | 0.1 |
| epochs configured / reached | 50 / stopped at 5 |
| **checkpoint used** | **epoch 4**, val_ema 0.69 |

**Scheduled sampling** (the one training-side intervention). Gated by a `SCHEDULED_SAMPLING_FLAG`
file; `constants.py` holds the schedule:

| | value | meaning |
|---|---|---|
| `SS_P_MAX` | 0.5 | max fraction of training steps conditioned on self-generated history |
| `SS_RAMP_FRAC` | **0.0** | no gradual ramp — one teacher-forced epoch, then straight to `SS_P_MAX` |
| rollout sampler | DDIM, 10 steps, eta 0 | the model rolls forward with the *same* sampler used at simulation time |

The rollout sampler matters and is easy to misreport: `SAMPLING_TYPE` defaulted to `DDPM`, so an
earlier retrain generated its self-conditioning with the 100-step schedule. Fixed in `bb87b79`
(2026-07-31). This checkpoint comes from the corrected, pure-DDIM retrain.

## 3. Data-pipeline interventions — FILE-GATED, not CLI

**The reproducibility trap.** These are set by the *presence of a file* in the repo root, are not
recorded in the output directory name, and cannot be recovered from a run's artefacts afterwards.
State them explicitly in the paper; a reader given only the command line would not reproduce the run.

| flag file | what it does |
|---|---|
| `UNCLAMP_DEPTH_FLAG` | stops clamping the depth channel, so the crossing-limit tail survives preprocessing |
| `PRICE_REANCHOR_FLAG` | re-anchors prices in the conditioning window instead of z-scoring against a global reference |

Both must match between training and simulation.

## 4. Decode-time interventions — the three CLI flags

| flag | value | what it does |
|---|---|---|
| `--depth-noise` | 0.3 | per-sample N(0, σ) added to `z_depth` at decode, **LIMIT orders only**. Splits the collapsed depth atom. Never enters the sampler or other channels. |
| `--size-reshape` | on (`data/quantile_targets`) | quantile-reshapes decoded size onto the real per-type size marginals. Also removes the 30–40% negative-size decode waste. |
| `--type-decode` | `prior` | Bayes prior-corrected type decode, penalising the geometrically oversized MARKET region (default is `l1`). |

`--size-reshape` needs `data/quantile_targets/real_size_*.npy` — build with
`scripts/build_quantile_targets.py`.

## 5. Sampler

DDIM, **10 steps**, eta 0.0 (deterministic). Against TRADES's DDPM-100 default.

Measured on this run (`timing_summary.txt`): 228,131 orders, **11.52 ms/order**
(augmenter 0.45 + network 11.07), 86.8 orders/s, 2628.4 s total network wall-clock.

## 6. Deliberately NOT used

All default to 0/off. Worth listing — several were tested and rejected, and the headline run
uses none of them:

`--depth-drift`, `--depth-drift-phi`, `--book-target-thick`, `--book-cancel-rate`,
`--cancel-boost`, `--dn-target-exec`, `--cond-clip`, `--flow-balance`, `--guidance-scale`,
`--fix-cancel-bind`, `--fix-lob-pad`, `--drop-type2-cond`

---

## Two things not to get wrong

**The book-balancing lever run is a different run.** `..._dn0.3_bt2.0r0.5` adds
`--book-target-thick 2.0 --book-cancel-rate 0.5`. That is the P4 variance-ratio experiment
(VR(60s) 0.047 → 0.112), *not* the LOB-Bench headline. Don't merge their numbers.

**The output directory name does not identify the epoch.** `world_agent_sim.py:545` builds it from
`checkpoint_reference.name[:13]`, which truncates to `val_ema=0.69_` — identical for
`0.69_epoch=2` and `0.69_epoch=4`. This run's epoch is established from
`scripts/paper_figure_runs.sh:40` (`HEADLINE="0.69_epoch=4"`, matched as an exact substring),
not from the path.

## Verified how

- decode flags, sampler, date/window/seed → the run directory name
- timing → `timing_summary.txt` in the run directory
- epoch → `scripts/paper_figure_runs.sh:40`
- architecture, lr, dropout, batch → `configuration.py`
- SS schedule → `constants.py:38-42`; rollout-sampler fix dated via `git log` (`bb87b79`, 2026-07-31)

**Not verifiable from this machine:** the checkpoint file lives on the remote filestore. Confirm the
baked-in lr and conditional dropout with `scripts/check_checkpoint_configs.py`. `SS_RAMP_FRAC` is
*not* stored in the checkpoint — that it was 0.0 is inferred from commit dates (`544109b`,
2026-07-31, before the retrain), which is strong but indirect.
