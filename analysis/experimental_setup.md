# Experimental setup — what was held fixed, what varied, and when

*2026-08-09. Reconstructed from the sweep scripts, the retained result markdown, and the run
directory listing. Companion to `eval_methodology_handoff.md`, which covers the metrics themselves.*

The work falls into **three phases** with genuinely different designs. The distinction matters for
the write-up: the diagnostic phase deliberately used one day and a broken checkpoint, and defending
that choice is easy — but only if it is stated as a choice rather than left to look like a limitation.

---

## The invariant core

Held constant across every phase unless explicitly varied:

| | Value |
|---|---|
| Instrument | INTC (Intel), LOBSTER level-10 |
| Period | January 2015, 20 trading days |
| Session start | **09:30:00** |
| Warm-up | **09:45** — the first 15 minutes are replayed real orders; all metrics are computed on events after this cutoff |
| Seed | **30** (31 and 32 only in the robustness cells) |
| Simulator | ABIDES, real matching engine, single world agent |
| Sampler default | DDIM, η = 0 |

The 15-minute warm-up is a real design decision worth one sentence: the world agent needs a
populated book and a conditioning history before its own output can be judged, so the first quarter
hour is replay and is discarded from every statistic.

---

## Phase 1 — Diagnosis (5–11 July)

**Design: one day, one window, one broken checkpoint, vary the sampler.**

| | |
|---|---|
| Day | **2015-01-30**, almost exclusively |
| Window | **09:30–10:00** (30 min) |
| Checkpoints | `0.681`, `0.719`, `0.627`, `0.7_epoch=2` — the **pre-fix lineage** (conditional dropout 0.0, lr 1e-3) |
| Decode config | **vanilla, or one fix at a time** — isolating mechanisms was the whole point |
| Varied | sampler (DDPM-100; DDIM-1/10/20/100; η ∈ {0, 0.4, 1}), noise placement (head/tail hybrids), CHURN, classifier-free guidance, depth-temperature, depth-reshape |

**Why 2015-01-30.** It is one of TRADES's own two test days and the day they released output for, so
every diagnostic number sits directly against their published behaviour. Using their test day for
diagnosis is the defensible choice, not a shortcut.

**Why a broken checkpoint.** `0.681` freezes under deterministic sampling — which is the phenomenon
under study. A checkpoint that does not freeze cannot be used to diagnose a freeze. This is worth
stating plainly, because it pre-empts the obvious objection when §5.2's phenomenon turns out not to
occur on the checkpoints eventually adopted.

**Where the results live:** `hypothesis_results.md`, `ETA_SUMMARY.md`, `new_ckpt.md`,
`analysis/churn_results.md`, `analysis/reshape_results.md`, `analysis/cancel_sweep_table.md`

---

## Phase 2 — Checkpoint selection and cross-day stability (late July – early August)

**Design: fix the configuration, vary the day and the checkpoint.**

| | |
|---|---|
| Days | **all 20 January trading days** |
| Window | 09:30–10:00 (30 min), unchanged from Phase 1 |
| Checkpoints | `0.724_epoch=0` (post-fix baseline) and scheduled-sampling epochs `0.701`, `0.697`, `0.69`, `0.682` |
| Decode config | **frozen** at the adopted set: `--depth-noise 0.3 --size-reshape --type-decode prior` |
| Data-pipeline flags | `UNCLAMP_DEPTH`, `PRICE_REANCHOR` on throughout |
| Per-day cap | **40 minutes** wall-clock; a timeout counts as unstable |
| Abandonment | a checkpoint failing on **any** day is dropped immediately rather than exhaustively confirmed |

The inversion from Phase 1 is the point: configuration becomes the control and the *day* becomes the
independent variable. That is what licenses a generalisation claim, and it is why the ordering in the
results chapter (diagnose → fix → generalise) reads correctly.

Hardest days run **first** (`20150107`, `20150129`) so an unstable checkpoint fails fast.

**Where:** `analysis/appendix_checkpoint_evidence.md`, `ckpt_search/`, `lob_bench_0724_full_month/`

---

## Phase 3 — Long-horizon and final characterisation (6–8 August)

**Design: two days, longer windows, the adopted checkpoints.**

| | |
|---|---|
| Days | **2015-01-29** primarily, **2015-01-30** as the second day |
| Windows | **10:00–12:00** (2 h) · **10:00–11:00** (60 min, drift sweep) · **09:45–11:00** (75 min, matching TRADES's released coverage) |
| Checkpoints | `0.724_epoch=0` and SS epoch 4 (`0.69`) |
| Seeds | 30, plus 31 and 32 for the robustness cells |
| Decode config | adopted set, plus lever arms (`--depth-drift`, `--book-target-thick`) |

**Why the horizon changed.** Every earlier evaluation stopped at 30 minutes — an hour before the
single-step divergence at ~minute 73 becomes visible. The 2-hour window exists because 30 minutes
cannot see the failure mode the work addresses.

**Why 2015-01-29.** It is the other TRADES test day, has released output, and is the busiest session
in the month — roughly twice the wall-clock cost of 2015-01-30 at the same window, which is worth
knowing when budgeting.

**Where:** `analysis/plots/longhorizon/`, `analysis/replication_baselines/`, `drift_sweep/`

---

## Day roster — which day does what

| Day | Role |
|---|---|
| **2015-01-30** | diagnostic day (Phase 1); TRADES test day; **released output available** |
| **2015-01-29** | long-horizon day (Phase 3); TRADES test day; **released output available**; busiest session |
| **2015-01-07** | hardest known drift day; run first in every stability sweep |
| all 20 | cross-day stability and full-month LOB-Bench |

Only 0129 and 0130 permit a like-for-like comparison against TRADES, because those are the only days
they released output for. Everything else is our own model against real.

---

## Reading a run directory name

```
world_agent_{TICKER}_{DATE}_{ET}_{SEED}_{SAMPLER}_{ETA}_{NSTEPS}_{ckpt[:13]}{flags}
```

Flag suffixes, in the order they are appended:

| Suffix | Flag |
|---|---|
| `_tdprior` | `--type-decode prior` |
| `_sr` | `--size-reshape` |
| `_dn0.3` | `--depth-noise 0.3` |
| `_te0.045` | `--dn-target-exec 0.045` |
| `_dd0.25p0.9998` | `--depth-drift 0.25 --depth-drift-phi 0.9998` |
| `_bt2.0r0.5` | `--book-target-thick 2.0 --book-cancel-rate 0.5` |
| `_cc5.0` | `--cond-clip 5.0` |

**A directory with no suffix is a vanilla run** — no decode corrections. That is how the Phase 1
diagnostics and the checkpoint-dependence cells are identified.

### Two traps in the naming

1. **Only the END time appears, never the start.** `_11-00-00_` is both the 60-minute drift-sweep
   window (10:00–11:00) and the 75-minute replication window (09:45–11:00). Check the launching
   script, not the path.
2. **The checkpoint is truncated to 13 characters.** `val_ema=0.69_` is identical for
   `0.69_epoch=2` and `0.69_epoch=4`. The epoch is **not recoverable from the path** — see
   `final_model_config.md`.

Real replays follow `market_replay_{TICKER}_{DATE}_{ET}_{SEED}` and are produced by the same entry
point without `-d True`.

---

## Undocumented runs on the remote

Three sets appear in the log listing with no corresponding entry in any analysis document:

- `world_agent_AAME_2015-01-{29,30}_*_DDPM_0.0_1_val_ema=2.99*` — a different instrument on a model
  with validation loss ~3.0, i.e. a far weaker checkpoint. Almost certainly early exploratory work.
- `world_agent_INTC_2008-09-25_12-00-00_30_DDIM_0.0_1_val_ema=0.774` — an out-of-period date.
- `rl_execution_INTC_*` — several hundred directories from the reinforcement-learning execution
  workstream, unrelated to the simulator evaluation.

**Do not cite any of these without re-establishing what they were.** None is referenced in the
retained result files, and inferring purpose from a directory name is how the checkpoint-truncation
error nearly happened.

---

## The distinction to draw in the write-up

The cleanest framing, and the one the evidence supports:

> **Phase 1 varies the sampler on a fixed day and a checkpoint chosen because it fails.** Its job is
> mechanism, and its findings are causal claims about why deterministic few-step sampling collapses.
>
> **Phases 2 and 3 fix the configuration and vary the day, the horizon and the seed.** Their job is
> generalisation, and their findings are claims about robustness.

The single-day, single-checkpoint design of Phase 1 is a strength for the question it answers and
would be a weakness for the question Phase 2 answers — which is exactly why the design changed.
Saying so explicitly is better than letting an examiner ask why the diagnosis rests on one day.

---

## Methodology facts — verified against source

*Added 2026-08-09 to close specific gaps in the methodology chapter. Every value below was read
from the code, not recalled; file and line given so each can be re-checked.*

### Train / validation / test split

`constants.py:271` → `SPLIT_RATES = (.85, .05, .10)`, applied **by trading day** (not by sample) in
`preprocessing/LOBSTERDataBuilder.py:225-234`. Files are enumerated with `sorted(os.listdir(path))`
(lines 79, 84), and LOBSTER filenames begin with the date, so the split is **chronological and
sequential**. For 20 days:

| Split | Index | Days | Dates |
|---|---|---|---|
| train | 0–16 | 17 | 2015-01-02 → 2015-01-27 |
| validation | 17 | 1 | 2015-01-28 |
| **test** | **18–19** | **2** | **2015-01-29, 2015-01-30** |

**The evaluation days are the held-out test days**, and they coincide with the two days TRADES
declares as its test set — so the comparison against their released output is like-for-like on a set
neither model trained on.

⚠️ **One qualification to state.** The 20-day cross-day stability sweep runs on *every* day,
including the 17 training days. So the **fidelity** results on 0129/0130 are out-of-sample, while
the **stability** result is partly in-sample. That is defensible — stability under closed-loop
rollout is not the quantity the model was fitted to, and a model can be trained on a day and still
diverge on it — but it should be said rather than left for a reader to notice.

### `val_ema`

The number in every checkpoint filename. It is the **validation loss evaluated under
exponential-moving-average weights**, averaged over validation batches:

- `diffusion_engine.py:62` — `ExponentialMovingAverage(self.parameters(), decay=0.999)`, updated
  every training step (line 223)
- `diffusion_engine.py:287` — validation runs inside `with self.ema.average_parameters()`
- `diffusion_engine.py:303` — `loss_ema = sum(val_ema_losses) / len(val_ema_losses)`

The loss itself is the hybrid diffusion objective, `L_simple + L_vlb`. Lower is better; it is the
quantity `ModelCheckpoint` and `EarlyStopping` monitor (`run.py:129`, patience 6, min_delta 0.005).

**It is not a fidelity measure**, which is the whole point of §5.2.6 — it selects on denoising
accuracy, not on simulator behaviour.

### Hyperparameters (all from `configuration.py`)

| Parameter | Value | Line |
|---|---|---|
| learning rate | 2.5e-4 | 56 |
| batch size (train / test) | 256 / 512 | 54–55 |
| conditional dropout | 0.1 | 72 |
| epochs configured | 50 (stopped at 5) | 57 |
| **sequence length `N`** | **256** | 66 |
| masked sequence size (events generated at a time) | 1 | 67 |
| augmenter | MLP, dim 64 | 24, 94 |
| CDT depth | 8 | 81 |
| CDT MLP ratio | 4 | 82 |
| **CDT attention heads** | **1** | 83, and see below |
| diffusion steps `T` | **100** | 74 |
| noise schedule | cosine, offset **s = 0.008**, `max_beta = 0.99` | `utils/utils.py:15-33` |
| conditioning | concatenation, `full` (LOB levels) | 87–90 |

`CDT_DEPTH = 8` builds **8 blocks alternating self-attention and cross-attention**
(`models/diffusers/TRADES/Transformer.py:22`), not 8 self-attention blocks.

### ⚠️ Attention heads: **1**, not 2

`run.py:169` overwrites the config value on the TRADES path:

```python
config.HYPER_PARAMETERS[CDT_NUM_HEADS] = aug_dim // 64
```

`AUGMENT_DIM = 64`, so `64 // 64 = **1**` — which also matches the `configuration.py:83` default and
the `au_64` field in every checkpoint filename. **A claim of "two attention heads" is incorrect** and
should be corrected to one.

### Scheduled-sampling lineage: resumed, not trained from scratch

`run.py:89-101`. With `RESUME_TRAINING_FLAG` present, training resumes via Lightning's
`trainer.fit(ckpt_path=...)`, which restores **weights, optimizer state and epoch counter** from the
newest checkpoint. The SS lineage therefore continues the `0.724` run rather than starting fresh —
"resumed" is more accurate than "fine-tuned", since the optimizer state is not reset.

Related, and worth a clause: `constants.py:42` notes `current_epoch` resets on resume, which is why
the gradual ramp was dropped (`SS_RAMP_FRAC = 0.0`) in favour of one teacher-forced epoch then full
strength.

### The rollout-sampler correction

`SAMPLING_TYPE` defaulted to `"DDPM"`, so an earlier scheduled-sampling run generated its
self-conditioning with the 100-step schedule rather than the 10-step DDIM used at simulation time.
Corrected in **`bb87b79` (2026-07-31)**; the adopted checkpoint comes from the subsequent pure-DDIM
retrain.

State this. The claim "the rollout sampler matches deployment" is true of the adopted checkpoint
*because of* that fix, and saying so makes the claim verifiable rather than merely asserted.

### File-gated flags are unrecoverable from artefacts

`UNCLAMP_DEPTH_FLAG` and `PRICE_REANCHOR_FLAG` are set by the **presence of a file** in the repo
root (`constants.py:17, 28`). They appear in no command line, no run directory name and no output
file. Two consequences, both worth stating:

1. They must match between training and simulation — a model trained under them requires them at
   simulation time.
2. **They cannot be recovered from a completed run.** Provenance depends on the launching script, so
   any reproduction must set them explicitly.

### Not recorded anywhere in the repository

**Training hardware and wall-clock duration.** No training log retains epoch timings. What is known
from session records: recent work ran on an RTX 4070 (16 GB); earlier training used the UCL CS
GPU boxes (`cream`/`vanilla`, sm_75). Supply the specific machine and duration from your own notes —
this cannot be reconstructed from the repository.
