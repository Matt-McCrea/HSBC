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
