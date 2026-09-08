# Instability paper — Track B runbook (remote GPU box)

Companion to `/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md`. Track A was meant to run
locally (CPU-only, Experiments 0/1/5), but Experiment 0's full sweep OOM'd the local Mac — a
single 390-minute real-replay run hit 23GB RSS (ABIDES's real-replay keeps the whole order stream
+ LOB snapshots in memory for the session, uncapped). Moved to this runbook instead, in its own
CPU-only window (`exp0-cpu` below) — it does NOT need the GPU, so it can run alongside window
`gpu-main` rather than queuing behind it. Experiments 2, 3, 4 (the `gpu-main` window) are
everything that actually needs the GPU, on a single-GPU box, so THAT'S one serial queue, not
something to parallelize across shells. Every step prints a summary to stdout — paste it back for
review, same convention as `RUNBOOK_replay.md`.

## Before anything: the checkpoint gate

**`analysis/model_under_test.md`'s "Confirmed checkpoint" section must be filled in before any
script below is run for real.** All three scripts below refuse to run (or, for
`exp3_teacher_forced.sh`, require an explicit `--ckpt-path`) until that's true — deliberate, not a
bug to work around. Start with:

```bash
bash scripts/exp1_pin_checkpoint.sh
```

It lists every checkpoint in `data/checkpoints/TRADES/`, flags known decoys (0.627/0.681/0.719 —
pre-fix; 0.7_epoch — the documented fallback), and tells you whether `val_ema=0.724*` resolves to
exactly one file. **If a checkpoint directory holds more than one file, always pass the checkpoint
to every script below via the exact `--ckpt-path`, never a val_ema number** — two checkpoints
(e.g. the plain baseline and an SS-resumed one) can share a rounded val_ema, which is exactly the
landmine `analysis/MASTER_RESULTS.md` §1.4 documents. Once resolved, copy the exact file path into
`analysis/model_under_test.md`'s "Confirmed checkpoint" section **before** anything else here.

## tmux layout

```
tmux new -s instability
# window 0: exp0-cpu    -- Experiment 0's real-flow sweep, CPU-only, no GPU/checkpoint needed at all
# window 1: gpu-main     -- the only window that touches the GPU, everything below runs here in order
# window 2: gpu-monitor  -- tails logs / runs the cheap CSV-only rollups, never touches the GPU
```

### window `exp0-cpu`

No GPU, no checkpoint, no dependency on anything else in this runbook — start this immediately,
it can run for hours in the background while the checkpoint gate above gets sorted out.

```bash
# check this box's actual free RAM first, and size --max-mem-gb below well under it (leave
# headroom for whatever gpu-main is doing at the same time -- GPU training/sampling still wants
# host RAM, not just VRAM):
free -g   # or: awk '/MemAvailable/{printf "%.0f GB available\n", $2/1e6}' /proc/meminfo

# full sweep (20 days x 30/60/90/120/180/240/390 min) -- 390min alone hit 23GB locally, so on a
# box with e.g. 64GB free this should be safe at the default cap; raise --max-mem-gb if you know
# this box can take more. Resumable via .done sentinels -- safe to re-launch after a kill.
bash scripts/exp0_abides_replay_error.sh --max-mem-gb 32
```

If any cell OOM-kills under the cap (check `logs/real_<day>_<et>.txt` for the "exit ... OOM-kill"
note the script prints), that day/length combo just gets an ERROR row in `summary.md` and the
sweep continues — rerun that one cell alone with a higher `--max-mem-gb` once you've confirmed the
box has room, rather than raising the default for everything.

### window `gpu-main`

```bash
# B0 — smoke test: does the pinned checkpoint load, do the flags take, does one short run
# produce a scoreable CSV? (small scale, ~15-20 min real+gen)
bash scripts/exp2_survival_sweep.sh --days 20150130 --seeds "30 31" --et 10:00:00 \
    --reanchor off --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt
cat exp2_results/*/summary.md   # both runs should produce a real score.json, not an ERROR block

# B1 — Exp 2 pilot (per the plan's own suggestion: read this before paying full cost)
bash scripts/exp2_survival_sweep.sh --days "20150130 20150107 20150115" --seeds "30 31 32 33 34" \
    --et 11:30:00 --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt
# READ: does the pilot's failure pattern look like what the dissertation already reports?
# If not, stop and figure out why before spending the full sweep's compute.

# B2 — Exp 2 full sweep (the dominant cost; defaults are 20 days x 20 seeds x both reanchor
# variants, --et defaults to a 4h horizon -- override --days/--seeds/--et to size it to the
# GPU window actually available)
bash scripts/exp2_survival_sweep.sh --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt

# B3 — Exp 3 teacher-forced control (same checkpoint, both reanchor variants; no day/seed grid --
# see the script's own header for why)
bash scripts/exp3_teacher_forced.sh --ckpt-path data/checkpoints/TRADES/<confirmed>.ckpt

# B4 — Exp 4 checkpoint sweep (after B2's headline number is in hand)
bash scripts/exp4_checkpoint_sweep.sh --day 20150107 --seeds "30 31 32 33 34" --et 10:30:00
```

### window `gpu-monitor`

```bash
tail -f exp2_results/*/summary.md   # or whichever phase is currently running
# cheap rollup against whatever .done runs exist so far, without touching the GPU:
python3 -c "
import glob, json
for jf in sorted(glob.glob('exp2_results/*/logs/*.score.json')):
    print(jf, json.load(open(jf)).get('survived_to_end'))
"
```

## Reading the results

- Exp 2's real headline is the survival-fraction-over-time curve, pooled and per-day — build it
  from every `exp2_results/*/logs/*.score.json` (`time_to_freeze`, `time_to_diverge`,
  `survived_to_end`) plus `data`viz once there's enough of them to be worth plotting.
- Exp 3: compare `bucket_early` vs `bucket_late` in `exp3_results/*/open_loop_*.json` against
  Exp 2's own early-vs-late behaviour. Flat open-loop + failing closed-loop = compounding
  self-generated error confirmed. Degrading open-loop too = a different story, write it up
  honestly.
- Exp 4: `exp4_results/*/summary.md`'s per-checkpoint table. Most/all checkpoints failing = not
  one unlucky snapshot. A from-scratch second training run (stretch, not scripted here — it's a
  multi-day GPU booking of its own) is the strongest version of this test if compute remains.
- Append one dated paragraph per completed phase to `analysis/instability_paper_log.md` as it
  happens, same as Track A did — don't wait until the end.
