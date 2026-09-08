# Instability paper — running log

*One dated paragraph per completed step, written as it happens — not reconstructed afterward.
See `/Users/Matthew/.claude/plans/moonlit-leaping-hamming.md` for the execution plan this log
tracks (Track A = local/CPU, Experiments 0/1/5; Track B = remote/GPU, Experiments 2/3/4).*

## 2026-09-07 — Track A build + Exp 0 kickoff

Built `rl_execution/book_state_error.py` (the Exp-0 core measurement) and
`scripts/exp0_abides_replay_error.sh` (the 20-day x 7-length driver). Design: two independent
matching-book reconstructions diffed at 5-minute intervals — ground truth from the raw LOBSTER
message CSV via `rl_execution/orderbook_reconstructor` (crossing-aware, needed because raw
exchange feeds have gaps), and ABIDES's own record from that run's `EXCHANGE_AGENT.bz2` (converted
via ABIDES's own `util/formatting/convert_order_stream.py`, replayed with pure add/reduce/remove
bookkeeping — no re-derived crossing, since ABIDES's own stream already reports every match's
outcome and re-deriving it would double-count). Deliberately did NOT use `processed_orders.csv`
for the ABIDES-side reconstruction: it's built with `ignore_cancellations=True`
(`ABIDES/realism/realism_utils.py`), which drops exactly the events this experiment needs to not
lose.

**Smoke test finding (2015-01-30, 30-minute window, one day):** `shares_missing` (ground-truth
resting shares minus ABIDES's own resting shares) grows from 0 at open to **-202,308 at 30
minutes**, and the growth is not linear-then-flat — it's still accelerating at the last sample
(-97,733 at 25 min -> -202,308 at 30 min). Sign convention: negative means ABIDES is retaining
*more* resting inventory than the real book ever had (128,229 real shares resting at 30 min vs
330,537 in ABIDES's own reconstruction) — consistent in direction with the prior finding cited in
the paper plan (1.47M vs 0.21M unretired inventory). One-sided (unquoted-side) pathology did NOT
occur on this day/window once a definition bug was fixed (see below) — real ground truth and
ABIDES's reconstruction both stayed two-sided throughout.

**Bug caught and fixed before the full sweep launched:** the first cut flagged "one-sided" any
time ABIDES's book had an empty side, which trivially fires at t=market_open (both
reconstructions start empty) and would have contaminated every day's headline flag. Fixed to
require the real ground truth be two-sided at that same instant — `one_sided = ab_one_sided and
not gt_one_sided` in `book_state_error.py`.

**Located a concrete candidate mechanism** for the growing-inventory finding (not yet confirmed as
the sole cause — Exp 0's plan step 6, per-order life story, would confirm it properly, deferred
until the full sweep's result is in): `ABIDES/agent/WorldAgent.py:_preprocess_events_for_market_replay`
drops any cancel/execute row whose order was never seen as a same-day `NEW_ORDER` row *before
simulation even starts* — a pre-simulation filter on the loaded LOBSTER file, not a kernel-level
bug. Whether this fully explains the magnitude above is exactly what Exp 0's fuller result (and,
if warranted, the per-order diff) should settle.

**Full sweep launched** (20 INTC days x {30, 60, 90, 120, 180, 240, 390} minutes, CPU-only,
resumable via `.done` sentinels) — running in the background as of this entry; results land in
`exp0_results/<timestamp>/`. Multi-day splicing (plan step 4's "longer where the data allows") was
scoped out for this pass — a single INTC day already reaches 390 minutes (the full continuous
session), which covers the "longer than the 30-120min used elsewhere" ask without the added
complexity of reconciling order-id namespaces across day boundaries. Noted, not silently dropped.

## 2026-09-07 — Exp 1 gap found: the "0.724" checkpoint is not in this repo

Built `scripts/exp1_pin_checkpoint.sh` and `analysis/model_under_test.md`. Finding: **only one
checkpoint has ever been committed to this repo on any branch** (`val_ema=0.7_epoch=2`), and its
own commit message documents it as an explicit fallback, not the preferred one ("Not the 0.627
winner ... but a usable checkpoint"). Reading `analysis/MASTER_RESULTS.md` as it exists on
`origin/main` (not present on `rl-execution`) identifies `val_ema=0.724` as the actual Phase-2
baseline checkpoint (`UNCLAMP_DEPTH` + `PRICE_REANCHOR` on, no scheduled sampling) — the closest
existing match to "the model under test" — but that checkpoint file itself is not present in this
repo, in git history, on any branch. It exists only on whatever machine trained it. Full reasoning
and the required next step (locate it on the remote GPU box, or retrain it with the flag state
logged from the start this time) are in `analysis/model_under_test.md`; its "Confirmed checkpoint"
section is deliberately left blank until this is resolved on the remote box — Track B must not
proceed past this point on a guess.

## 2026-09-07 — Exp 0 full sweep OOM'd locally, moved to the remote box

The full 20-day x 7-length sweep launched in the earlier entry ran for ~6 combos before a single
390-minute (full continuous session) real-replay run hit **23GB RSS** and exhausted the local
Mac's memory (down to ~64MB free, degrading the whole machine). Root cause: ABIDES's real-replay
keeps the full order stream + LOB snapshots (`log_orders=True`, `wide_book=True`) in memory for
the whole session with no bounded logging — cost scales badly with session length, not linearly.
Killed the run. Hardened `scripts/exp0_abides_replay_error.sh` with a `ulimit -v`-based
`--max-mem-gb` cap (default 16GB) around each real-replay subprocess, so a single cell OOMs
cleanly (one ERROR row, sweep continues) instead of taking the box down — note this is a no-op on
macOS (Darwin doesn't enforce `RLIMIT_AS`) but works as intended on Linux. Moved Experiment 0 to
run on the remote box instead (still CPU-only, no GPU needed — its own `exp0-cpu` tmux window in
`rl_execution/RUNBOOK_instability.md`, alongside rather than queued behind `gpu-main`) rather than
re-risking the local machine. Not yet re-launched as of this entry.

## 2026-09-08 — Checkpoint selection hardened against the val_ema-collision landmine

User placed checkpoints on the remote box, "among others" (i.e. a directory now holding more than
one file). `analysis/MASTER_RESULTS.md` §1.4 already documents that two checkpoints can share a
rounded val_ema and be indistinguishable by `--id`/`-id` matching (e.g. `0.69_epoch=2` vs
`0.69_epoch=4`) — exactly the failure mode that would silently select an SS-resumed checkpoint
instead of the plain baseline, or vice versa. Closed this before it could bite:

- `evaluation/diagnostics/open_loop_eval.py`: added `--ckpt-path` (exact file, bypasses `--id`
  entirely, mirrors `world_agent_sim.py`'s own flag), and made `find_checkpoint` raise instead of
  silently keeping the last match when `--id` is ambiguous.
- `scripts/exp3_teacher_forced.sh`: switched from `--ckpt-val-loss` (ambiguous) to `--ckpt-path`
  (exact), matching `exp2_survival_sweep.sh`'s and `exp4_checkpoint_sweep.sh`'s existing
  `--ckpt-path` convention — every Track B script now takes an exact file everywhere, no script
  takes a bare val_ema number.
- `scripts/exp1_pin_checkpoint.sh`: now lists every checkpoint, flags known decoys by val_ema
  (0.627/0.681/0.719 pre-fix, 0.7_epoch the documented fallback), and explicitly reports
  zero/one/many matches for `val_ema=0.724*` rather than just checking existence — many-match
  case prints all candidates and refuses to pick one.
- `scripts/exp4_checkpoint_sweep.sh`: excludes the same known decoys from its default sweep (a
  mixed checkpoint directory would otherwise silently turn "robustness across checkpoints from
  this training run" into "robustness across unrelated runs"); `--include-decoys` overrides.

## 2026-09-08 — Redesigned for a no-copy-paste remote terminal

User feedback after the first remote attempt: pasting the checkpoint path into
`analysis/model_under_test.md`'s markdown did nothing — no script ever read it back out, so it was
purely decorative. Combined with "I cannot copy and paste on the remote," hand-typing a full
checkpoint path was exactly the failure mode that happened. Also: the runbook gave phase labels
("B1 pilot") instead of literal commands. Fixed both:

- `analysis/model_under_test.md`'s "Confirmed checkpoint" section is now a machine-readable
  `CKPT_PATH=...` line, not freeform text.
- `scripts/exp1_pin_checkpoint.sh` now WRITES that line automatically the moment it finds exactly
  one unambiguous `val_ema=0.724*` match — nothing to type or paste.
- New `scripts/_ckpt_lib.sh`, sourced by `exp2_survival_sweep.sh` and `exp3_teacher_forced.sh`:
  resolves the checkpoint from that line automatically; `--ckpt-path` is now an optional override,
  not a required argument.
- `exp2_survival_sweep.sh` gained `--smoke`/`--pilot` zero-typing presets; every Track B script
  now runs as a single short command with no flags in the common case.
- `rl_execution/RUNBOOK_instability.md` rewritten as six numbered, literal, copy-paste-free
  commands (each stated as what to type, what it does, and what to check), replacing the old
  phase-label ("B0"/"B1"/...) style.

## 2026-09-07 — Track B package built (scripts only, not yet run)

Built the paste-back package for the remote GPU box: `rl_execution/survival_metrics.py` (frozen/
diverged scoring, pre-registered definitions — see the plan file), `scripts/exp2_survival_sweep.sh`,
a `--bucket-by-time` patch to `evaluation/diagnostics/open_loop_eval.py`, `scripts/exp3_teacher_forced.sh`,
`scripts/exp4_checkpoint_sweep.sh`, and `rl_execution/RUNBOOK_instability.md` tying them together.
None of these have been run — they need the checkpoint gap above resolved first, and a GPU.
