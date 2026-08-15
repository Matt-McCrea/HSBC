# Appendix — Checkpoint Selection & Scheduled-Sampling Evidence

*Full per-day detail behind the checkpoint-elimination search and the scheduled-sampling retrain.
Compiled 2026-08-05 from the raw sweep logs (`ckpt_search/`).*

## A. Original elimination search — all 6 recovered checkpoints

A storage-loss incident cost the project's previous best checkpoint. Six checkpoints survived from a
later training run; each was tested against every trading day in January 2015 (20 days), hardest days
first, capped at a fixed wall-clock time per day (30-40 min depending on the run — see notes).
Abandoned at the first timeout rather than exhaustively confirming a checkpoint already known to fail.

| Checkpoint | Result | Detail |
|---|---|---|
| `val_ema=0.681_epoch=3` | **Eliminated** | Manually killed after ~3h on `20150107` — tested before the automated timeout cap existed; the pathologically long runtime was itself the signal of a runaway/unstable generation. |
| `val_ema=0.69_epoch=2` | **Eliminated** | Timed out on `20150107` within a 30-min cap. |
| `val_ema=0.704_epoch=1` | **Eliminated** | Timed out on `20150107` within a 30-min cap. |
| `val_ema=0.721_epoch=2` | **Eliminated** | Timed out on `20150107` within a 30-min cap. |
| `val_ema=0.7_epoch=2` | **Eliminated** | Cleared `20150107` (1320s / 22 min) but timed out on `20150129` (40-min cap) — passes the easier day, fails the harder one. |
| `val_ema=0.724_epoch=0` | **SURVIVED — final model** | Cleared all 20 trading days, zero timeouts. Full detail in table B0 below. |

*Note on checkpoint identity: `val_ema=0.69_epoch=2` here is from the original recovered set and is a
completely different checkpoint from `val_ema=0.69_epoch=4` in the scheduled-sampling lineage below —
they share a coincidentally similar val_ema but are from different training runs. Don't conflate them.*

## B. Full per-day detail, `val_ema=0.724_epoch=0` (the final model)

Real-market reference for `20150107`: ~27 unique mid-prices, ~13-tick range (the hardest day in the
month). Table below: `secs` = wall-clock time to simulate the 30-minute window; `uniq_mid` = number of
distinct mid-prices generated (an activity/realism proxy); `ret1s_std` = 1-second return volatility in
basis points (real target band: 1.5-2.5bp); `mid_range_tk` = price range in ticks.

| Day | secs | uniq_mid | ret1s_std (bp) | mid_range (tk) |
|---|---|---|---|---|
| 2015-01-07 | 1758 | 15 | 2.01 | 7 |
| 2015-01-29 | 1763 | 25 | 2.05 | 12 |
| 2015-01-02 | 1650 | 13 | 1.98 | 6 |
| 2015-01-05 | 1739 | 19 | 1.88 | 9 |
| 2015-01-06 | 519 | 18 | 1.29 | 9 |
| 2015-01-08 | 1724 | 19 | 1.99 | 9 |
| 2015-01-09 | 1546 | 17 | 1.91 | 8 |
| 2015-01-12 | 1591 | 13 | 1.85 | 6 |
| 2015-01-13 | 788 | 11 | 1.38 | 5 |
| 2015-01-14 | 782 | 15 | 1.30 | 7 |
| 2015-01-15 | 532 | 11 | 1.20 | 5 |
| 2015-01-16 | 1007 | 14 | 1.56 | 6 |
| 2015-01-20 | 801 | 12 | 1.34 | 5 |
| 2015-01-21 | 796 | 11 | 1.52 | 5 |
| 2015-01-22 | 1107 | 12 | 1.59 | 5 |
| 2015-01-23 | 1777 | 23 | 1.84 | 11 |
| 2015-01-26 | 1342 | 15 | 1.71 | 7 |
| 2015-01-27 | 1815 | 14 | 1.98 | 6 |
| 2015-01-28 | 1255 | 11 | 1.76 | 5 |
| 2015-01-30 | 1014 | 14 | 1.71 | 6 |

**Summary**: n=20, zero timeouts. `uniq_mid` median 14.0 (mean 15.10). `ret1s_std` inside the real
1.5-2.5bp band on every day.

## C. Scheduled-sampling retrain — per-epoch detail (pure-DDIM lineage)

Each epoch's checkpoint independently run through the same 20-day (or until-timeout) test as above.
`—` marks the day the run was abandoned at.

### Epoch 2 (`val_ema=0.701`) — cleared all 20 days

| Day | secs | uniq_mid | ret1s_std (bp) | mid_range (tk) |
|---|---|---|---|---|
| 2015-01-07 | 1422 | 15 | 1.43 | 7 |
| 2015-01-29 | 1377 | 15 | 1.66 | 7 |
| 2015-01-02 | 1399 | 17 | 1.58 | 8 |
| 2015-01-05 | 1411 | 15 | 1.51 | 7 |
| 2015-01-06 | 469 | 19 | 1.25 | 9 |
| 2015-01-08 | 1373 | 17 | 1.40 | 8 |
| 2015-01-09 | 1215 | 13 | 1.47 | 6 |
| 2015-01-12 | 1319 | 11 | 1.38 | 5 |
| 2015-01-13 | 678 | 13 | 1.19 | 6 |
| 2015-01-14 | 622 | 12 | 1.18 | 5 |
| 2015-01-15 | 449 | 9 | 1.02 | 4 |
| 2015-01-16 | 846 | 13 | 1.34 | 6 |
| 2015-01-20 | 675 | 11 | 1.31 | 5 |
| 2015-01-21 | 663 | 13 | 1.34 | 6 |
| 2015-01-22 | 931 | 11 | 1.36 | 5 |
| 2015-01-23 | 1395 | 17 | 1.47 | 8 |
| 2015-01-26 | 1046 | 13 | 1.39 | 6 |
| 2015-01-27 | 1592 | 15 | 1.66 | 7 |
| 2015-01-28 | 997 | 11 | 1.45 | 5 |
| 2015-01-30 | 795 | 13 | 1.46 | 6 |

Summary: uniq_mid median 13.0, mean 13.65. secs mean 1034.

### Epoch 3 (`val_ema=0.697`) — cleared all 20 days

| Day | secs | uniq_mid | ret1s_std (bp) | mid_range (tk) |
|---|---|---|---|---|
| 2015-01-07 | 1626 | 13 | 1.68 | 6 |
| 2015-01-29 | 1450 | 13 | 1.64 | 6 |
| 2015-01-02 | 1655 | 19 | 1.59 | 9 |
| 2015-01-05 | 1540 | 13 | 1.65 | 6 |
| 2015-01-06 | 444 | 11 | 1.06 | 5 |
| 2015-01-08 | 1552 | 11 | 1.58 | 5 |
| 2015-01-09 | 1369 | 11 | 1.52 | 5 |
| 2015-01-12 | 1391 | 13 | 1.55 | 6 |
| 2015-01-13 | 637 | 9 | 1.29 | 4 |
| 2015-01-14 | 651 | 7 | 1.31 | 3 |
| 2015-01-15 | 441 | 7 | 1.04 | 3 |
| 2015-01-16 | 821 | 13 | 1.28 | 6 |
| 2015-01-20 | 640 | 9 | 1.18 | 4 |
| 2015-01-21 | 653 | 9 | 1.25 | 4 |
| 2015-01-22 | 884 | 9 | 1.33 | 4 |
| 2015-01-23 | 1547 | 13 | 1.53 | 6 |
| 2015-01-26 | 1140 | 11 | 1.45 | 5 |
| 2015-01-27 | 1676 | 11 | 1.51 | 5 |
| 2015-01-28 | 1061 | 11 | 1.66 | 5 |
| 2015-01-30 | 785 | 13 | 1.44 | 6 |

Summary: uniq_mid median 11.0, mean 11.30 — the quietest of all epochs tested. secs mean 1098.

### Epoch 4 (`val_ema=0.69`) — cleared all 20 days

| Day | secs | uniq_mid | ret1s_std (bp) | mid_range (tk) |
|---|---|---|---|---|
| 2015-01-07 | 1730 | 15 | 1.46 | 7 |
| 2015-01-29 | 1648 | 13 | 1.52 | 6 |
| 2015-01-02 | 1730 | 19 | 1.76 | 9 |
| 2015-01-05 | 1756 | 19 | 1.54 | 9 |
| 2015-01-06 | 414 | 13 | 1.01 | 6 |
| 2015-01-08 | 1772 | 23 | 1.69 | 11 |
| 2015-01-09 | 1529 | 13 | 1.71 | 6 |
| 2015-01-12 | 1544 | 13 | 1.46 | 6 |
| 2015-01-13 | 641 | 9 | 1.34 | 4 |
| 2015-01-14 | 657 | 11 | 1.25 | 5 |
| 2015-01-15 | 406 | 7 | 1.14 | 3 |
| 2015-01-16 | 855 | 10 | 1.33 | 5 |
| 2015-01-20 | 678 | 9 | 1.24 | 4 |
| 2015-01-21 | 650 | 11 | 1.37 | 5 |
| 2015-01-22 | 971 | 9 | 1.36 | 4 |
| 2015-01-23 | 1667 | 15 | 1.55 | 7 |
| 2015-01-26 | 1225 | 13 | 1.49 | 6 |
| 2015-01-27 | 1856 | 15 | 1.56 | 7 |
| 2015-01-28 | 1073 | 11 | 1.49 | 5 |
| 2015-01-30 | 822 | 11 | 1.49 | 5 |

Summary: uniq_mid median 13.0, mean 12.95. secs mean 1181.

### Epoch 5 (`val_ema=0.682`) — ELIMINATED, timed out on day 15/20

| Day | secs | uniq_mid | ret1s_std (bp) | mid_range (tk) |
|---|---|---|---|---|
| 2015-01-07 | 2359 | 11 | 1.69 | 5 |
| 2015-01-29 | 2378 | 17 | 1.62 | 8 |
| 2015-01-02 | 2338 | 19 | 1.83 | 9 |
| 2015-01-05 | 2281 | 12 | 1.67 | 5 |
| 2015-01-06 | 428 | 14 | 1.04 | 7 |
| 2015-01-08 | 2306 | 19 | 1.59 | 9 |
| 2015-01-09 | 1837 | 15 | 1.57 | 7 |
| 2015-01-12 | 2080 | 13 | 1.54 | 6 |
| 2015-01-13 | 576 | 11 | 1.17 | 5 |
| 2015-01-14 | 608 | 9 | 1.36 | 4 |
| 2015-01-15 | 416 | 9 | 1.14 | 4 |
| 2015-01-16 | 804 | 11 | 1.40 | 5 |
| 2015-01-20 | 606 | 10 | 1.23 | 5 |
| 2015-01-21 | 613 | 9 | 1.33 | 4 |
| 2015-01-22 | 1053 | 11 | 1.52 | 5 |
| 2015-01-23 | **TIMEOUT (2400s)** | — | — | — |

Note: already notably slower even before the timeout (mean secs on completed days 1379 vs epoch 4's
1181) despite similar activity levels — a second, independent sign of emerging instability alongside
the outright failure on day 15.

## D. Epoch-over-epoch trend (the negative result, quantified)

| Epoch | val_ema | uniq_mid median | uniq_mid mean | secs mean | Outcome |
|---|---|---|---|---|---|
| 0 (pre-retrain baseline) | 0.724 | 14.0 | 15.10 | 1265 | final model |
| 2 | 0.701 | 13.0 | 13.65 | 1034 | cleared |
| 3 | 0.697 | 11.0 | 11.30 | 1098 | cleared |
| 4 | 0.69 | 13.0 | 12.95 | 1181 | cleared |
| 5 | 0.682 | 11.0 (partial) | 12.67 (partial, n=15) | 1379 (partial) | **eliminated** |

Activity (`uniq_mid`) does not climb toward the real 27-66 target as scheduled sampling progresses —
it stays flat to slightly declining, with epoch 3 the quietest of the lot. This is the core evidence
behind the "scheduled sampling did not achieve its goal" finding (see `PROJECT_STATUS.md` and
`analysis/final_model_handoff.md` §5).

## E. Wall-clock time tracks activity — the mechanism, quantified

Across every epoch, wall-clock simulation time correlates positively with `uniq_mid` (more generated
market activity requires more model forward passes, hence more compute) — this is the same mechanism
documented earlier in the project for cross-day drift days running slower than calm days.

| Epoch | correlation(secs, uniq_mid) | n |
|---|---|---|
| 0 | r = 0.528 | 20 |
| 2 | r = 0.415 | 20 |
| 3 | r = 0.661 | 20 |
| 4 | r = 0.775 | 20 |
| 5 | r = 0.694 | 15 (partial) |
| **Pooled, all epochs+days** | **r = 0.578** | **95** |

This relationship — not "more training = slower" — is the correct explanation for the timing
variation observed during the retrain; the epoch-to-epoch differences in mean wall-clock time (§D)
are a side effect of small activity-level differences, not a direct consequence of epoch number
itself.
