# LOB-Bench integration

Score a TRADES simulation against real market data with
[LOB-Bench](https://github.com/peernagy/lob_bench) — a published, standardized benchmark that
computes Wasserstein-1 / L1 distances on LOB distributions (spread, depth, imbalance,
interarrival, cancel depth, …) and **price-impact response curves**.

## Install
```sh
pip install -r requirements-eval.txt
```

## Run
Both inputs are `processed_orders.csv` files (the `world_agent_sim` output format) — one real
market-replay, one generated simulation:
```sh
python evaluation/lob_bench/run_lob_bench.py \
    --real ABIDES/log/market_replay_INTC_2015-01-30_10-00-00_30/processed_orders.csv \
    --gen  ABIDES/log/world_agent_INTC_..._DDPM_.../processed_orders.csv \
    --out-dir lob_bench_run --window 09:45
```
- `--real` takes a processed_orders.csv (converted); **or** use `--real-lobster` (below).
- `--window 09:45` drops the replay warm-up (keeps the generation phase).
- `--real-lobster <file>` — use a **raw LOBSTER** message/orderbook file directly for the real
  side (no conversion; it's already in LOB-Bench format). This is the natural real source and is
  auto-sliced to the generated window. Give it either the `_message_` or `_orderbook_` file:
  ```sh
  python evaluation/lob_bench/run_lob_bench.py \
      --real-lobster data/INTC/INTC_2015-01-02_2015-01-30_10/INTC_2015-01-30_34140000_57660000_orderbook_10.csv \
      --gen sweep_results/DDPM_100/DDPM_100_generated_orders.csv \
      --out-dir lob_bench_run --window 09:45
  ```

- `--n-splits N` chops the session into N contiguous sequences (restores bootstrapped error
  bars; default 1 = one whole-session sequence, point estimates only).
- `--prepare-only` writes the LOBSTER folders and stops (guaranteed to work) so you can call the
  `lob_bench` API yourself.

## Pieces
- `to_lobster.py` — converts `processed_orders.csv` → LOBSTER message+orderbook CSVs
  (headerless: `time, event_type, order_id, size, price, direction`; 40-col orderbook; price
  ×10000; event types LIMIT→1 / CANCEL→3 / EXEC→4). Run standalone with `--self-test`.
- `run_lob_bench.py` — converts real+gen, lays out the folders `Simple_Loader` expects, calls
  the benchmark.

## Known first-run adjustments
Two things depend on `lob_bench`'s exact API and may need a one-line tweak on first run
(both clearly marked in `run_lob_bench.py`):
1. **Filename convention** — reverse-engineered from `Simple_Loader`'s globs; if it fails to
   pair files, adjust the `real_name` / `gen_name` lambdas.
2. **`score_cfg` / `metric_cfg`** — their structure is defined by `lob_bench`; copy from its
   README/example and pass into `scoring.run_benchmark`. The runner stops with a clear message
   rather than guessing.

Tip: run it on the authors' released `data/TRADES-LOB` (their own DDPM output) as a converter
sanity check and to get reference numbers — note those files ship a truncated timestamp index,
so time-based scores need a full-datetime source (any `world_agent_sim` output).
