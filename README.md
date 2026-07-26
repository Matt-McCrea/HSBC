# TRADES — Diffusion-based Limit Order Book Simulation

A transformer-diffusion generative "world model" that simulates a full limit-order-book (LOB)
trading session order-by-order, inside an [ABIDES](https://github.com/abides-sim/abides) market.
Based on the paper [TRADES: Generating Realistic Market Simulations with Diffusion Models](https://arxiv.org/abs/2502.07071)
(Berti, Prenkaj, Velardi, 2025); this fork adds sampler-acceleration work and diagnostics
(see [`analysis/`](analysis/)).

---

## Quickstart (recommended)

Everything runs through one script — it creates a local `./env` virtualenv and calls it directly,
so **you never need to activate anything**. Run the steps in order:

```sh
git clone <this-repo-url> && cd <repo>

bash scripts/quickstart.sh setup                    # 1. create ./env, install dependencies
bash scripts/quickstart.sh data path/to/INTC.zip    # 2. unpack your LOBSTER data into data/INTC/
bash scripts/quickstart.sh train                     # 3. preprocess + train a TRADES checkpoint
bash scripts/quickstart.sh simulate                  # 4. simulate a session (DDPM, 100 steps)
```

Defaults target **INTC, 2015-01-30, 09:30–10:30** — which match the repo's config defaults, so for
INTC 2015 data you don't need to edit any config files. The simulation writes an orders CSV and the
paper's stylized-fact plots to `ABIDES/log/world_agent_INTC_…/`.

> **GPU strongly recommended.** Training on CPU is impractical (15M-row dataset); simulation runs on
> CPU but slowly. Check with `env/bin/python -c "import torch; print(torch.cuda.is_available())"`.
> `requirements.txt` pins a CUDA 11.8 torch build — for other CUDA/CPU, reinstall torch for your
> platform (the `setup` step prints how).

### Simulate with a different sampler

```sh
bash scripts/quickstart.sh simulate DDPM 100     # recommended: stochastic, realistic
bash scripts/quickstart.sh simulate DDIM 10      # fast/deterministic (see analysis/ for caveats)
bash scripts/quickstart.sh simulate DDPM 100 0.681   # 3rd arg = specific checkpoint val-loss id
```

If no checkpoint id is given, the best (lowest val-loss) checkpoint in `data/checkpoints/TRADES/`
is used automatically.

---

## Data format

LOBSTER-format CSVs (message + orderbook), placed at:

```
data/INTC/INTC_2015-01-02_2015-01-30/
    2015-01-30_34200000_57600000_message_10.csv
    2015-01-30_34200000_57600000_orderbook_10.csv
    …one pair per trading day…
```

The folder name is `data/{STOCK}/{STOCK}_{first-day}_{last-day}`; the date range must match
`DATE_TRADING_DAYS` in `constants.py` (default `2015-01-02 … 2015-01-30`). `quickstart.sh data`
unzips into `data/INTC/` for you; just check the resulting path matches the above.

To use a **different stock or date range**: add the stock to `cst.Stocks` in `constants.py`, set
`DATE_TRADING_DAYS`, and set `CHOSEN_STOCK` in `configuration.py`.

---

## Manual steps (equivalent to the script)

**Setup**
```sh
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

**Train** — `configuration.py` controls the run. Key flags:
- `IS_DATA_PREPROCESSED` — `False` (default) preprocesses the raw LOBSTER data into
  `data/INTC/{train,val,test}.npy` before training; set `True` to reuse existing `.npy` and skip it.
- `CHOSEN_MODEL = cst.Models.TRADES`, `CHOSEN_STOCK = [cst.Stocks.INTC]`, `EPOCHS = 50`.
```sh
python main.py                       # preprocess (first run) + train
```
The best checkpoint is saved to `data/checkpoints/TRADES/val_ema=<loss>_epoch=<n>_…ckpt`.

**Simulate**
```sh
python -u ABIDES/abides.py -c world_agent_sim -t INTC -date 20150130 \
    -st 09:30:00 -et 10:30:00 -d True -m TRADES -type DDPM -nsteps 100
```

---

## What this fork adds

- **Extra samplers** in `models/diffusers/gaussian_diffusion.py`: `DPM_SOLVER`, `DPM_SOLVER_PP`,
  `UNIPC`, and the hybrids `HYBRID_PP_DDPM` / `HYBRID_DDPM_PP` (pass via `-type`).
- **Diagnostics & flag-gated experiments** in `ABIDES/agent/WorldAgent.py` (all default-off; see
  `ABIDES/config/world_agent_sim.py --help`), e.g. `--type-decode prior`, `--depth-temp`,
  `--guidance-scale`, and an always-on end-of-run diagnostics block.
- **Evaluation tools**: `evaluation/quantitative_eval/flow_mix.py` (order-flow vs a real CSV) and
  `evaluation/diagnostics/open_loop_eval.py` (sample the model on real windows, no ABIDES loop).
- **Findings & figures** in [`analysis/`](analysis/) — `hsbc_call_brief.tex`,
  `deterministic_sampler_findings.md`, and `figures/`.

Headline result: **DDPM (100 steps) reproduces a realistic, moving market; deterministic
fast samplers (few-step DDIM / DPM-Solver) collapse it**. Full reasoning in `analysis/`.

---

## Other configurations (from the original repo)

- **CGAN** instead of TRADES: add `-m CGAN` to the simulate command.
- **IABS agent-based baseline** (no generative model):
  `python -u ABIDES/abides.py -c rsmc_03 -date 20150130 -st 09:30:00 -et 12:00:00`
- **Author checkpoints / TRADES-LOB dataset**: see the original repo,
  [LeonardoBerti00/DeepMarket](https://github.com/LeonardoBerti00/DeepMarket). The released
  `TRADES-LOB` synthetic dataset (INTC/TSLA, 29–30 Jan 2015) is used here as a realism benchmark.

---

## Citation

```bibtex
@article{berti2025trades,
  title={TRADES: Generating Realistic Market Simulations with Diffusion Models},
  author={Berti, Leonardo and Prenkaj, Bardh and Velardi, Paola},
  journal={arXiv preprint arXiv:2502.07071},
  year={2025}
}
```
