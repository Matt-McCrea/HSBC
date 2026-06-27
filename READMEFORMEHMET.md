# Setup Guide — DeepMarket / TRADES (project fork)
Written by CLAUDE 

Welcome! This repo is our fork of [DeepMarket](https://github.com/LeonardoBerti00/DeepMarket)
(the TRADES paper codebase), with a batch of fixes already applied so it runs on a
**modern Python stack** (pandas 3.x / numpy 2.x / matplotlib 3.x). You shouldn't have to
fight the deprecation errors that the original repo throws — those are done.

A few things are **deliberately not in the repo** (too large, or licensed) and you'll
grab them separately. This guide walks through the whole thing. Should take ~20 minutes
plus download time.

---

## 1. Prerequisites

- **Python 3.11+** (3.11–3.14 all fine).
- **git** and — importantly — **git-lfs**. The repo stores model checkpoints and dataset
  CSVs via Git LFS; without it they'll clone as tiny text "pointer" files and nothing works.
  - macOS: `brew install git-lfs`
  - Ubuntu/Debian: `sudo apt install git-lfs`
  - Windows: included with Git for Windows, or from git-lfs.com
  - Then once, on your machine: `git lfs install`

## 2. Clone (with LFS)

```sh
git lfs install
git clone https://github.com/Matt-McCrea/HSBC.git
cd HSBC
git lfs pull          # pulls the actual .ckpt / .csv files, not pointers
```

Quick check it worked: `ls -la data/checkpoints/CGAN/` should show `.ckpt` files that are
**megabytes**, not a few hundred bytes. If they're tiny, run `git lfs pull` again.

## 3. Environment

```sh
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

⚠️ **If you're on a Mac (or any machine without an NVIDIA GPU):** the `torch` and
`torchvision` lines in `requirements.txt` point at a CUDA build (`--index-url …/cu118`)
that has no Mac wheel, so the install will fail. Fix: install torch first from the default
index, then the rest —

```sh
pip install torch torchvision            # gets the right CPU/MPS build for your machine
# then delete the two "--index-url .../cu118" torch lines from requirements.txt, and:
pip install -r requirements.txt
```

(On a Linux/Windows box with an NVIDIA GPU, leave `requirements.txt` as-is.)

## 4. Download what isn't in the repo

**Model checkpoints (TRADES).** Not committed — download from the authors' Google Drive
([link](https://drive.google.com/drive/folders/1fg5G9KzmzC6E4FUYSCjObJ7sCEdjo43W?usp=sharing))
and place them in `data/checkpoints/TRADES/`. There's one trained on TSLA and one on INTC.
(`gdown` is already in requirements if you prefer downloading from the command line.)
The CGAN checkpoints are already in the repo via LFS.

**Market data (LOBSTER).** Not committed — it's licensed, so we can't redistribute our copy.
Two options:
- *Just to get it running:* use the free LOBSTER sample (INTC, 2012-06-21) from
  <https://lobsterdata.com/info/DataSamples.php>. The model wasn't trained on this, so it's
  for smoke-testing the pipeline, not for results.
- *To reproduce paper results:* you need the licensed TSLA/INTC January 2015 data (see the
  paper for sourcing). Save it in `data/{stock}/...` in LOBSTER format — there's an example
  layout under `data/INTC/`.

## 5. Smoke test — run a sim and make plots

Run a short TRADES market simulation (no-LOBSTER-data path, uses the sample):

```sh
python -u ABIDES/abides.py -c world_agent_sim -t INTC -date 2012-06-21 \
    -d True -m TRADES -st '09:30:00' -et '12:00:00' -id 2.317
```

This writes a log folder under `ABIDES/log/world_agent_INTC_...`.

**Important gotcha (this one cost us a day):** the stylized-facts plotting script does **not**
read the sim's `.bz2` output directly — you must first convert it to an `orders_*.pkl`, then
point the plotter at *that folder*. Otherwise you get blank plots with no error. So:

```sh
LOG="ABIDES/log/<the_folder_the_sim_just_made>"
mkdir -p "$LOG/pkl"
PYTHONPATH=ABIDES python3 ABIDES/util/formatting/convert_order_stream.py \
    "$LOG/EXCHANGE_AGENT.bz2" INTC 10 plot-scripts -o "$LOG/pkl"

cd ABIDES
PYTHONPATH=. python3 realism/order_flow_stylized_facts.py "../$LOG/pkl" -o ../visualizations -z
```

Plots land in `visualizations/`. The `-z` flag forces a recompute so you never get served a
stale empty cache.

---

## Good to know

- **Deprecation fixes are already done** across the plotting, realism, and preprocessing
  code (the `Series.append`, `np.NaN`, `fillna(method=)`, `resample('T'/'S')`, `get_cmap`
  issues). If you hit a *new* one, shout — there's a known list of where they lived.
- **One fix is still pending:** the multi-line `.append` in
  `ABIDES/realism/market_impact/marketreplay_market_impact.py`. It only bites if you run the
  **market-impact / responsiveness** experiments, so it's safe to ignore until then.
- **Don't commit** your `env/` virtualenv, the LOBSTER data, the downloaded checkpoints, or
  the `cache/`/`visualizations/` outputs — `.gitignore` already excludes them. (And please
  don't add our licensed market data to git.)
- **Terminal tip:** pasting big multi-line blocks into macOS Terminal can scramble them.
  If a command looks mangled, run it line by line.

Questions → just ping me.
