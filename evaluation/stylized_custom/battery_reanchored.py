"""
Stylized-facts battery: Real vs generated reanchored-checkpoint configs.

Real is derived from the local LOBSTER message+orderbook (row-aligned; prices in 1/10000 $).
Everything is sliced to the pure-generation window 09:45-10:00 and, for return-based panels,
resampled to 1-second bars so the series are comparable despite very different event counts.

Panels (3x2):  mid-price trace | standardised log-return distribution | returns ACF
               |abs|-returns ACF (vol clustering) | order-size histogram | spread histogram

Run:  python evaluation/stylized_custom/battery_reanchored.py
Out:  analysis/plots/stylized_battery_reanchored.png
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATE = "2015-01-30"
WIN_LO, WIN_HI = "09:45:00", "10:00:00"           # pure-generation window
SENT = 9e9                                        # sentinel magnitude
MAX_LAG = 30
CSV_DIR = "reanchored_csvs"
LOB_DIR = "data/INTC/INTC_2015-01-02_2015-01-30_10"
REAL_OB = f"{LOB_DIR}/INTC_2015-01-30_34140000_57660000_orderbook_10.csv"
REAL_MSG = f"{LOB_DIR}/INTC_2015-01-30_34140000_57660000_message_10.csv"

# Fallback runs used only when --series is not passed (the shells always pass --series explicitly).
# Named by the run-name SUFFIX ("end"), so they resolve whether the runs live as
# ABIDES/log/<run>/processed_orders.csv (on the GPU box) or as exported reanchored_csvs/<run>.csv
# (locally). Any suffix with no matching run is skipped with a warning, so the default never crashes.
DEFAULT_RUNS = [
    ("baseline (no noise)", "tdprior_sr"),
    ("dn0.3",               "tdprior_sr_dn0.3"),
    ("dn0.5",               "tdprior_sr_dn0.5"),
    ("dn0.6",               "tdprior_sr_dn0.6"),
    ("dn0.3 te0.045",       "tdprior_sr_dn0.3_te0.045"),
]


def _slice(df):
    lo = pd.Timestamp(f"{DATE} {WIN_LO}"); hi = pd.Timestamp(f"{DATE} {WIN_HI}")
    return df[(df["dt"] >= lo) & (df["dt"] < hi)].copy()


def _resolve_gen_path(path):
    """Accept any of the layouts a run can produce: a processed_orders.csv, a run DIRECTORY (->
    <dir>/processed_orders.csv), or a flat exported CSV. Returns the resolved file, or None."""
    if os.path.isdir(path):
        cand = os.path.join(path, "processed_orders.csv")
        return cand if os.path.isfile(cand) else None
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "processed_orders.csv")   # path given without its processed_orders leaf
    return cand if os.path.isfile(cand) else None


def _resolve_run_suffix(suffix):
    """Find a run whose name ENDS with `suffix`, as an ABIDES/log run dir (-> its processed_orders.csv)
    or an exported reanchored_csvs/<run>.csv. Returns the CSV path, or None."""
    for d in sorted(glob.glob(f"ABIDES/log/*{suffix}")):
        if os.path.basename(d.rstrip("/")).endswith(suffix):
            cand = os.path.join(d, "processed_orders.csv")
            if os.path.isfile(cand):
                return cand
    for f in sorted(glob.glob(os.path.join(CSV_DIR, f"*{suffix}.csv"))):
        if os.path.basename(f).endswith(suffix + ".csv"):
            return f
    return None


def load_gen(path):
    resolved = _resolve_gen_path(path)
    if resolved is None:
        raise FileNotFoundError(f"no CSV at {path} (nor {path}/processed_orders.csv)")
    df = pd.read_csv(resolved)
    df["dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    for c in ("ask_price_1", "bid_price_1", "SIZE"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["ask_price_1"].abs() < SENT) & (df["bid_price_1"].abs() < SENT)]
    df = df[(df["ask_price_1"] > 0) & (df["bid_price_1"] > 0)]
    df["mid"] = (df["ask_price_1"] + df["bid_price_1"]) / 2 / 10000.0   # -> dollars
    df["spread_tk"] = (df["ask_price_1"] - df["bid_price_1"]) / 100.0    # -> ticks
    df["vol"] = df["SIZE"].abs()
    return _slice(df.dropna(subset=["dt", "mid"]))


def load_real():
    ob = pd.read_csv(REAL_OB, header=None, usecols=[0, 2], names=["ask_p1", "bid_p1"])
    msg = pd.read_csv(REAL_MSG, header=None, usecols=[0, 3], names=["t", "size"])
    n = min(len(ob), len(msg)); ob, msg = ob.iloc[:n], msg.iloc[:n]
    df = pd.DataFrame({
        "dt": pd.Timestamp(DATE) + pd.to_timedelta(msg["t"].values, unit="s"),
        "mid": (ob["ask_p1"].values + ob["bid_p1"].values) / 2 / 10000.0,
        "spread_tk": (ob["ask_p1"].values - ob["bid_p1"].values) / 100.0,
        "vol": np.abs(msg["size"].values),
    })
    df = df[(df["mid"] > 0) & (df["spread_tk"] >= 0)]
    return _slice(df)


def bars_1s(df):
    s = df.set_index("dt")["mid"].resample("1s").last().ffill().dropna()
    r = np.log(s).diff().dropna()
    return r.values


def acf(x, max_lag=MAX_LAG):
    x = pd.Series(x)
    return np.array([x.autocorr(lag=l) for l in range(1, max_lag + 1)])


def _resolve_real(lob_dir, date):
    """Find the LOBSTER orderbook + message CSVs for a date. Tries the given directory first, then
    falls back to a recursive search under it, its parent, and data/ — so the exact folder name
    (whether it ends in _10 or not) does not matter."""
    def _find(base, recursive):
        pat_ob = os.path.join(base, "**", f"*{date}*orderbook*.csv") if recursive else os.path.join(base, f"*{date}*orderbook*.csv")
        pat_ms = os.path.join(base, "**", f"*{date}*message*.csv") if recursive else os.path.join(base, f"*{date}*message*.csv")
        return sorted(glob.glob(pat_ob, recursive=recursive)), sorted(glob.glob(pat_ms, recursive=recursive))
    ob, msg = _find(lob_dir, False)
    if not ob or not msg:
        for base in (lob_dir, os.path.dirname(lob_dir.rstrip("/")) or ".", "data"):
            ob, msg = _find(base, True)
            if ob and msg:
                break
    if not ob or not msg:
        raise FileNotFoundError(f"no LOBSTER orderbook/message CSV for {date} under {lob_dir} (or data/)")
    return ob[0], msg[0]


def main():
    global DATE, LOB_DIR, REAL_OB, REAL_MSG, WIN_LO, WIN_HI
    ap = argparse.ArgumentParser(description="Stylized-facts battery: Real vs generated configs.")
    ap.add_argument("--series", action="append", default=[], metavar="LABEL=CSV",
                    help="Generated config as LABEL=path (repeatable). The path may be a "
                         "processed_orders.csv, a run directory (its processed_orders.csv is used), or a "
                         "flat CSV. Missing paths are skipped with a warning. If omitted, resolves the "
                         "built-in DEFAULT_RUNS by run-name suffix.")
    ap.add_argument("--date", default=DATE, help="Trading day YYYY-MM-DD (default 2015-01-30).")
    ap.add_argument("--lob-dir", default=LOB_DIR, help="Directory holding the LOBSTER ob/msg CSVs.")
    ap.add_argument("--win-lo", default=WIN_LO, help="Window start HH:MM:SS (default 09:45:00).")
    ap.add_argument("--win-hi", default=WIN_HI, help="Window end HH:MM:SS (default 10:00:00).")
    ap.add_argument("--out", default="analysis/plots/stylized_battery_reanchored.png",
                    help="Output PNG path.")
    ap.add_argument("--title", default=None, help="Figure suptitle override.")
    args = ap.parse_args()

    DATE, LOB_DIR, WIN_LO, WIN_HI = args.date, args.lob_dir, args.win_lo, args.win_hi
    REAL_OB, REAL_MSG = _resolve_real(LOB_DIR, DATE)

    # Okabe-Ito categorical order (colourblind-safe), Real always first as neutral dark.
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    if args.series:
        raw = []
        for i, spec in enumerate(args.series):
            if "=" not in spec:
                raise SystemExit(f"--series expects LABEL=CSV, got: {spec}")
            label, path = spec.split("=", 1)
            raw.append((label.strip(), palette[i % len(palette)], path.strip()))
    else:
        raw = [(label, palette[i % len(palette)], _resolve_run_suffix(suffix) or suffix)
               for i, (label, suffix) in enumerate(DEFAULT_RUNS)]

    # Resolve each generated path; skip (with a warning) any that is absent so a partial sweep plots.
    series = [("Real", "#222222", None)]
    for label, color, path in raw:
        if _resolve_gen_path(path) is None:
            print(f"  WARN: skipping '{label}' — no CSV at {path} (nor {path}/processed_orders.csv)")
            continue
        series.append((label, color, path))
    if len(series) < 2:
        raise SystemExit("no generated series resolved — pass --series LABEL=path (a processed_orders.csv, "
                         "a run directory, or a flat CSV)")

    data = {}
    for name, color, path in series:
        df = load_real() if name == "Real" else load_gen(path)
        r = bars_1s(df)
        data[name] = dict(color=color, df=df, ret=r)
        print(f"{name:6s}: events={len(df):6d}  1s-returns={len(r):4d}  "
              f"mid {df['mid'].min():.3f}-{df['mid'].max():.3f}")

    os.makedirs("analysis/plots", exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    lags = np.arange(1, MAX_LAG + 1)

    # (1) mid-price trace (indexed to 100 at window open, so shapes compare)
    for name, d in data.items():
        s = d["df"].set_index("dt")["mid"].resample("1s").last().ffill()
        ax[0, 0].plot(s.index, 100 * s / s.iloc[0], color=d["color"], lw=1.5, label=name)
    ax[0, 0].set_title("Mid-price trace (indexed to 100 at 09:45)")
    ax[0, 0].set_ylabel("indexed mid"); ax[0, 0].legend(frameon=False)

    # (2) standardised log-return distribution vs N(0,1)
    xs = np.linspace(-6, 6, 200)
    for name, d in data.items():
        r = d["ret"]; sd = r.std()
        if sd > 0:
            ax[0, 1].hist(r / sd, bins=60, range=(-6, 6), density=True, histtype="step",
                          color=d["color"], lw=1.6, label=name)
    ax[0, 1].plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color="0.5", ls="--", lw=1.2, label="N(0,1)")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Standardised 1s log-return density")
    ax[0, 1].set_xlabel("return / std"); ax[0, 1].legend(frameon=False)

    # (3) returns ACF
    for name, d in data.items():
        ax[0, 2].plot(lags, acf(d["ret"]), color=d["color"], lw=1.5, marker="o", ms=3, label=name)
    ax[0, 2].axhline(0, color="0.6", lw=0.8)
    ax[0, 2].set_title("Log-return autocorrelation"); ax[0, 2].set_xlabel("lag (s)"); ax[0, 2].legend(frameon=False)

    # (4) |returns| ACF -> volatility clustering
    for name, d in data.items():
        ax[1, 0].plot(lags, acf(np.abs(d["ret"])), color=d["color"], lw=1.5, marker="o", ms=3, label=name)
    ax[1, 0].axhline(0, color="0.6", lw=0.8)
    ax[1, 0].set_title("|log-return| autocorrelation (vol clustering)"); ax[1, 0].set_xlabel("lag (s)"); ax[1, 0].legend(frameon=False)

    # (5) order-size histogram (log-x)
    allv = np.concatenate([data[n]["df"]["vol"].values for n in data])
    allv = allv[(allv > 0) & (allv < np.percentile(allv, 99.5))]
    bins = np.logspace(np.log10(max(allv.min(), 1)), np.log10(allv.max()), 40)
    for name, d in data.items():
        v = d["df"]["vol"].values; v = v[v > 0]
        ax[1, 1].hist(v, bins=bins, density=True, histtype="step", color=d["color"], lw=1.6, label=name)
    ax[1, 1].set_xscale("log"); ax[1, 1].set_title("Order-size distribution"); ax[1, 1].set_xlabel("size (shares)"); ax[1, 1].legend(frameon=False)

    # (6) spread histogram (ticks)
    for name, d in data.items():
        sp = d["df"]["spread_tk"].values; sp = sp[(sp >= 0) & (sp <= 15)]
        ax[1, 2].hist(sp, bins=np.arange(0, 16) - 0.5, density=True, histtype="step", color=d["color"], lw=1.6, label=name)
    ax[1, 2].set_title("Spread distribution"); ax[1, 2].set_xlabel("spread (ticks)"); ax[1, 2].legend(frameon=False)

    title = args.title or f"Stylized-facts battery — INTC {DATE}, {WIN_LO[:5]}–{WIN_HI[:5]} (Real vs configs)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
