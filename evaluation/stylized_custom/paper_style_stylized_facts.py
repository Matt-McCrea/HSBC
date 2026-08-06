import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MAX_LAG = 30

# Bar interval the series is resampled to before returns/volatility are computed. Lags are then
# expressed in THESE units, so "1min" + MAX_LAG=30 reproduces the TRADES paper's "Lag (minutes)"
# 0-30 axis and makes our panels directly comparable to theirs.
#
# Why not 1s: on a 75-120 minute session, 1s bars give 4500-7200 points, so a lag of 1-30 removes
# under 1% of the series and the correlation is essentially unchanged across lags --- the panels
# come out flat (measured: vol-volatility varies by 0.0027 across lags 1-30 on the released TRADES
# output at 1s). At 1min the same lag range spans a quarter to half the session and the structure
# is visible. Override with --resample for a different convention.
RESAMPLE = "1min"

# The mid-price trace panel keeps second resolution regardless of RESAMPLE --- it is a time
# series, not a lag statistic, and minute bars smooth away short sharp events.
TRACE_RESAMPLE = "1s"


def load_processed(path, label, resample=None):
    """resample: pandas offset alias (e.g. '1s') to bar the raw event stream onto a fixed time
    grid before computing returns, or None to keep the old raw-event-level behaviour.

    Raw per-event returns are the wrong sampling for lag-based stylized facts here: a LOB session
    has hundreds of thousands of events, so a lag of 1-30 EVENTS overlaps 99.99%+ of the series and
    any lag structure is invisible (checked empirically: correlation varies by ~1e-4 across lags
    1-30 on raw events, indistinguishable from a flat line at any plot scale). Resampling to a fixed
    time interval first — the same convention used elsewhere in this project for ret1s_std — makes
    a lag of 1-30 a meaningful fraction of a 30-minute window (30-1800 bars) instead of a rounding
    error against 400k+ rows."""
    resample = RESAMPLE if resample is None else resample
    df = pd.read_csv(path)

    # Time column
    if "Unnamed: 0" in df.columns:
        df["datetime"] = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["datetime"] = pd.RangeIndex(len(df))
        resample = None  # no real timestamps to resample against

    # Mid price
    if "MID_PRICE" in df.columns:
        df["mid"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
    elif "ask_price_1" in df.columns and "bid_price_1" in df.columns:
        ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
        bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
        df["mid"] = (ask + bid) / 2
    else:
        raise ValueError(f"No MID_PRICE or ask/bid columns found in {path}")

    # Volume
    if "SIZE" in df.columns:
        df["volume"] = pd.to_numeric(df["SIZE"], errors="coerce").abs()
    elif "ORDER_SIZE" in df.columns:
        df["volume"] = pd.to_numeric(df["ORDER_SIZE"], errors="coerce").abs()
    elif "QUANTITY" in df.columns:
        df["volume"] = pd.to_numeric(df["QUANTITY"], errors="coerce").abs()
    else:
        df["volume"] = 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mid", "datetime"])
    df = df[df["mid"] > 0].copy()

    if resample:
        g = df.set_index("datetime").sort_index()
        mid = g["mid"].resample(resample).last().ffill()
        vol = g["volume"].resample(resample).sum()
        df = pd.DataFrame({"mid": mid, "volume": vol}).dropna(subset=["mid"]).reset_index()

    df["label"] = label
    df["log_mid"] = np.log(df["mid"])
    df["log_return"] = df["log_mid"].diff()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["log_return"])

    return df


def autocorr(x, max_lag=30):
    x = pd.Series(x).dropna()
    if len(x) < max_lag + 5:
        return np.full(max_lag, np.nan)

    values = []
    for lag in range(1, max_lag + 1):
        values.append(x.autocorr(lag=lag))
    return np.array(values)


def rolling_features(df, window=None):
    # window is in BARS. At 1min bars a 50-bar window would eat most of a 75-120 min
    # session, so scale it down; at 1s keep the original 50.
    if window is None:
        window = 5 if RESAMPLE.endswith(("min", "T")) else 50
    out = df.copy()
    out["abs_return"] = out["log_return"].abs()
    out["volatility"] = out["log_return"].rolling(window).std()
    out["rolling_volume"] = out["volume"].rolling(window).sum()
    return out.dropna(subset=["volatility", "rolling_volume"])


def corr_by_lag(x, y, max_lag=30):
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    vals = []
    for lag in range(1, max_lag + 1):
        if len(x) <= lag + 5:
            vals.append(np.nan)
        else:
            vals.append(x.iloc[:-lag].corr(y.iloc[lag:]))
    return np.array(vals)


def main(real_path, gen_path, out_path, real_label="Real", gen_label="TRADES"):
    global LAGLBL
    LAGLBL = "Lag (minutes)" if RESAMPLE.endswith(("min","T")) else f"Lag ({RESAMPLE} bars)"
    real = load_processed(real_path, real_label)
    gen = load_processed(gen_path, gen_label)

    print(real_label, "rows:", len(real))
    print(gen_label, "rows:", len(gen))

    if len(real) < 50 or len(gen) < 50:
        print("WARNING: very few rows; plots will be noisy.")

    datasets = [(real_label, real), (gen_label, gen)]

    # Cap the lag range to the session length. At 1-minute bars a 30-minute session yields only
    # ~30 bars, so a lag of 30 would compare a series against almost nothing --- autocorr() would
    # return NaN outright and corr_by_lag() would rest on a handful of overlapping points. Allow a
    # lag of at most ~40% of the available bars, which keeps every estimate on a usable sample.
    n_bars = min(len(real), len(gen))
    max_lag = max(3, min(MAX_LAG, int(0.4 * n_bars)))
    if max_lag < MAX_LAG:
        print(f"note: {n_bars} {RESAMPLE} bars available -> lag range capped at {max_lag} "
              f"(a {MAX_LAG}-lag range would exceed the session)")
    lags = np.arange(1, max_lag + 1)

    # The mid-price trace is a time series, not a lag statistic, so it keeps SECOND resolution
    # regardless of the bar size used for the correlation panels. At 1-minute bars a 75-minute
    # session collapses to ~75 points and short, sharp events -- the single-step price collapse
    # among them -- are smoothed away entirely.
    fine = [(real_label, load_processed(real_path, real_label, resample=TRACE_RESAMPLE)),
            (gen_label, load_processed(gen_path, gen_label, resample=TRACE_RESAMPLE))]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    # 1. Log-return autocorrelation
    for label, df in datasets:
        axes[0].plot(lags, autocorr(df["log_return"], max_lag), marker="o", linewidth=1, label=label)
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set_title("Log returns autocorrelation")
    axes[0].set_xlabel(LAGLBL)
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend()

    # 2. Correlation between volume and volatility
    for label, df in datasets:
        f = rolling_features(df)
        axes[1].plot(lags, corr_by_lag(f["rolling_volume"], f["volatility"], max_lag),
                     marker="o", linewidth=1, label=label)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_title("Correlation between volume and volatility")
    axes[1].set_xlabel(LAGLBL)
    axes[1].set_ylabel("Correlation")
    axes[1].legend()

    # 3. Correlation between returns and volatility
    for label, df in datasets:
        f = rolling_features(df)
        axes[2].plot(lags, corr_by_lag(f["log_return"], f["volatility"], max_lag),
                     marker="o", linewidth=1, label=label)
    axes[2].axhline(0, linewidth=0.8)
    axes[2].set_title("Correlation between returns and volatility")
    axes[2].set_xlabel(LAGLBL)
    axes[2].set_ylabel("Correlation")
    axes[2].legend()

    # 4. Log-return distribution
    for label, df in datasets:
        r = df["log_return"].dropna()
        r = r[(r > r.quantile(0.001)) & (r < r.quantile(0.999))]
        axes[3].hist(r, bins=80, density=True, alpha=0.45, label=label)
    axes[3].set_yscale("log")
    axes[3].set_title("Log-price trace generated orders")
    axes[3].set_xlabel("Log return")
    axes[3].set_ylabel("Log frequency")
    axes[3].legend()

    # 5. Absolute log-return autocorrelation
    for label, df in datasets:
        axes[4].plot(lags, autocorr(df["log_return"].abs(), max_lag),
                     marker="o", linewidth=1, label=label)
    axes[4].axhline(0, linewidth=0.8)
    axes[4].set_title("Autocorrelation log returns distribution")
    axes[4].set_xlabel(LAGLBL)
    axes[4].set_ylabel("Autocorrelation")
    axes[4].legend()

    # 6. Mid-price trace — trim every series to the SAME actual time window (the shortest
    # series' length) before plotting, so the x-axis represents identical real minutes for
    # both rather than each series' own full span. The real reference file often covers more
    # wall-clock time than the generated session; plotting each against its own raw index
    # made the shorter one look like it "ran out" partway across the plot, when really the
    # two were just never aligned to the same time window in the first place.
    common_len = min(len(df) for _, df in fine)
    for label, df in fine:
        trace = df["mid"].reset_index(drop=True).iloc[:common_len]
        if len(trace) > 5000:
            trace = trace.iloc[np.linspace(0, len(trace)-1, 5000).astype(int)]
        axes[5].plot(np.arange(len(trace)), trace, linewidth=1, label=label)
    axes[5].set_title("Mid-price traces")
    axes[5].set_xlabel(f"Seconds into session ({TRACE_RESAMPLE} bars, common window)")
    axes[5].set_ylabel("Mid price")
    axes[5].legend()

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    if "--resample" in sys.argv:
        i = sys.argv.index("--resample"); RESAMPLE = sys.argv[i+1]; del sys.argv[i:i+2]
    if len(sys.argv) != 4:
        print("Usage: python paper_style_stylized_facts.py REAL_CSV GENERATED_CSV OUT_PNG")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
