import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


MAX_LAG = 30
HIST_BINS = 60
NORMALIZE_MIDPRICE_TRACE = True
MIDPRICE_BASE = 100.0

COLOR_MAP = {
    "Real": "black",
    "AAME": "red",
    "INTC": "blue",
    "AAME_REAL": "darkred",
    "AAME_TRADES": "red",
    "INTC_REAL": "black",
    "INTC_TRADES": "blue",
}


def load_processed_orders(path, label):
    path = Path(path)
    df = pd.read_csv(path).copy()

    if "Unnamed: 0" in df.columns:
        df["datetime"] = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["datetime"] = pd.date_range("2000-01-01", periods=len(df), freq="s")

    if "ask_price_1" in df.columns and "bid_price_1" in df.columns:
        df["ask_price_1"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
        df["bid_price_1"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
        df = df.query("ask_price_1 < 9999999 and bid_price_1 < 9999999").copy()
        df = df.query("ask_price_1 > -9999999 and bid_price_1 > -9999999").copy()

    if "MID_PRICE" in df.columns:
        df["mid"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
    elif "ask_price_1" in df.columns and "bid_price_1" in df.columns:
        df["mid"] = (df["ask_price_1"] + df["bid_price_1"]) / 2
    else:
        raise ValueError(f"No mid-price columns found in {path}")

    if "SIZE" in df.columns:
        df["volume"] = pd.to_numeric(df["SIZE"], errors="coerce").abs()
    elif "ORDER_SIZE" in df.columns:
        df["volume"] = pd.to_numeric(df["ORDER_SIZE"], errors="coerce").abs()
    elif "QUANTITY" in df.columns:
        df["volume"] = pd.to_numeric(df["QUANTITY"], errors="coerce").abs()
    else:
        df["volume"] = 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mid", "volume"])
    df = df[df["mid"] > 0].copy()

    df["log_mid"] = np.log(df["mid"])
    df["log_return_event"] = df["log_mid"].diff()

    df = df.dropna(subset=["datetime"]).copy()
    df["minute"] = df["datetime"].dt.floor("min")

    minute = pd.DataFrame({
        "mid": df.groupby("minute")["mid"].last(),
        "volume": df.groupby("minute")["volume"].sum(),
    }).dropna()

    minute["log_mid"] = np.log(minute["mid"])
    minute["log_return"] = minute["log_mid"].diff()
    minute["abs_log_return"] = minute["log_return"].abs()
    minute = minute.dropna().copy()

    print(f"{label}: raw rows={len(df)}, active minutes={len(minute)}")

    return {
        "label": label,
        "raw": df,
        "minute": minute,
        "path": str(path),
    }


def adaptive_window(n, preferred=15):
    if n >= preferred * 3:
        return preferred
    if n >= 10:
        return 5
    if n >= 5:
        return 3
    return None


def autocorr_series(x, max_lag=30):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    max_lag = min(max_lag, max(1, len(x) - 2))
    vals = []
    for lag in range(1, max_lag + 1):
        vals.append(x.autocorr(lag=lag))
    return np.arange(1, max_lag + 1), np.array(vals)


def rolling_corr_distribution(x, y):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    n = min(len(x), len(y))
    x = x.iloc[:n]
    y = y.iloc[:n]

    window = adaptive_window(n)
    if window is None:
        return np.array([])

    vals = x.rolling(window).corr(y)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    return vals.values


def safe_clip(x, q_low=0.001, q_high=0.999):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 5:
        return x.values
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)
    return x[(x >= lo) & (x <= hi)].values


def density_plot(ax, values, label, color):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().values

    if len(values) == 0:
        print(f"WARNING: no valid values for {label}")
        return

    if len(values) < 10 or np.nanstd(values) < 1e-12 or not SCIPY_OK:
        ax.hist(values, bins=min(20, max(5, len(values))), density=True,
                alpha=0.35, color=color, label=label)
        return

    xs = np.linspace(np.nanmin(values), np.nanmax(values), 400)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)


def get_color(label, idx):
    if label in COLOR_MAP:
        return COLOR_MAP[label]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)]


def normalize_trace(y):
    y = pd.Series(y).dropna().values
    if len(y) == 0:
        return y
    if not NORMALIZE_MIDPRICE_TRACE:
        return y
    return MIDPRICE_BASE * y / y[0]


def make_figure(datasets, out_path, title=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # 1. Log returns autocorrelation
    ax = axes[0]
    for i, ds in enumerate(datasets):
        lags, vals = autocorr_series(ds["minute"]["log_return"], MAX_LAG)
        ax.plot(lags, vals, marker="o", linewidth=1.5,
                color=get_color(ds["label"], i), label=ds["label"])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Log Returns Autocorrelation")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()

    # 2. Volume-volatility correlation distribution
    ax = axes[1]
    for i, ds in enumerate(datasets):
        minute = ds["minute"].copy()
        w = adaptive_window(len(minute))
        if w is not None:
            minute["volatility"] = minute["log_return"].rolling(w).std()
            vals = rolling_corr_distribution(minute["volume"], minute["volatility"])
            density_plot(ax, vals, ds["label"], get_color(ds["label"], i))
    ax.set_title("Correlation between Volume and Volatility")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.legend()

    # 3. Returns-volatility correlation distribution
    ax = axes[2]
    for i, ds in enumerate(datasets):
        minute = ds["minute"].copy()
        w = adaptive_window(len(minute))
        if w is not None:
            minute["volatility"] = minute["log_return"].rolling(w).std()
            vals = rolling_corr_distribution(minute["log_return"], minute["volatility"])
            density_plot(ax, vals, ds["label"], get_color(ds["label"], i))
    ax.set_title("Correlation between Returns and Volatility")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.legend()

    # 4. Minutely log returns comparison
    ax = axes[3]
    for i, ds in enumerate(datasets):
        vals = safe_clip(ds["minute"]["log_return"])
        density_plot(ax, vals, ds["label"], get_color(ds["label"], i))
    ax.set_title("Minutely Log Returns Comparison")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()

    # 5. Absolute log-return autocorrelation distribution
    ax = axes[4]
    for i, ds in enumerate(datasets):
        _, ac = autocorr_series(ds["minute"]["log_return"].abs(), MAX_LAG)
        density_plot(ax, ac, ds["label"], get_color(ds["label"], i))
    ax.set_title("Autocorrelation Log Returns Distribution")
    ax.set_xlabel("Autocorrelation")
    ax.set_ylabel("Density")
    ax.legend()

    # 6. Mid-price traces
    ax = axes[5]
    for i, ds in enumerate(datasets):
        trace = normalize_trace(ds["minute"]["mid"])
        ax.plot(np.arange(len(trace)), trace, linewidth=1.8,
                color=get_color(ds["label"], i), label=ds["label"])
    ax.set_title("Mid-Price Traces")
    ax.set_xlabel("Minute Index")
    ax.set_ylabel("Normalized Mid Price")
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", nargs=2, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    datasets = []
    for label, path in args.series:
        print("Loading", label, path)
        datasets.append(load_processed_orders(path, label))

    make_figure(datasets, args.out, args.title)


if __name__ == "__main__":
    main()
