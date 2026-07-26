#!/usr/bin/env python3
"""TRADES predictive score (MAE) — reproducible train-on-synthetic / test-on-real LSTM.

We train a small LSTM to predict the next mid-price from a window of order-flow features,
train it on ONE series (generated or real) and measure its mean absolute error on the REAL
series. Lower error means the training series is as useful as real data for learning to
predict the real market. This is the predictive-score metric used by TRADES (Berti et al.)
and the TimeGAN lineage, where the paper reports figures such as 1.213 -> 3.146.

What this file adds over the older predictive_lstm*.py scripts (which stay as-is):
  - fixed seeds and averaging over several trainings (reports mean and std, not one noisy run),
  - a real-on-real baseline (train on real, test on real) so the generated score is interpretable,
  - correct MAE labelling (the old scripts printed "Test MSE" for an L1 loss),
  - an explicit, order-independent feature set instead of fragile positional column slicing,
  - a clean function API and CLI, and JSON output, so it can be driven in batch.

Single run:
  python -m evaluation.quantitative_eval.predictive_score --real REAL.csv --gen GEN.csv \
      --seeds 3 --out score.json
See predictive_batch.py to score many generated files against one real reference.
"""
import argparse, json, random, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# order-independent feature set (touch book + event + one-hot type). MID_PRICE is both a
# feature (its present value) and, at a future offset, the prediction target.
FEATURES = ['PRICE', 'SIZE', 'BUY_SELL_FLAG', 'ask_price_1', 'ask_size_1',
            'bid_price_1', 'bid_size_1', 'VWAP', 'MID_PRICE',
            'TYPE_LIMIT_ORDER', 'TYPE_ORDER_CANCELLED', 'TYPE_ORDER_EXECUTED']
TARGET = 'MID_PRICE'
SENTINEL = 9e9


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_series(path, start_time='09:45:00'):
    """Read a processed_orders.csv, keep our feature columns, per-series z-score, drop NaNs.
    Per-series normalisation matches the original TRADES eval (each series by its own stats)."""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns and start_time:
        tod = df['Unnamed: 0'].astype(str).str.slice(11, 19)
        df = df[tod >= start_time]
    if 'TYPE' in df.columns:
        df = pd.get_dummies(df, columns=['TYPE'])
    for c in ['TYPE_LIMIT_ORDER', 'TYPE_ORDER_CANCELLED', 'TYPE_ORDER_EXECUTED']:
        if c not in df.columns:
            df[c] = 0
    if 'BUY_SELL_FLAG' in df.columns:
        df['BUY_SELL_FLAG'] = (df['BUY_SELL_FLAG'].astype(str)
                               .map({'True': 1, 'False': 0, '1': 1, '-1': 0, '1.0': 1, '-1.0': 0})
                               .fillna(0))
    for c in ['ask_price_1', 'bid_price_1']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df[df[c].abs() < SENTINEL]
    df = df.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in FEATURES if c in df.columns]
    if TARGET not in cols:
        raise ValueError(f"{TARGET} column not found in {path}")
    df = df[cols].apply(pd.to_numeric, errors='coerce').astype(np.float64)
    # per-column z-score; a degenerate column (constant, or all-NaN like VWAP in these CSVs)
    # becomes 0 rather than NaN, so it does not wipe every row at the dropna below.
    for c in df.columns:
        mu, sd = df[c].mean(), df[c].std()
        if not np.isfinite(sd) or sd < 1e-8:
            df[c] = 0.0
        else:
            df[c] = (df[c] - mu) / sd
    return df.dropna().reset_index(drop=True)


class SeqDataset(Dataset):
    """Windows of `lookback` rows -> the target `horizon` steps after the window end."""
    def __init__(self, x, y, lookback, horizon):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.offset = lookback - 1 + horizon

    def __len__(self):
        return max(0, len(self.x) - self.offset)

    def __getitem__(self, i):
        return self.x[i:i + self.lookback], self.y[i + self.offset]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _loaders(train_x, train_y, test_x, test_y, cfg):
    tr = SeqDataset(train_x, train_y, cfg['lookback'], cfg['horizon'])
    te = SeqDataset(test_x, test_y, cfg['lookback'], cfg['horizon'])
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(f"too few rows for lookback={cfg['lookback']} horizon={cfg['horizon']} "
                         f"(train seqs {len(tr)}, test seqs {len(te)})")
    g = torch.Generator().manual_seed(cfg['_seed'])
    return (DataLoader(tr, batch_size=cfg['batch_size'], shuffle=True, generator=g),
            DataLoader(te, batch_size=cfg['batch_size'], shuffle=False), len(tr), len(te))


def _train_eval(train_x, train_y, test_x, test_y, cfg, seed, device):
    """One seeded training run. Returns test MAE (L1) on the given test set."""
    cfg = {**cfg, '_seed': seed}
    set_seed(seed)
    tl, vl, n_tr, n_te = _loaders(train_x, train_y, test_x, test_y, cfg)
    model = LSTMModel(train_x.shape[1], cfg['hidden'], cfg['layers']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    mse = nn.MSELoss()
    model.train()
    last = float('inf')
    for epoch in range(cfg['epochs']):
        losses = []
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = mse(model(xb), yb.unsqueeze(1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        m = float(np.mean(losses))
        if epoch > 10 and m + 1e-4 > last:       # early stop once training loss plateaus
            break
        last = m
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in vl:
            preds.append(model(xb.to(device)).cpu())
            labels.append(yb)
    preds, labels = torch.cat(preds), torch.cat(labels)
    return nn.functional.l1_loss(preds, labels.unsqueeze(1)).item(), n_tr, n_te


def predictive_score(real_path, gen_path, cfg, device=None, verbose=True):
    """Train-on-synthetic/test-on-real MAE, averaged over cfg['seeds'] trainings.
    Also computes the real-on-real baseline over the same test set when cfg['real_baseline']."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    real = load_series(real_path, cfg['start_time'])
    gen = load_series(gen_path, cfg['start_time'])
    common = [c for c in real.columns if c in gen.columns]
    real, gen = real[common], gen[common]
    ti = common.index(TARGET)
    xr, yr = real.values, real[TARGET].values
    xg, yg = gen.values, gen[TARGET].values
    split = len(xr) // 2                          # shared real test set = latter half of real
    test_x, test_y = xr[split:], yr[split:]

    seeds = list(range(cfg['seeds']))
    gen_maes = [_train_eval(xg, yg, test_x, test_y, cfg, s, device)[0] for s in seeds]
    out = {
        'real_path': real_path, 'gen_path': gen_path,
        'features': common, 'n_real': int(len(xr)), 'n_gen': int(len(xg)),
        'lookback': cfg['lookback'], 'horizon': cfg['horizon'], 'seeds': cfg['seeds'],
        'gen_mae_mean': float(np.mean(gen_maes)), 'gen_mae_std': float(np.std(gen_maes)),
        'gen_mae_runs': [float(v) for v in gen_maes],
    }
    if cfg.get('real_baseline', True):
        base = [_train_eval(xr[:split], yr[:split], test_x, test_y, cfg, s, device)[0] for s in seeds]
        out['real_mae_mean'] = float(np.mean(base))
        out['real_mae_std'] = float(np.std(base))
        out['real_mae_runs'] = [float(v) for v in base]
    if verbose:
        b = f"  real-on-real baseline {out['real_mae_mean']:.4f} +/- {out['real_mae_std']:.4f}" \
            if 'real_mae_mean' in out else ""
        print(f"predictive score (MAE, lower=better)  gen {out['gen_mae_mean']:.4f} "
              f"+/- {out['gen_mae_std']:.4f}{b}   [n_gen={out['n_gen']} n_real={out['n_real']}]")
    return out


def cfg_from_args(a):
    return dict(lookback=a.lookback, horizon=(a.horizon if a.horizon is not None else a.lookback),
                hidden=a.hidden, layers=a.layers, epochs=a.epochs, batch_size=a.batch_size,
                lr=a.lr, seeds=a.seeds, start_time=a.start_time, real_baseline=not a.no_real_baseline)


def add_common_args(p):
    p.add_argument('--lookback', type=int, default=100, help='window length (default 100, the paper-scale config)')
    p.add_argument('--horizon', type=int, default=None, help='steps ahead of the window end to predict (default = lookback)')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seeds', type=int, default=3, help='number of seeded trainings to average')
    p.add_argument('--start-time', default='09:45:00', help="drop rows before this HH:MM:SS (warm-up); '' to keep all")
    p.add_argument('--no-real-baseline', action='store_true', help='skip the real-on-real baseline')
    p.add_argument('--device', default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', required=True)
    ap.add_argument('--gen', required=True)
    ap.add_argument('--out', default=None, help='write the result dict to this JSON file')
    add_common_args(ap)
    a = ap.parse_args()
    res = predictive_score(a.real, a.gen, cfg_from_args(a), device=a.device)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=2)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
