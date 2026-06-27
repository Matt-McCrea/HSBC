import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

LOOKBACK = 10
TARGET_OFFSET = 19
BATCH_SIZE = 16
EPOCHS = 100

class TinyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.x) - TARGET_OFFSET)

    def __getitem__(self, i):
        return self.x[i:i+LOOKBACK], self.y[i+TARGET_OFFSET]

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def clean(path):
    df = pd.read_csv(path)

    # Filter from 09:45 onwards if timestamp column exists
    if "Unnamed: 0" in df.columns:
        df["tmp_time"] = df["Unnamed: 0"].astype(str).str.slice(11, 19)
        df = df.query("tmp_time >= '09:45:00'")
        df = df.drop(columns=["tmp_time"])

    # Create missing type dummies safely
    if "TYPE" in df.columns:
        df = pd.get_dummies(df, columns=["TYPE"])

    for c in ["TYPE_LIMIT_ORDER", "TYPE_ORDER_CANCELLED", "TYPE_ORDER_EXECUTED"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].astype(int)

    if "BUY_SELL_FLAG" in df.columns:
        df["BUY_SELL_FLAG"] = df["BUY_SELL_FLAG"].astype(str).map(
            {"True": 1, "False": 0, "1": 1, "-1": 0}
        ).fillna(0).astype(int)

    for c in ["ORDER_ID", "SPREAD", "ORDER_VOLUME_IMBALANCE", "Unnamed: 0"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.replace([np.inf, -np.inf], np.nan)

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df = df[num_cols].copy()

    if "MID_PRICE" not in df.columns:
        raise ValueError("MID_PRICE column not found")

    for c in df.columns:
        std = df[c].std()
        mean = df[c].mean()
        if std is None or np.isnan(std) or std < 1e-8:
            df[c] = 0.0
        else:
            df[c] = (df[c] - mean) / std

    df = df.dropna()
    return df

def score(real_path, gen_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = clean(real_path)
    gen = clean(gen_path)

    print("real rows after cleaning:", len(real))
    print("generated rows after cleaning:", len(gen))

    common = [c for c in real.columns if c in gen.columns]
    real = real[common]
    gen = gen[common]

    y_real = real["MID_PRICE"].values
    y_gen = gen["MID_PRICE"].values

    x_real = real.values
    x_gen = gen.values

    # Match original idea: train predictor on generated data, test on real data
    train_x = x_gen
    train_y = y_gen

    # Use latter half of real data as test because AAME is tiny
    split = len(x_real) // 2
    test_x = x_real[split:]
    test_y = y_real[split:]

    train_data = TinyDataset(train_x, train_y)
    test_data = TinyDataset(test_x, test_y)

    print("generated train sequences:", len(train_data))
    print("real test sequences:", len(test_data))

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Still too few rows. Need even smaller LOOKBACK/TARGET_OFFSET.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_size=train_x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    mse = nn.MSELoss()

    model.train()
    last = 1e9
    for epoch in range(EPOCHS):
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb.unsqueeze(1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        if epoch > 10 and mean_loss > last:
            break
        last = mean_loss

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            preds.append(pred)
            labels.append(yb)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    mae = nn.functional.l1_loss(preds, labels.unsqueeze(1)).item()
    print("AAME adapted predictive score MAE:", mae)
    return mae

if __name__ == "__main__":
    score(sys.argv[1], sys.argv[2])
