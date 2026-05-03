
# --- Imports & notebook defaults (self-contained) ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Window/horizon defaults for NB03 (we’ll reuse these later)
S = 7  # history window length
H = 7  # forecast horizon

DATA_PATH = "1_Daily_minimum_temps.csv"
print(f"[NB03] Defaults: S={S}, H={H}")

# Load the CSV with strict date parsing and use the date as the index (so time-aware ops are easy and safe).
# Make a time-aware split:   - Train: 1981-01-01 → 1988-12-31  - Test: 1989-01-01 → 1990-12-31  
# Keep the daily calendar intact. Dropping rows with missing values would “compress” time. Instead, we: 
# - Reindex to a complete daily grid, then  
# - Impute missing temps per split (train and test separately) using time-based linear interpolation. 
# Scale with a train-only fit of `MinMaxScaler` to the range [0,1]: 

# --- Load CSV with strict date parsing & numeric Temp ---
# The file uses MM/DD/YY, e.g., 01/13/81 = Jan 13, 1981
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")  # strict format
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")     # coerce bad entries to NaN
df = df.set_index("Date").sort_index()

print("Raw shape (rows, cols):", df.shape)
display(df.head(5))
display(df.tail(5))

# Sanity: is the index monotonic by time?
print("Date index monotonic increasing? ->", df.index.is_monotonic_increasing)

# --- Keep the daily calendar intact: reindex to complete daily grid ---
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_idx)
df.index.name = "Date"

missing_total = int(df["Temp"].isna().sum())
print(f"Missing days after reindex (before imputation): {missing_total}")

# Quick peek around any missing spots (optional)
# df[df["Temp"].isna()].head()

# --- Time-aware split: Train (1981–1988) vs Test (1989–1990) ---
split_date = pd.Timestamp("1989-01-01")  # test starts here

df_train = df.loc[: split_date - pd.Timedelta(days=1)].copy()
df_test  = df.loc[split_date :].copy()

print(f"Train range: {df_train.index.min().date()} → {df_train.index.max().date()} | rows={len(df_train)}")
print(f" Test range: {df_test.index.min().date()}  → {df_test.index.max().date()}  | rows={len(df_test)}")
print("Missing in train BEFORE impute:", int(df_train['Temp'].isna().sum()))
print("Missing in test  BEFORE impute:", int(df_test['Temp'].isna().sum()))

# --- Impute per split (no cross-split peeking) using time-based interpolation ---
# 'method="time"' uses the DatetimeIndex; 'limit_direction="both"' fills edges within each split safely.
df_train["Temp"] = df_train["Temp"].interpolate(method="time", limit_direction="both")
df_test["Temp"]  = df_test["Temp"].interpolate(method="time",  limit_direction="both")

print("Missing in train AFTER  impute:", int(df_train['Temp'].isna().sum()))
print("Missing in test  AFTER  impute:", int(df_test['Temp'].isna().sum()))

# Optional: preview ends to verify continuity
display(df_train.head(3)); display(df_train.tail(3))
display(df_test.head(3));  display(df_test.tail(3))

# --- Fit MinMaxScaler on TRAIN ONLY (no leakage), then transform both splits ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[["Temp"]])                 # train-only fit

train_scaled = scaler.transform(df_train[["Temp"]])  # ndarray (N_train, 1)
test_scaled  = scaler.transform(df_test[["Temp"]])   # ndarray (N_test, 1)

print("Scaled range (TRAIN) min/max:", float(train_scaled.min()), float(train_scaled.max()))
print("Scaled range (TEST)  min/max:", float(test_scaled.min()),  float(test_scaled.max()))

# print shape of scaled arrays
print("Scaled TRAIN shape (rows, cols):", train_scaled.shape)
print("Scaled  TEST shape (rows, cols):", test_scaled.shape)

# --- Fit MinMaxScaler on TRAIN ONLY (no leakage), then transform both splits ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[["Temp"]])                 # train-only fit

train_scaled = scaler.transform(df_train[["Temp"]])  # ndarray (N_train, 1)
test_scaled  = scaler.transform(df_test[["Temp"]])   # ndarray (N_test, 1)

print("Scaled range (TRAIN) min/max:", float(train_scaled.min()), float(train_scaled.max()))
print("Scaled range (TEST)  min/max:", float(test_scaled.min()),  float(test_scaled.max()))

# print shape of scaled arrays
print("Scaled TRAIN shape (rows, cols):", train_scaled.shape)
print("Scaled  TEST shape (rows, cols):", test_scaled.shape)
# --------------------------------------------------------------------------------------------------------
 # ---- One-time sanity audit on °C (before scaling is fine; here we use df_train/df_test after impute) ----
PLAUDIBLE_MIN, PLAUDIBLE_MAX = -15.0, 45.0   # tweak if your locale differs
BIG_JUMP = 12.0                              # day-to-day jump threshold in °C

def _audit(split_name, s):
    s = s.dropna()
    print(f"\n[{split_name}] rows={len(s)}  min={s.min():.2f}°C  max={s.max():.2f}°C  mean={s.mean():.2f}°C")
    # Range check
    if (s.min() < PLAUDIBLE_MIN) or (s.max() > PLAUDIBLE_MAX):
        print(f"  ⚠ Outside plausible range [{PLAUDIBLE_MIN}, {PLAUDIBLE_MAX}] °C")
    # Day-to-day jumps
    jumps = s.diff().abs()
    big = int((jumps > BIG_JUMP).sum())
    print(f"  Day-to-day jumps > {BIG_JUMP:.0f}°C: {big}  ({100*big/max(1,len(jumps)):.1f}%)")
    # Global z-score outliers
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-8)
    out = int((z.abs() > 4).sum())
    print(f"  |z| > 4 outliers: {out}")
    # Seasonality (month medians; warmest/coldest 3)
    by_mon = s.groupby(s.index.month).median()
    warm = by_mon.sort_values(ascending=False).head(3)
    cold = by_mon.sort_values().head(3)
    wtxt = ", ".join([f"{int(m)}({v:.1f}°C)" for m, v in warm.items()])
    ctxt = ", ".join([f"{int(m)}({v:.1f}°C)" for m, v in cold.items()])
    print(f"  Warmest median months: {wtxt}")
    print(f"  Coldest median months: {ctxt}")

_audit("TRAIN", df_train["Temp"])
_audit("TEST",  df_test["Temp"])

# Optional: quick histogram (comment in if you want a visual)
# import matplotlib.pyplot as plt
# df_train["Temp"].hist(bins=30, alpha=0.6, label="Train"); df_test["Temp"].hist(bins=30, alpha=0.6, label="Test")
# plt.xlabel("Daily minimum temperature (°C)"); plt.ylabel("Count"); plt.legend(); plt.show()

# --------------------------------------------------------------------------------------------------------
# --- Helper: create multi-step sequences without crossing a segment boundary ---
import numpy as np
from typing import Tuple

def create_multi_step_sequences(values: np.ndarray, S: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D (or Nx1) array 'values' into (X, y) pairs for multi-step forecasting.

    Args:
      values: array-like of shape (N,) or (N,1) containing a single time series (scaled or raw).
      S: input window length (history steps).
      H: forecast horizon length (future steps).

    Returns:
      X: ndarray of shape (N_samples, S, 1)   [batch-first windows]
      y: ndarray of shape (N_samples, H)      [multi-step targets]

    Notes:
      - We build windows strictly within the given 'values' segment (no boundary crossing).
      - We keep the final singleton feature dimension so X is ready for batch-first RNN/LSTM/GRU.
    """
    v = values.reshape(-1)  # flatten to 1D array
    N = len(v)
    N_samples = N - (S + H) + 1
    if N_samples <= 0:
        raise ValueError(f"Not enough data points ({N}) for S={S}, H={H}. Need at least S+H samples.")
    X = np.zeros((N_samples, S, 1), dtype=np.float32)
    Y = np.zeros((N_samples, H),   dtype=np.float32)
    for i in range(N_samples):
        X[i, :, 0] = v[i : i + S] # 0 means 
        Y[i, :]    = v[i + S : i + S + H]
    return X, Y

# --- Build train/test multi-step datasets from Section 2 outputs ---
# We assume we already have:
#   df_train, df_test : DataFrames with 'Temp' in °C (after per-split interpolation)
#   train_scaled, test_scaled : numpy arrays shaped (N,1) scaled to [0,1] by train-only MinMax
#   S, H              : window and horizon sizes (defaults 7, 7)

# 1) Create (X, y) in SCALED space (these will feed the models)
X_train_sc, y_train_sc = create_multi_step_sequences(train_scaled, S=S, H=H)
X_test_sc,  y_test_sc  = create_multi_step_sequences(test_scaled,  S=S, H=H)

print("Scaled arrays for modeling:")
print("  X_train_sc:", X_train_sc.shape, "| y_train_sc:", y_train_sc.shape)
print("  X_test_sc :", X_test_sc.shape,  "| y_test_sc :",  y_test_sc.shape)

# print sample values (optional)
print("X_train_sc sample:\n", X_train_sc[:2])
print("y_train_sc sample:\n", y_train_sc[:2])
# --------------------------------------------------------------------------------------------------------
# Human-readable check (in °C)
def preview_windows_human(df_segment: pd.DataFrame, S: int, H: int, n_rows: int = 3) -> pd.DataFrame:
    vals = df_segment["Temp"].to_numpy().reshape(-1)
    dates = df_segment.index.to_numpy()
    N = len(vals)
    rows = []
    max_i = min(n_rows, N - (S + H) + 1)
    for i in range(max_i):
        X_vals = vals[i : i + S]
        X_dates = dates[i : i + S]
        y_vals = vals[i + S : i + S + H]
        y_dates = dates[i + S : i + S + H]
        rows.append({
            "input_end_date": pd.to_datetime(X_dates[-1]).date(),
            "X (past S days)": [round(float(x), 1) for x in X_vals],
            "y (next H days)": [round(float(y), 1) for y in y_vals],
            "X_dates": [str(pd.to_datetime(d).date()) for d in X_dates],
            "y_dates": [str(pd.to_datetime(d).date()) for d in y_dates],
        })
    return pd.DataFrame(rows)

preview_df = preview_windows_human(df_train, S=S, H=H, n_rows=3)
display(preview_df)

# --------------------------------------------------------------------------------------------------------
# CT_Task 1 — Build (S=14, H=7) windows in °C from df_test['Temp']
CT_S14, CT_H7 = 14, 7
CT_vals_test = df_test["Temp"].to_numpy().reshape(-1, 1)

# Use the notebook helper:
CT_X_C_14, CT_y_C_14 = create_multi_step_sequences(CT_vals_test, S=CT_S14, H=CT_H7)

print("CT_X_C_14 shape:", CT_X_C_14.shape)
print("CT_y_C_14 shape:", CT_y_C_14.shape)
#print("CT_X_C_14[0] (first window, °C):", CT_X_C_14[:, 0] )  # expect length 14 in the last dim sliced as [:,0]

# CT_Task 2 — Print the first y vector (H=7) for the S=14 windows
print("CT_y_C_14[0] (first target, °C):", CT_y_C_14[0])
#print("First y length:", CT_y_C_14[:, 0])

# --------------------------------------------------------------------------------------------------------
# --- Build test windows/targets in °C for easy interpretation (from df_test prepared in Section 2) ---
# We already have df_test['Temp'] in degrees C (after per-split interpolation).
# We'll reuse our helper to create (X, y) pairs from the ORIGINAL series (not scaled) for baseline evaluation & plotting.

X_test_C, y_test_C = create_multi_step_sequences(
    df_test["Temp"].to_numpy().reshape(-1, 1), S=S, H=H
)

print("In °C (for baselines/plots):")
print("  X_test_C:", X_test_C.shape, "| y_test_C:", y_test_C.shape)
# print sample X_test_C and y_test_C (optional)
print("X_test_C sample:\n", X_test_C[:2])
print("y_test_C sample:\n", y_test_C[:2])
# --------------------------------------------------------------------------------------------------------
# --- Define baselines on the °C windows ---
def baseline_persistence(X_windows: np.ndarray, H: int) -> np.ndarray:
    """
    X_windows: (N, S, 1) in °C
    Returns:   (N, H) predictions in °C
    """
    last_vals = X_windows[:, -1, 0]            # T_t
    return np.tile(last_vals.reshape(-1, 1), (1, H))

def baseline_mean_S(X_windows: np.ndarray, H: int) -> np.ndarray:
    """
    X_windows: (N, S, 1) in °C
    Returns:   (N, H) predictions in °C using mean of last S days
    """
    means = X_windows[:, :, 0].mean(axis=1)    # mean over window length S
    return np.tile(means.reshape(-1, 1), (1, H))

# Generate baseline predictions
yhat_pers = baseline_persistence(X_test_C, H=H)
yhat_mean = baseline_mean_S(X_test_C, H=H)

print("Baseline preds shapes:", yhat_pers.shape, yhat_mean.shape)

# --- Metrics by horizon (MAE and RMSE in °C) ---
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Compute horizon-wise metrics
rows = []
for k in range(H):  # 0..H-1 corresponds to horizons 1..H
    yk_true = y_test_C[:, k]
    yk_pers = yhat_pers[:, k]
    yk_mean = yhat_mean[:, k]
    rows.append({
        "Horizon": k + 1,
        "Persistence_MAE": mae(yk_true, yk_pers),
        "Persistence_RMSE": rmse(yk_true, yk_pers),
        "MeanS_MAE": mae(yk_true, yk_mean),
        "MeanS_RMSE": rmse(yk_true, yk_mean),
    })

import pandas as pd
df_base = pd.DataFrame(rows).set_index("Horizon")
df_base_rounded = df_base.round(3)
display(df_base_rounded)

# Also compute the mean across horizons as a compact summary
summary = df_base.mean(axis=0).to_frame(name="Mean over horizons").round(3)
display(summary)

# --------------------------------------------------------------------------------------------------------
# CT_Task 3 — Persistence baseline for S=14 windows (H=7)
CT_yhat_persist_14 = baseline_persistence(CT_X_C_14, H=7)
print("CT_yhat_persist_14 shape:", CT_yhat_persist_14.shape)
print("CT_yhat_persist_14[0]:", CT_yhat_persist_14[:1])

# CT_Task 4 — Mean-S (S=14) baseline for H=7
CT_yhat_meanS_14 = baseline_mean_S(CT_X_C_14, H=7)
print("CT_yhat_meanS_14 shape:", CT_yhat_meanS_14.shape)
print("CT_yhat_meanS_14[0]:", CT_yhat_meanS_14[0])

# CT_Task 5 — Overall MAE for persistence and mean-S (S=14)
# Use provided mae(y_true, y_pred)

CT_mae_persist_14 = mae(CT_y_C_14, CT_yhat_persist_14)
CT_mae_meanS_14   = mae(CT_y_C_14, CT_yhat_meanS_14)

print("CT_mae_persist_14:", CT_mae_persist_14)
print("CT_mae_meanS_14:", CT_mae_meanS_14)

# --------------------------------------------------------------------------------------------------------
# --- Ensure models & weights are available (reuse from Section 1 if already loaded) ---
import os, torch

# If models aren't defined (e.g., running this section standalone), rebuild them:
try:
    model_rnn, model_lstm, model_gru
except NameError:
    # Recreate architectures with the SAME hyperparams as NB02
    class RNNModel(torch.nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
            super().__init__()
            self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers,
                                    nonlinearity="tanh", batch_first=batch_first)
            self.head = torch.nn.Linear(hidden_size, 1)
            self.hidden_size, self.num_layers, self.batch_first = hidden_size, num_layers, batch_first
        def forward(self, x):
            B = x.size(0)
            h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            out, _ = self.rnn(x, h0)      # (B,S,H)
            return self.head(out[:, -1, :])

    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
            self.head = torch.nn.Linear(hidden_size, 1)
            self.hidden_size, self.num_layers, self.batch_first = hidden_size, num_layers, batch_first
        def forward(self, x):
            B = x.size(0)
            h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))   # (B,S,H)
            return self.head(out[:, -1, :])

    class GRUModel(torch.nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
            super().__init__()
            self.gru  = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
            self.head = torch.nn.Linear(hidden_size, 1)
            self.hidden_size, self.num_layers, self.batch_first = hidden_size, num_layers, batch_first
        def forward(self, x):
            B = x.size(0)
            h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            out, _ = self.gru(x, h0)          # (B,S,H)
            return self.head(out[:, -1, :])

    DEVICE = torch.device("cpu")
    model_rnn  = RNNModel(batch_first=True).to(DEVICE)
    model_lstm = LSTMModel(batch_first=True).to(DEVICE)
    model_gru  = GRUModel(batch_first=True).to(DEVICE)

# Try to load weights saved in ./artifacts from NB02 (or Project 7 if compatible)
ART_DIR = "artifacts"
def try_load(model, fname):
    path = os.path.join(ART_DIR, fname)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only = False))
        print(f"✅ loaded: {fname}")
        return True
    else:
        print(f"⚠️ missing: {fname} (using random init)")
        return False

ok_rnn  = try_load(model_rnn,  "rnn_state.pkl")
ok_lstm = try_load(model_lstm, "lstm_state.pkl")
ok_gru  = try_load(model_gru,  "gru_state.pkl")

for m in (model_rnn, model_lstm, model_gru):
    m.eval()
# --------------------------------------------------------------------------------------------------------
USE_CLAMP = False  # set True only if you see drift in rollout of predictions 

@torch.no_grad()
def rollout_recursive(model, X_windows_scaled, H, clamp=(0.0, 1.0)):
    """
    Args:
      model: trained single-step model that maps (B,S,1) -> (B,1) in scaled space.
      X_windows_scaled: numpy array (N, S, 1) in scaled space for the starting windows.
      H: forecast horizon (number of steps to roll out).
      clamp: tuple (min, max) to clamp predictions if USE_CLAMP is True.
    Returns:
      y_pred_scaled: numpy array (N, H) of scaled forecasts.
    """
    model.eval()
    curr = torch.tensor(X_windows_scaled, dtype=torch.float32)
    N = curr.size(0)
    preds = torch.zeros((N, H), dtype=torch.float32)
    for k in range(H):
        y_hat = model(curr)
        if USE_CLAMP:
            y_hat = y_hat.clamp(min=clamp[0], max=clamp[1])
        preds[:, k] = y_hat.squeeze(1)
        curr = torch.cat([curr[:, 1:, :], y_hat.unsqueeze(-1)], dim=1)
    return preds.numpy()
# --------------------------------------------------------------------------------------------------------
# --- Run recursive rollout for each model on the TEST set (scaled), then inverse to °C ---
# We assume from Section 3:
#   X_test_sc (N,S,1), y_test_sc (N,H)  --> scaled space
#   y_test_C  (N,H)                     --> degrees C (for reporting)
# And from Section 2:
#   scaler (MinMaxScaler fit on TRAIN ONLY)

# 1) Roll out (scaled)
yhat_rnn_sc  = rollout_recursive(model_rnn,  X_test_sc, H) if ok_rnn  else None
yhat_lstm_sc = rollout_recursive(model_lstm, X_test_sc, H) if ok_lstm else None
yhat_gru_sc  = rollout_recursive(model_gru,  X_test_sc, H) if ok_gru  else None

# 2) Inverse-transform to °C for evaluation/plots
def inverse_scale_2d(y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform (N,H) using the same scaler trained on train; returns (N,H) in °C."""
    flat = y_scaled.reshape(-1, 1)
    back = scaler.inverse_transform(flat)
    return back.reshape(y_scaled.shape)

yhat_rnn_C  = inverse_scale_2d(yhat_rnn_sc)   if yhat_rnn_sc  is not None else None
yhat_lstm_C = inverse_scale_2d(yhat_lstm_sc)  if yhat_lstm_sc is not None else None
yhat_gru_C  = inverse_scale_2d(yhat_gru_sc)   if yhat_gru_sc  is not None else None

# --------------------------------------------------------------------------------------------------------
# --- Metrics: MAE/RMSE by horizon, and overlay with baselines ---

def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# Baselines from Section 4 (recompute here quickly to be self-contained)
def baseline_persistence(X_windows_C: np.ndarray, H: int) -> np.ndarray:
    last_vals = X_windows_C[:, -1, 0]
    return np.tile(last_vals.reshape(-1, 1), (1, H))

def baseline_mean_S(X_windows_C: np.ndarray, H: int) -> np.ndarray:
    means = X_windows_C[:, :, 0].mean(axis=1)
    return np.tile(means.reshape(-1, 1), (1, H))

yhat_pers = baseline_persistence(X_test_C, H)
yhat_mean = baseline_mean_S(X_test_C, H)

rows = []
for k in range(H):  # horizons 1..H
    true_k = y_test_C[:, k]
    row = {"Horizon": k+1,
           "Persistence_MAE": mae(true_k, yhat_pers[:, k]),
           "MeanS_MAE":      mae(true_k, yhat_mean[:, k])}
    if yhat_rnn_C  is not None:  row["RNN_MAE"]  = mae(true_k, yhat_rnn_C[:,  k])
    if yhat_lstm_C is not None:  row["LSTM_MAE"] = mae(true_k, yhat_lstm_C[:, k])
    if yhat_gru_C  is not None:  row["GRU_MAE"]  = mae(true_k, yhat_gru_C[:,  k])

    row["Persistence_RMSE"] = rmse(true_k, yhat_pers[:, k])
    row["MeanS_RMSE"]       = rmse(true_k, yhat_mean[:, k])
    if yhat_rnn_C  is not None:  row["RNN_RMSE"]  = rmse(true_k, yhat_rnn_C[:,  k])
    if yhat_lstm_C is not None:  row["LSTM_RMSE"] = rmse(true_k, yhat_lstm_C[:, k])
    if yhat_gru_C  is not None:  row["GRU_RMSE"]  = rmse(true_k, yhat_gru_C[:,  k])

    rows.append(row)

df_rec = pd.DataFrame(rows).set_index("Horizon").round(3)
display(df_rec)

# mean across horizons (compact ranking)
mean_summary = df_rec.mean(axis=0).to_frame(name="Mean over horizons").round(3)
display(mean_summary)

# --------------------------------------------------------------------------------------------------------
# --- DataLoaders for Direct training (scaled space) ---
# We assume from Section 3:
#   X_train_sc, y_train_sc  -> (N_train, S, 1), (N_train, H)
#   X_test_sc,  y_test_sc   -> (N_test,  S, 1), (N_test,  H)
# And from Section 2 for reporting:
#   scaler (MinMax fit on train only), X_test_C, y_test_C (in °C)

import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH = 64

train_ds_direct = TensorDataset(
    torch.tensor(X_train_sc, dtype=torch.float32),
    torch.tensor(y_train_sc, dtype=torch.float32),
)
test_ds_direct = TensorDataset(
    torch.tensor(X_test_sc, dtype=torch.float32),
    torch.tensor(y_test_sc, dtype=torch.float32),
)

train_loader_direct = DataLoader(train_ds_direct, batch_size=BATCH, shuffle=True,  drop_last=False)
test_loader_direct  = DataLoader(test_ds_direct,  batch_size=BATCH, shuffle=False, drop_last=False)

len(train_ds_direct), len(test_ds_direct)

# --------------------------------------------------------------------------------------------------------
# --- GRU-Direct: (B,S,1) -> (B,H) ---
import torch.nn as nn

class GRUModelDirect(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, H=7, batch_first=True):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.head = nn.Linear(hidden_size, H)   # direct multi-output head

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeros(self.gru.num_layers, B, self.gru.hidden_size)
        out, _ = self.gru(x, h0)                # (B,S,Hh)
        h_last = out[:, -1, :]                  # (B,Hh)
        y_hat  = self.head(h_last)              # (B,H) in scaled space
        return y_hat

# Instantiate
H = y_train_sc.shape[1]  # ensure we use the same horizon as our data
GRU_model_direct = GRUModelDirect(input_size=1, hidden_size=64, num_layers=1, H=H, batch_first=True)
GRU_model_direct

# --------------------------------------------------------------------------------------------------------
# --- Inference to °C and horizon-wise metrics (compare to baselines and (optionally) recursive) ---
import pandas as pd

@torch.no_grad()
def predict_direct(model, X_sc):
    model.eval()
    yhat_sc = []
    for i in range(0, len(X_sc), BATCH):
        xb = torch.tensor(X_sc[i:i+BATCH], dtype=torch.float32).to(DEVICE)
        yhat_sc.append(model(xb).cpu().numpy())
    return np.vstack(yhat_sc)  # (N,H) scaled

def inverse_scale_2d(y_scaled):
    flat = y_scaled.reshape(-1,1)
    back = scaler.inverse_transform(flat)
    return back.reshape(y_scaled.shape)

def mae(a,b):  return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

# Predictions
yhat_direct_sc = predict_direct(GRU_model_direct, X_test_sc)      # (N,H) scaled
yhat_direct_C  = inverse_scale_2d(yhat_direct_sc)             # (N,H) °C

# Baselines (from Section 4); rebuild quickly to be self-contained
def baseline_persistence(X_windows_C, H):
    last_vals = X_windows_C[:, -1, 0]
    return np.tile(last_vals.reshape(-1,1), (1,H))
def baseline_mean_S(X_windows_C, H):
    means = X_windows_C[:, :, 0].mean(axis=1)
    return np.tile(means.reshape(-1,1), (1,H))

yhat_pers = baseline_persistence(X_test_C, H)
yhat_mean = baseline_mean_S(X_test_C, H)

# If we already computed recursive predictions earlier, we can compare:
have_rec = False
try:
    _ = yhat_gru_C  # from Section 5
    have_rec = True
except:
    pass

# Horizon-wise table
rows = []
for k in range(H):
    y_true = y_test_C[:, k]
    row = {
        "Horizon": k+1,
        "Direct_GRU_MAE":  mae(y_true, yhat_direct_C[:, k]),
        "Direct_GRU_RMSE": rmse(y_true, yhat_direct_C[:, k]),
        "Persistence_MAE": mae(y_true, yhat_pers[:,    k]),
        "MeanS_MAE":       mae(y_true, yhat_mean[:,    k]),
    }
    if have_rec:
        row["GRU_rec_MAE"] = mae(y_true, yhat_gru_C[:, k])
    rows.append(row)

df_direct = pd.DataFrame(rows).set_index("Horizon").round(3)
display(df_direct)

print("\nMean MAE over horizons:")
display(df_direct[["Direct_GRU_MAE","MeanS_MAE","Persistence_MAE"] + (["GRU_rec_MAE"] if have_rec else [])].mean().to_frame("Mean").round(3))

# --------------------------------------------------------------------------------------------------------
# --- Plot: MAE vs horizon (Direct vs Baselines (+ Recursive if present)) ---
plt.figure(figsize=(7.5, 4.8))
plt.plot(df_direct.index, df_direct["Persistence_MAE"], marker="o", label="Persistence (MAE)")
plt.plot(df_direct.index, df_direct["MeanS_MAE"],      marker="o", label=f"Mean-{S} (MAE)")
plt.plot(df_direct.index, df_direct["Direct_GRU_MAE"], marker="o", label="Direct GRU (MAE)")
if have_rec:
    plt.plot(df_direct.index, df_direct["GRU_rec_MAE"], marker="o", label="GRU (recursive) MAE")
plt.xlabel("Horizon k (days ahead)")
plt.ylabel("MAE (°C)")
plt.title("Direct vs Recursive — MAE by horizon")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------------------
