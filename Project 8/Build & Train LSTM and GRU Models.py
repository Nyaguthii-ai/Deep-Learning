import pandas as pd
import random, numpy as np, torch
from sklearn.preprocessing import MinMaxScaler, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------
# Data load and peek a batch (confirm shapes)
# 1) Load the dataset (adjust path if needed)
# Expecting two columns: Date (MM/DD/YY), Temp
df = pd.read_csv("1_Daily_minimum_temps.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
df = df.set_index("Date").dropna(subset=["Temp"])

# 2) Time-aware split (1981–1988 train, 1989–1990 test)
split_date = pd.Timestamp("1989-01-01")
df_train = df.loc[: split_date - pd.Timedelta(days=1)]
df_test  = df.loc[split_date :]

# 3) Scale with train-only fit (MinMax to [0,1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[["Temp"]])
train_scaled = scaler.transform(df_train[["Temp"]])
test_scaled  = scaler.transform(df_test[["Temp"]])

# 4) Sliding windows → (N, window_size, 1) and (N, 1)

def create_sliding_windows(series, window_size=7):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X); y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape((-1, 1))

WINDOW_SIZE = 7
X_train, y_train = create_sliding_windows(train_scaled, WINDOW_SIZE)
X_test,  y_test  = create_sliding_windows(test_scaled,  WINDOW_SIZE)

# 5) TensorDataset + DataLoader (batch-first)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

CT_train_loader_10 = DataLoader(train_ds, batch_size=10, shuffle=False)

# --- Peek a batch and confirm shapes ---
xb, yb = next(iter(train_loader))
print("Batch X shape (batch-first):", xb.shape)   # expect (B, 7, 1)
print("Batch y shape:", yb.shape)                 # expect (B, 1)

# --- Show time-first view if needed ---
xb_time_first = xb.permute(1, 0, 2)
print("Time-first view of X:", xb_time_first.shape)  # (7, B, 1)

#-----------------------------------------------------------------------------------------
# --- Task: Peek at last batch from CT_train_loader_10 ---
CT_xb10_last, CT_yb10_last = None, None
for _xb, _yb in CT_train_loader_10:       # iterate through all batches to get the last one
    CT_xb10_last, CT_yb10_last = _xb, _yb

print("Last batch X (batch-first):", CT_xb10_last.shape)
CT_time_first = CT_xb10_last.permute(1, 0, 2)
CT_back = CT_time_first.permute(1, 0, 2)
print("Round-trip equal shape:", CT_back.shape == CT_xb10_last.shape)
#-----------------------------------------------------------------------------------------
#  Model code (batch-first)
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",     # default; explicit for clarity
            batch_first=batch_first,
            bidirectional=False
        )
        self.head = nn.Linear(hidden_size, 1)  # predict one value (next day)

    def forward(self, x):
        """
        x: (B, S, F) because we use batch_first=True (e.g., (B, 7, 1))
        returns: (B, 1)
        """
        B = x.size(0)

        # Initialize hidden state h0 per batch (layers, batch, hidden)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        # RNN forward: out has hidden for every time step
        out, hn = self.rnn(x, h0)   # out: (B, S, H) since batch_first=True

        # Take last time step’s hidden state
        h_last = out[:, -1, :]      # (B, H)

        # Linear head to scalar prediction
        y_hat = self.head(h_last)   # (B, 1)
        return y_hat
    
# Quick Sanity Checks
# Ensure we have a batch from Section 2 (or the fallback there created loaders)
xb, yb = next(iter(train_loader))     # xb: (B, 7, 1), yb: (B, 1)

model_rnn = RNNModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
with torch.no_grad():
    y_hat = model_rnn(xb)
print("Input shape :", xb.shape)      # (B, 7, 1)
print("Output shape:", y_hat.shape)   # (B, 1)
#-----------------------------------------------------------------------------------------
# Main Model: LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (B, S, F) when batch_first=True (e.g., (B, 7, 1))
        returns: (B, 1)
        """
        B = x.size(0)
        # init states on the same device/dtype as x
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))   # out: (B, S, H) if batch_first=True
        # Option A: take last timestep from `out`
        h_last = out[:, -1, :]                   # (B, H)
        # Option B (equivalent): final layer's hidden state
        # h_last = hn[-1]                        # (B, H)

        y_hat = self.head(h_last)                # (B, 1)
        return y_hat

# ✅ Sanity run on a real batch
xb, yb = next(iter(train_loader))               # xb: (B, 7, 1)
model_lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True).eval()

with torch.no_grad():
    y_hat = model_lstm(xb)

print("Input shape :", xb.shape)                # (B, 7, 1)
print("Output type :", type(y_hat))
print("Output shape:", y_hat.shape)             # (B, 1)
print("Example preds:", y_hat[:5].flatten())
#-----------------------------------------------------------------------------------------
# Alternative: GRU (lighter gating)¶
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False
        )
        self.head = nn.Linear(hidden_size, 1)  # predict one value (next day)

    def forward(self, x):
        """
        x: (B, S, F) because we use batch_first=True (e.g., (B, 7, 1))
        returns: (B, 1)
        """
        B = x.size(0)

        # Initialize hidden state h0 (layers, batch, hidden), match device/dtype
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        # GRU forward:
        # out: (B, S, H) if batch_first=True
        # hn:  (layers, B, H) final hidden state
        out, hn = self.gru(x, h0)

        # Take the last time step’s hidden state
        h_last = out[:, -1, :]     # (B, H)

        # Linear head to scalar prediction
        y_hat = self.head(h_last)  # (B, 1)
        return y_hat
    
# Ensure we have a batch (from Section 2 or its self-contained fallback)
xb, yb = next(iter(train_loader))        # xb: (B, 7, 1), yb: (B, 1)

model_gru = GRUModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True).eval()
with torch.no_grad():
    y_hat = model_gru(xb)

print("Input shape :", xb.shape)         # (B, 7, 1)
print("Output shape:", y_hat.shape)      # (B, 1)
print("Example preds:", y_hat[:5].flatten())
#-----------------------------------------------------------------------------------------
# Train all three in one loop 
def train_one_model(model, train_loader, val_loader, epochs=20, lr=1e-3, clip=None, verbose=True):
    """
    Train a sequence model (RNN/LSTM/GRU head) on scaled targets.
    Assumes:
      - inputs:  (B, S, F) with batch_first=True (as built in our loaders)
      - targets: (B, 1), scaled to [0,1]
    Returns:
      model (trained in-place), history: {'train_loss': [...], 'val_loss': [...]}
    """
    device = torch.device("cpu")  # CPU-only per project constraints
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()

            if clip is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        train_loss = running / max(n, 1)
        history["train_loss"].append(train_loss)

        # ---- Validate ----
        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                y_hat = model(xb)
                loss = criterion(y_hat, yb)
                running += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = running / max(n, 1)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

    return model, history

def plot_loss(history, title="Training vs Validation Loss"):
    """Plot train/val loss curves for a single model."""
    plt.figure(figsize=(7,4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled space)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # x ticks from 1 to len(history) with step 1
    plt.xticks(range(0, len(history["train_loss"]), 1))
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------
# train each model with the same loop (CPU-friendly defaults)
# If these classes were defined in Sections 5–7:
#   RNNModel, LSTMModel, GRUModel

EPOCHS = 20
LR     = 1e-3
CLIP   = 1.0

# RNN baseline
model_rnn = RNNModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_rnn, hist_rnn = train_one_model(model_rnn, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)
plot_loss(hist_rnn, title="RNN: Train vs Val Loss")

# LSTM
model_lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_lstm, hist_lstm = train_one_model(model_lstm, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)
plot_loss(hist_lstm, title="LSTM: Train vs Val Loss")

# GRU
model_gru = GRUModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_gru, hist_gru = train_one_model(model_gru, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)
plot_loss(hist_gru, title="GRU: Train vs Val Loss")
#-----------------------------------------------------------------------------------------
# predict → inverse transform → metrics → plots
def predict_on_loader(model, loader):
    """
    Returns:
      y_true_scaled: (N, 1)
      y_pred_scaled: (N, 1)
    """
    model.eval()
    ys, yhs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yhat = model(xb)
            ys.append(yb.cpu().numpy())
            yhs.append(yhat.cpu().numpy())
    y_true_scaled = np.vstack(ys)
    y_pred_scaled = np.vstack(yhs)
    return y_true_scaled, y_pred_scaled


def inverse_to_celsius(arr_scaled, scaler):
    """arr_scaled: (N,1) → inverse_transform → (N,1) in °C"""
    return scaler.inverse_transform(arr_scaled)


def evaluate_in_celsius(modmodel, loader, scaler):
    """
    Returns:
      metrics dict and (y_true_C, y_pred_C) arrays, each (N,1)
    """
    y_s, yhat_s = predict_on_loader(model, loader)
    y_C    = inverse_to_celsius(y_s,    scaler)
    yhat_C = inverse_to_celsius(yhat_s, scaler)

    mae  = mean_absolute_error(y_C, yhat_C)
    mse  = mean_squared_error(y_C, yhat_C)
    rmse = mse ** 0.5
    r2   = r2_score(y_C, yhat_C)

    metrics = {"MAE_C": float(mae), "RMSE_C": float(rmse), "R2": float(r2)}
    return metrics, y_C, yhat_C


def plot_line_slice(y_C, yhat_C, first_n=200, title="Actual vs Predicted (°C) — first 200 test points"):
    n = min(first_n, len(y_C))
    plt.figure(figsize=(10,4))
    plt.plot(y_C[:n],    label="Actual (°C)")
    plt.plot(yhat_C[:n], label="Predicted (°C)")
    plt.xlabel("Test index")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------
# Evaluate and Visualise 

# RNN
metrics_rnn, yC_rnn, yhC_rnn = evaluate_in_celsius(model_rnn, test_loader, scaler)
print("RNN:", metrics_rnn)
plot_line_slice(yC_rnn, yhC_rnn, first_n=200, title="RNN — Actual vs Predicted (°C) — first 200")

# LSTM
metrics_lstm, yC_lstm, yhC_lstm = evaluate_in_celsius(model_lstm, test_loader, scaler)
print("LSTM:", metrics_lstm)
plot_line_slice(yC_lstm, yhC_lstm, first_n=200, title="LSTM — Actual vs Predicted (°C) — first 200")

# GRU
metrics_gru, yC_gru, yhC_gru = evaluate_in_celsius(model_gru, test_loader, scaler)
print("GRU:", metrics_gru)
plot_line_slice(yC_gru, yhC_gru, first_n=200, title="GRU — Actual vs Predicted (°C) — first 200")
#-----------------------------------------------------------------------------------------
# Moving Average for baseline Comparison

# MA(7) prediction for each test window (still in scaled space)
yhat_ma_test_scaled = X_test.mean(axis=1)  # shape: (N_test, 1)

# Invert scaling to °C to match the RNN plot
y_true_c = scaler.inverse_transform(y_test).squeeze()
yhat_ma_c = scaler.inverse_transform(yhat_ma_test_scaled).squeeze()

# Plot first 200 like the RNN figure
n = 200
plt.figure(figsize=(12,4))
plt.plot(np.arange(n), y_true_c[:n], label="Actual (°C)")
plt.plot(np.arange(n), yhat_ma_c[:n], label="MA(7) (°C)")
plt.title("MA(7) — Actual vs Predicted (°C) — first 200")
plt.xlabel("Test index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------------
# Side-by-side comparison 

# === MA(7) baseline on validation (fallback to test if no val set) ===
X_eval = X_val if 'X_val' in globals() else X_test
y_eval = y_val if 'y_val' in globals() else y_test

# MA(7) prediction in *scaled* space to match val_loss
yhat_ma_eval_scaled = X_eval.mean(axis=1)                    # (N,1)
val_mse_ma = float(np.mean((y_eval.squeeze() - yhat_ma_eval_scaled.squeeze())**2))

# Also compute metrics in °C for the table
y_eval_C  = scaler.inverse_transform(y_eval).ravel()
yhat_ma_C = scaler.inverse_transform(yhat_ma_eval_scaled).ravel()
metrics_ma = {
    "MAE_C":  float(np.mean(np.abs(y_eval_C - yhat_ma_C))),
    "RMSE_C": float(np.sqrt(np.mean((y_eval_C - yhat_ma_C)**2))),
    "R2":     float(r2_score(y_eval_C, yhat_ma_C)),
}

# --- Metrics table (now includes MA(7)) ---
df_metrics = pd.DataFrame.from_dict(
    {
        "RNN":   metrics_rnn,
        "LSTM":  metrics_lstm,
        "GRU":   metrics_gru,
        "MA(7)": metrics_ma,
    },
    orient="index",
)
df_metrics = df_metrics[["MAE_C", "RMSE_C", "R2"]]
display(df_metrics.sort_values("RMSE_C").round(3))

# --- Validation curves + MA(7) flat line ---
plt.figure(figsize=(8,5))
plt.plot(hist_rnn["val_loss"],  label="RNN (val)")
plt.plot(hist_lstm["val_loss"], label="LSTM (val)")
plt.plot(hist_gru["val_loss"],  label="GRU (val)")

E = len(hist_rnn["val_loss"])
plt.plot([val_mse_ma]*E, label="MA(7) (val)", linestyle="--")  # flat baseline

plt.xlabel("Epoch")
plt.ylabel("MSE (scaled space)")
plt.title("Validation Loss — RNN vs LSTM vs GRU (+ MA(7))")
plt.xticks(range(0, E, 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------------
# Save the weights
import os

# Make a folder to keep things tidy
os.makedirs("artifacts", exist_ok=True)

# Save just the state_dict (weights) for each model
torch.save(model_rnn.state_dict(),  "artifacts/rnn_state.pkl")
torch.save(model_lstm.state_dict(), "artifacts/lstm_state.pkl")
torch.save(model_gru.state_dict(),  "artifacts/gru_state.pkl")

print("Saved files in ./artifacts:")
print(os.listdir("artifacts"))