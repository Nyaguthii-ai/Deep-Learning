# Predicting temperature using the min-max values for the past 10years
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

#Step 0: Load and Inspect the Data The dataset is stored in a CSV file with two columns:
# Load raw CSV
df = pd.read_csv("1_Daily_minimum_temps.csv")

# Parse Date (format is MM/DD/YY, e.g., "01/13/81")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

# Set Date as index
df.set_index("Date", inplace=True)

# Ensure Temp is numeric
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")

# Basic info
print("Shape:", df.shape)
print(df.head(20))
print(df["Temp"].describe())

# Check for missing values
print("Missing values:", df["Temp"].isna().sum())
# --------------------------------------------------------------------------------------------
# Step 1: Full 10-Year View
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["Temp"], linewidth=1)
plt.title("Daily Minimum Temperatures in Melbourne (1981–1990)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
# Step 2: Zoom into One Year
plt.figure(figsize=(12, 4))
plt.plot(df.loc["1981"].index, df.loc["1981"]["Temp"], linewidth=1.2)
plt.title("Daily Minimum Temperatures in 1981", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
#Step 3: Smoothing for Clarity
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["Temp"], alpha=0.35, linewidth=0.8, label="Daily")
plt.plot(df["Temp"].rolling(window=30, min_periods=1).mean(),
         linewidth=1.5, color="darkorange", label="30-day rolling mean")
plt.title("Daily Temps with 30-day Rolling Mean (1981–1990)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
# Code: Create Sliding Windows
def create_sliding_windows(values, window_size=3):
    """
    Convert a 1D array (or 2D with shape (*,1)) into (X, y) pairs using a sliding window.
    X shape: (num_samples, window_size)
    y shape: (num_samples,)
    """
    values = np.array(values)
    if values.ndim == 2:
        values = values.flatten()
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i])  # previous window_size values
        y.append(values[i])                # next value
    return np.array(X), np.array(y)

# Example with a short sequence
toy_values = [20.7, 17.9, 18.8, 14.6, 15.8, 15.8]
X, y = create_sliding_windows(toy_values, window_size=3)

print("Inputs (X):\n", X)
print("Targets (y):\n", y)
# --------------------------------------------------------------------------------------------
# Applying to the Real Temperature Dataset
# Use the full temperature column from the dataset
temps = df["Temp"].values

# Create supervised pairs with a 7-day window
X_real, y_real = create_sliding_windows(temps, window_size=7)

print("Shape of inputs (X):", X_real.shape)
print("Shape of targets (y):", y_real.shape)

# Show the first 5 samples
for i in range(5):
    print(f"X[{i}] = {X_real[i]}, y[{i}] = {y_real[i]}")
# --------------------------------------------------------------------------------------------
# CT_Task - 14-day sliding windows (raw)
assert 'create_sliding_windows' in globals(), "❌ create_sliding_windows helper not found."

CT_values_raw = CT_df["Temp"].values
CT_X14_raw, CT_y14_raw = create_sliding_windows(CT_df["Temp"].values, window_size=14)

print("CT_X14_raw shape:", CT_X14_raw.shape, "| CT_y14_raw shape:", CT_y14_raw.shape)
for i in range(5):
    print(f"X[{i}] = {CT_X14_raw[i]}, y[{i}] = {CT_y14_raw[i]}")

# --------------------------------------------------------------------------------------------
# visualize the split so we can **see** which years are in train vs. test.
# Choose a split date so we get ~8 years train (1981–1988) and 2 years test (1989–1990)
split_date = pd.Timestamp("1989-01-01")

# Time-aware split
df_train = df.loc[: split_date - pd.Timedelta(days=1)]
df_test  = df.loc[split_date :]

print("Train period:", df_train.index.min().date(), "→", df_train.index.max().date(), "| rows:", len(df_train))
print("Test  period:", df_test.index.min().date(),  "→", df_test.index.max().date(),  "| rows:", len(df_test))
# --------------------------------------------------------------------------------------------
# plot the series again and color the time spans for train (left) and test (right).
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(df.index, df["Temp"], linewidth=1, label="Daily min temp")

# Shade regions
ax.axvspan(df_train.index.min(), df_train.index.max(), alpha=0.25, label="Train (1981–1988)")
ax.axvspan(df_test.index.min(),  df_test.index.max(),  alpha=0.15, label="Test (1989–1990)")

# Formatting
ax.set_title("Time-Aware Train/Test Split", fontsize=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
# Reuse the helper from Section 3
# def create_sequences(values, window_size=3): ...

window_size = 7  # one week of history → predict next day

X_train, y_train = create_sliding_windows(df_train["Temp"].values, window_size=window_size)
X_test,  y_test  = create_sliding_windows(df_test["Temp"].values,  window_size=window_size)

print("Train X shape:", X_train.shape, "| y shape:", y_train.shape)
print("Test  X shape:", X_test.shape,  "| y shape:",  y_test.shape)
# --------------------------------------------------------------------------------------------
# Scaling Train & Test Separately
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on training set
scaler.fit(df_train[["Temp"]])

# Transform train and test
train_scaled = scaler.transform(df_train[["Temp"]])
test_scaled  = scaler.transform(df_test[["Temp"]])

print("Train scaled shape:", train_scaled.shape)
print("Test scaled shape:", test_scaled.shape)
print("First 5 scaled train values:", train_scaled[:5].flatten())
# --------------------------------------------------------------------------------------------
# Pick the first 100 days for clarity
days = range(100)

plt.figure(figsize=(12, 5))

# Raw values
plt.subplot(1, 2, 1)
plt.plot(df_train["Temp"].values[:100], color="steelblue")
plt.title("Raw Temperatures (°C)")
plt.xlabel("Day")
plt.ylabel("Temp (°C)")

# Scaled values
plt.subplot(1, 2, 2)
plt.plot(train_scaled[:100], color="darkorange")
plt.title("Scaled Temperatures (0–1)")
plt.xlabel("Day")
plt.ylabel("Scaled Value")

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
# Create sliding windows on the scaled data: 
window_size = 7
X_train, y_train = create_sliding_windows(train_scaled, window_size)
X_test,  y_test  = create_sliding_windows(test_scaled,  window_size)

print("Train X shape:", X_train.shape, "| y shape:", y_train.shape)
print("Test  X shape:", X_test.shape,  "| y shape:",  y_test.shape)
print("Example X_train[0]:", X_train[0].flatten(), "→ y:", y_train[0])
# --------------------------------------------------------------------------------------------
# Corrected code to avoid shape error
CT_split_date = pd.Timestamp("1989-01-01")
CT_df_train = CT_df.loc[: CT_split_date - pd.Timedelta(days=1)]
CT_df_test  = CT_df.loc[CT_split_date :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(CT_df_train[["Temp"]])

CT_train_scaled = scaler.transform(CT_df_train[["Temp"]])
CT_test_scaled  = scaler.transform(CT_df_test[["Temp"]])

assert 'create_sliding_windows' in globals(), "❌ create_sliding_windows helper not found."
CT_X14_train, CT_y14_train = create_sliding_windows(CT_train_scaled, window_size=14)
CT_X14_test,  CT_y14_test  = create_sliding_windows(CT_test_scaled,  window_size=14)

print("Train shapes:", CT_X14_train.shape, CT_y14_train.shape)
print("Test  shapes:", CT_X14_test.shape,  CT_y14_test.shape)
for i in range(5):
    print(f"Train X[{i}] = {CT_X14_train[i].flatten()}, y = {CT_y14_train[i]}")
# --------------------------------------------------------------------------------------------
# Preparing PyTorch Datasets & Loaders
# 1) NumPy → PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)  # (N_train, seq_len, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # (N_train, 1)

X_test_t  = torch.tensor(X_test,  dtype=torch.float32)  # (N_test,  seq_len, 1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)  # (N_test,  1)

# 2) Wrap in datasets
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# 3) Build DataLoaders (batch-first batches by default)
BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)   # shuffle only for train
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 4) Inspect a batch
xb, yb = next(iter(train_loader))
print("Batch X shape (batch-first):", xb.shape)  # expected: (batch, seq_len, input_size) e.g., (64, 7, 1)
print("Batch y shape:", yb.shape)                # expected: (batch, 1)

# If a model expects time-first (e.g., batch_first=False), we can permute:
xb_seq_first = xb.permute(1, 0, 2)  # (seq_len, batch, input_size)
print("Permuted X for time-first model:", xb_seq_first.shape)