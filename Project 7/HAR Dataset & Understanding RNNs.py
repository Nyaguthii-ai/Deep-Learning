import numpy as np
import matplotlib.pyplot as plt

# For now, simulate one sequence for demonstration.
# Later we will replace with real HAR data.
timesteps = np.arange(128)
signal = np.sin(2 * np.pi * timesteps / 32) + 0.1 * np.random.randn(128)

plt.figure(figsize=(10,4))
plt.plot(timesteps, signal, label="Accelerometer X (Walking)")
plt.xlabel("Timestep")
plt.ylabel("Acceleration (g)")
plt.title("Example HAR Signal Window (Simulated)")
plt.legend()
plt.show()
# -------------------------------------------------------------------------------
# Imports & config for this section

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Update this path to point to your extracted "UCI HAR Dataset" folder
BASE_DIR = Path("UCI HAR Dataset")  
assert BASE_DIR.exists(), f"BASE_DIR does not exist: {BASE_DIR}"

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
NUM_CLASSES = 6
# -------------------------------------------------------------------------------
# Loader for the 9-channel raw inertial signals + labels

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x","total_acc_y","total_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
]

def load_split(split="train", base_dir=BASE_DIR):
    """
    Returns:
        X: np.ndarray, shape (N, 128, 9)
        y: np.ndarray, shape (N,) with class indices in {0..5}
        subjects: np.ndarray, shape (N,) with subject IDs (1..30)
    """
    sig_dir = base_dir / split / "Inertial Signals"
    matrices = []
    for name in SIGNAL_NAMES:
        fpath = sig_dir / f"{name}_{split}.txt"
        arr = np.loadtxt(fpath, dtype=np.float32)  # shape: (N, 128)
        matrices.append(arr[:, :, None])           # -> (N, 128, 1)

    X = np.concatenate(matrices, axis=2)          # (N, 128, 9)
    y = np.loadtxt(base_dir / split / f"y_{split}.txt", dtype=np.int64)
    y = y - 1  # convert {1..6} -> {0..5}
    subjects = np.loadtxt(base_dir / split / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subjects

X_train_np, y_train_np, subj_train = load_split("train", BASE_DIR)
X_test_np,  y_test_np,  subj_test  = load_split("test",  BASE_DIR)

print("Train X shape:", X_train_np.shape, "(N, 128, 9)")
print("Train y shape:", y_train_np.shape,  "(N,)  class indices in {0..5}")
print("Test  X shape:", X_test_np.shape)
print("Test  y shape:", y_test_np.shape)
print("Unique train labels:", np.unique(y_train_np))
# -------------------------------------------------------------------------------
# Optional: per-channel standardization using TRAIN statistics

EPS = 1e-8 # 1e-8 to avoid division by zero

# Compute mean/std over (N, T) for each of the 9 channels on TRAIN only
# X_train_np: (N, 128, 9)
train_mean = X_train_np.mean(axis=(0,1), keepdims=True)  # shape (1,1,9)
# axis(0,1) means "collapse N and T, keep channels" and get mean/std per channel. keepdims=True keeps the 3D shape for broadcasting
# axis=0 would collapse N only, giving (1, 128, 9) mean per timestep
# axis=1 would collapse T only, giving (N, 1, 9) mean per sample
# axis=2 would collapse channels only, giving (N, 128, 1) mean per sample/timestep
# We want (1,1,9) to broadcast correctly when subtracting from (N,128,9)
# So we get one mean/std per channel over the entire TRAIN set

train_std  = X_train_np.std(axis=(0,1),  keepdims=True)  # shape (1,1,9)

X_train_std = (X_train_np - train_mean) / (train_std + EPS)
X_test_std  = (X_test_np  - train_mean) / (train_std + EPS)  # use train stats!

print("Per-channel mean (train):", train_mean.reshape(-1)[:3], "... (showing first 3)")
print("Per-channel std  (train):", train_std.reshape(-1)[:3],  "... (showing first 3)")
# -------------------------------------------------------------------------------
# Convert to PyTorch tensors

X_train_t = torch.from_numpy(X_train_std)  # (N, 128, 9), float32
y_train_t = torch.from_numpy(y_train_np)   # (N,), int64

X_test_t  = torch.from_numpy(X_test_std)
y_test_t  = torch.from_numpy(y_test_np)

print(X_train_t.shape, y_train_t.shape, X_test_t.shape, y_test_t.shape)
# -------------------------------------------------------------------------------
# Dataset preview: show first sequence (a small slice) and its label name

def preview_sequence(X_np, y_np, index=0, timesteps_to_show=10):
    seq = X_np[index]  # (128, 9)
    label_idx = int(y_np[index])
    label_name = ACTIVITY_MAP[label_idx + 1]
    print(f"Sample index: {index}")
    print(f"Label index/name: {label_idx} / {label_name}")
    print(f"Sequence shape: {seq.shape} (timesteps, features)")
    print("First few timesteps (first 3 features):")
    print(seq[:timesteps_to_show, :3])  # show first 3 channels for brevity

preview_sequence(X_train_std, y_train_np, index=0, timesteps_to_show=8)
#  -------------------------------------------------------------------------------
# Plot sample sequences for two activities: WALKING (1-> index 0) vs SITTING (4-> index 3)
# We'll plot 3 channels for clarity (e.g., body_acc_x/y/z)

def find_first_index(y_np, class_index):
    idxs = np.where(y_np == class_index)[0]
    return int(idxs[0]) if len(idxs) > 0 else None

idx_walk   = find_first_index(y_train_np, 0)  # WALKING -> 1 -> index 0
idx_sit    = find_first_index(y_train_np, 3)  # SITTING -> 4 -> index 3

print("Example indices -> WALKING:", idx_walk, " SITTING:", idx_sit)

def plot_three_channels(seq, title, channels=(0,1,2)):
    t = np.arange(seq.shape[0])  # 0..127
    plt.figure(figsize=(10,4))
    for c in channels:
        plt.plot(t, seq[:, c], label=f"channel {c}")
    plt.xlabel("Timestep")
    plt.ylabel("Standardized value")
    plt.title(title)
    plt.legend()
    plt.show()

if idx_walk is not None:
    plot_three_channels(X_train_std[idx_walk], "WALKING: example (3 channels)")

if idx_sit is not None:
    plot_three_channels(X_train_std[idx_sit], "SITTING: example (3 channels)")
# -------------------------------------------------------------------------------
# Task — Plot a LAYING sample (channels 0,1,2)
# Find first index of a target class in y_train_np (0..5). LAYING is index 5.
def CT_find_first_index(y_np, class_index):
    idxs = np.where(y_np == class_index)[0]
    return int(idxs[0]) if len(idxs) > 0 else None

CT_idx_laying = CT_find_first_index(y_train_np, 5)  # LAYING is index 5

# Pull standardized sequence (128, 9)
CT_seq_laying = X_train_std[CT_idx_laying]   # expect X_train_std

# Channels to show
CT_channels = (0, 1, 2)

# Plot
t = np.arange(CT_seq_laying.shape[0])
plt.figure(figsize=(10,4))
for c in CT_channels:
    plt.plot(t, CT_seq_laying[:, c], label=f"channel {c}")
plt.xlabel("Timestep")
plt.ylabel("Standardized value")
plt.title("LAYING: example (3 channels)")
plt.legend()
plt.show()
# -------------------------------------------------------------------------------
# Minimal RNN (batch=1, hidden_size=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3.5 Select one sequence and one label
seq = X_train_t[0]       # shape (128, 9)
label = y_train_t[0].item()

print("Sequence shape:", seq.shape, "(timesteps=128, features=9)")
print("Label index:", label, "->", ACTIVITY_MAP[label+1])

# Reshape to (seq_len, batch, input_size)
seq_in = seq.unsqueeze(1)  # (128, 1, 9)
print("Input to RNN shape:", seq_in.shape)
# --------------------------------------------------------------------------------
# Define RNN with hidden_size=3
rnn2 = nn.RNN(input_size=9, hidden_size=3, batch_first=False)

# Forward pass on same sequence
out_seq2, h_n2 = rnn2(seq_in)   # out_seq2: (128, 1, 3), h_n2: (1, 1, 3)

# Classification head: hidden -> 6 classes
fc2 = nn.Linear(3, NUM_CLASSES)
logits2 = fc2(h_n2.squeeze(0))   # shape (1,6)
probs2 = F.softmax(logits2, dim=1)

print("out_seq2 shape:", out_seq2.shape, "(seq_len=128, batch=1, hidden=3)")
print("h_n2 shape:", h_n2.shape, "(num_layers=1, batch=1, hidden=3)")
print("logits2 shape:", logits2.shape, "(1,6)")
print("probs2 shape:", probs2.shape, "(1,6)")
# -------------------------------------------------------------------------------
# Inspect hidden states
print("First hidden state (3-dim):", out_seq2[0,0,:].detach().numpy())
print("Last hidden state (3-dim):", h_n2[0,0,:].detach().numpy())
#  -------------------------------------------------------------------------------
# Plot all 3 hidden dimensions over time
h_values2 = out_seq2[:,0,:].detach().numpy()  # shape (128,3)

plt.figure(figsize=(10,4))
for dim in range(3):
    plt.plot(range(128), h_values2[:,dim], label=f"hidden dim {dim+1}")
plt.xlabel("Timestep")
plt.ylabel("Hidden state value")
plt.title("Evolution of 3 hidden dimensions (hidden_size=3)")
plt.legend()
plt.show()
# -------------------------------------------------------------------------------
# Take a mini-batch of 32 sequences. Run them through an RNN with `hidden_size=32`. 
# Create a batch of 32 sequences
batch_size = 32
X_batch = X_train_t[:batch_size]   # (32, 128, 9)
y_batch = y_train_t[:batch_size]   # (32,)

print("Batch shape (batch-first):", X_batch.shape)

# Convert to time-first for default RNN
X_batch_tf = X_batch.permute(1, 0, 2)   # (128, 32, 9)
print("Time-first shape:", X_batch_tf.shape)
# --------------------------------------------------------------------------------
# Define RNN with hidden_size=32
rnn3 = nn.RNN(input_size=9, hidden_size=32, batch_first=False)

# Forward pass
out_seq3, h_n3 = rnn3(X_batch_tf)   # out_seq3: (128, 32, 32), h_n3: (1, 32, 32)

# Classification head
fc3 = nn.Linear(32, NUM_CLASSES)
logits3 = fc3(h_n3.squeeze(0))   # (32,6)
probs3 = F.softmax(logits3, dim=1)

print("out_seq3 shape:", out_seq3.shape, "(seq_len=128, batch=32, hidden=32)")
print("h_n3 shape:", h_n3.shape, "(layers=1, batch=32, hidden=32)")
print("logits3 shape:", logits3.shape, "(batch=32, classes=6)")
print("probs3 sum per row (first):", probs3[0].sum().item())
# ----------------------------------------------------------------------------------
# Inspect first few probability distributions
for i in range(3):
    probs_row = probs3[i].detach().numpy()
    pred_idx = int(probs_row.argmax())
    print(f"Sample {i}: true={ACTIVITY_MAP[int(y_batch[i])+1]}, "
          f"pred={ACTIVITY_MAP[pred_idx+1]}, "
          f"probs={np.round(probs_row,3)}")
# ----------------------------------------------------------------------------------
# Task: Convert batch-first ↔ time-first
def CT_to_time_first(x_btf):
    """
    x_btf: (B, T, F) -> return (T, B, F)
    """
    assert x_btf.ndim == 3, "x_btf must be 3D (B,T,F)"
    return x_btf.permute(1, 0, 2)

def CT_to_batch_first(x_tbf):
    """
    x_tbf: (T, B, F) -> return (B, T, F)
    """
    assert x_tbf.ndim == 3, "x_tbf must be 3D (T,B,F)"
    return x_tbf.permute(1, 0, 2)

# Quick self-check on a dummy tensor
CT_dummy = torch.zeros(4, 128, 9)   # (B,T,F)
CT_tfirst = CT_to_time_first(CT_dummy)
CT_back   = CT_to_batch_first(CT_tfirst)

print("Shapes:", CT_dummy.shape, "->", CT_tfirst.shape, "->", CT_back.shape)
# ----------------------------------------------------------------------------------
# Using out_seq3 from Section 5 (shape: 128, 32, 32)

# Sequence classification: use only last hidden state
last_hidden = h_n3.squeeze(0)        # shape (32, 32)
logits_cls = fc3(last_hidden)        # shape (32,6)

# Sequence labeling: use all hidden states
out_seq_flat = out_seq3.reshape(-1, 32)  # (128*32, 32)
logits_lbl = fc3(out_seq_flat)           # (128*32, 6)

print("Classification logits shape:", logits_cls.shape, "(batch=32, classes=6)")
print("Labeling logits shape:", logits_lbl.shape, "(timesteps*batch, classes=6)")
# ----------------------------------------------------------------------------------
# Sequence classification vs. labeling with batch=6
# (A) Classification from last hidden
CT_last_hidden_b6 = CT_hn_b6.squeeze(0)          # (6,6)
CT_logits_cls_b6 = CT_fc_b6_h6(CT_last_hidden_b6)

# (B) Labeling from all time steps
CT_out_seq_flat_b6 = CT_out_seq_b6.reshape(-1, 6) # (128*6, 6)
CT_logits_lbl_b6 = CT_fc_b6_h6(CT_out_seq_flat_b6)

print("cls:", CT_logits_cls_b6.shape, "lbl:", CT_logits_lbl_b6.shape)
