import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Dataset Preparation (Train/Test)
# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Activity label mapping (HAR uses 1..6 in files; we convert to 0..5)
ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
NUM_CLASSES = 6

# IMPORTANT: Set this to your local "UCI HAR Dataset" folder
BASE_DIR = Path("UCI HAR Dataset")  
assert BASE_DIR.exists(), f"BASE_DIR does not exist: {BASE_DIR}"
print("Using dataset folder:", BASE_DIR)
# ------------------------------------------------------------------------------------------------
# Utilities to load HAR raw inertial signals (9 channels) + labels, self-contained

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x","total_acc_y","total_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
]

def load_split(split: str, base_dir: Path):
    """
    Load HAR raw inertial signals and labels for a split ("train" or "test").

    Returns:
        X: np.ndarray, shape (N, 128, 9), float32
        y: np.ndarray, shape (N,), int64 in {0..5}
        subjects: np.ndarray, shape (N,), int64 in {1..30}
    """
    sig_dir = base_dir / split / "Inertial Signals"
    matrices = []
    for name in SIGNAL_NAMES:
        fpath = sig_dir / f"{name}_{split}.txt"
        arr = np.loadtxt(fpath, dtype=np.float32)   # (N, 128)
        matrices.append(arr[:, :, None])            # -> (N, 128, 1)
    X = np.concatenate(matrices, axis=2)            # (N, 128, 9)

    y = np.loadtxt(base_dir / split / f"y_{split}.txt", dtype=np.int64)  # 1..6
    y = y - 1  # -> {0..5}

    subjects = np.loadtxt(base_dir / split / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subjects

# Load both splits fresh in NB02
X_train_np, y_train_np, subj_train = load_split("train", BASE_DIR)
X_test_np,  y_test_np,  subj_test  = load_split("test",  BASE_DIR)

print("Train X:", X_train_np.shape, "(N, 128, 9)")
print("Train y:", y_train_np.shape,  "(N,) in {0..5}")
print("Test  X:", X_test_np.shape)
print("Test  y:", y_test_np.shape)
print("Train subjects range:", int(subj_train.min()), "to", int(subj_train.max()))
# ------------------------------------------------------------------------------------------------
# Channel-wise standardization using TRAIN statistics only (recommended)

EPS = 1e-8

# Compute per-channel mean/std over (N, T) for the TRAIN split
train_mean = X_train_np.mean(axis=(0, 1), keepdims=True)  # (1,1,9)
train_std  = X_train_np.std(axis=(0, 1), keepdims=True)   # (1,1,9)

X_train_std = (X_train_np - train_mean) / (train_std + EPS)
X_test_std  = (X_test_np  - train_mean) / (train_std + EPS)  # IMPORTANT: use train stats!

print("Sanity check (standardized) ->",
      "train mean ~", np.round(X_train_std.mean(axis=(0,1)), 4),
      "train std ~",  np.round(X_train_std.std(axis=(0,1)), 4))
# ------------------------------------------------------------------------------------------------
# Convert to PyTorch tensors and build DataLoaders (batch-first tensors)

X_train_t = torch.from_numpy(X_train_std)  # (N, 128, 9) float32
y_train_t = torch.from_numpy(y_train_np)   # (N,)        int64

X_test_t  = torch.from_numpy(X_test_std)
y_test_t  = torch.from_numpy(y_test_np)

BATCH_SIZE = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Inspect one batch
xb, yb = next(iter(train_loader))
print("Batch X shape (batch-first):", tuple(xb.shape))  # (32, 128, 9)
print("Batch y shape:", tuple(yb.shape))                # (32,)
# ------------------------------------------------------------------------------------------------
# Quick visual sanity checks: one example per activity (3 channels plotted)
import matplotlib.pyplot as plt
import numpy as np

def find_first_index(y_np, class_index):
    idxs = np.where(y_np == class_index)[0]
    return int(idxs[0]) if idxs.size > 0 else None

def plot_three_channels(seq_128x9, title, channels=(0,1,2)):
    t = np.arange(seq_128x9.shape[0])
    plt.figure(figsize=(10,4))
    for c in channels:
        plt.plot(t, seq_128x9[:, c], label=f"channel {c}")
    plt.xlabel("Timestep")
    plt.ylabel("Standardized value")
    plt.title(title)
    plt.legend()
    plt.show()

# Example indices (0=WALKING, 3=SITTING) if available
idx_walk = find_first_index(y_train_np, 0)
idx_sit  = find_first_index(y_train_np, 3)

if idx_walk is not None:
    plot_three_channels(X_train_std[idx_walk], "WALKING — example (3 channels)")

if idx_sit is not None:
    plot_three_channels(X_train_std[idx_sit], "SITTING — example (3 channels)")
# ------------------------------------------------------------------------------------------------

# CT_Task 1 — Plot 3 channels (0,1,2) for LAYING
def CT_find_first_index(y_np, class_index):
    idxs = np.where(y_np == class_index)[0]
    return int(idxs[0]) if len(idxs) > 0 else None

CT_idx_laying = CT_find_first_index(y_train_np, 5)
CT_seq_laying = X_train_np[CT_idx_laying]          # expect (128, 9)
CT_channels = (0, 1, 2)

t = np.arange(CT_seq_laying.shape[0])
plt.figure(figsize=(10, 4))
for c in CT_channels:
    plt.plot(t, CT_seq_laying[:, c], label=f"channel {c}")
plt.xlabel("Timestep")
plt.ylabel("Standardized value")
plt.title("LAYING: example (channels 0,1,2)")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------
# Save preprocessed arrays as .npy (fast, shape-preserving)

np.save("X_train_std.npy", X_train_std)  # (N_train, 128, 9), float32
np.save("y_train.npy",     y_train_np)   # (N_train,), int64
np.save("X_test_std.npy",  X_test_std)   # (N_test, 128, 9)
np.save("y_test.npy",      y_test_np)    # (N_test,)

# Also save the channel-wise normalization stats so other notebooks can reuse them if needed
np.savez("standardization_stats.npz", train_mean=train_mean, train_std=train_std)

print("Saved: X_train_std.npy, y_train.npy, X_test_std.npy, y_test.npy, standardization_stats.npz")
# ------------------------------------------------------------------------------------------------
# Save preprocessed arrays as CSV (human-readable; flattened per row)

# Column name builder: channel_order × timestep
channel_order = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x","total_acc_y","total_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
]
def make_flat_columns():
    cols = []
    for ch in channel_order:
        for t in range(128):
            cols.append(f"{ch}_t{t:03d}")
    return cols

flat_cols = make_flat_columns()

def to_wide_df(X_std: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """
    X_std: (N, 128, 9) standardized
    y:     (N,)        class indices in {0..5}
    subjects: (N,)     subject IDs
    returns a DataFrame of shape (N, 1152 + 2) with columns:
      [label, subject, body_acc_x_t000, ..., body_gyro_z_t127]
    """
    N = X_std.shape[0]
    X_flat = X_std.reshape(N, -1)  # (N, 128*9 = 1152)
    df = pd.DataFrame(X_flat, columns=flat_cols)
    df.insert(0, "subject", subjects.astype(int))
    df.insert(0, "label",   y.astype(int))  # class index {0..5}
    return df

train_df = to_wide_df(X_train_std, y_train_np, subj_train)
test_df  = to_wide_df(X_test_std,  y_test_np,  subj_test)

train_df.to_csv("har_train_standardized.csv", index=False)
test_df.to_csv("har_test_standardized.csv",   index=False)

# Save a small README-style CSV with metadata / mapping (optional but helpful)
meta = pd.DataFrame({
    "key": ["label_index_map", "channels_order", "timesteps", "standardized", "note"],
    "value": [
        "{0: WALKING, 1: WALKING_UPSTAIRS, 2: WALKING_DOWNSTAIRS, 3: SITTING, 4: STANDING, 5: LAYING}",
        str(channel_order),
        "128",
        "True (z-score per channel using train stats)",
        "Rows are one window each; columns are flattened as <channel>_t000..t127"
    ],
})
meta.to_csv("har_metadata.csv", index=False)

print("Saved: har_train_standardized.csv, har_test_standardized.csv, har_metadata.csv")
# ------------------------------------------------------------------------------------------------
# CT_Task 2 — Load CSV → DataFrame and print shape

CT_csv_path = "har_train_standardized.csv"  # e.g., "./har_sequences.csv"
assert os.path.exists(CT_csv_path), "CSV path does not exist. Please point to the saved CSV."

CT_df = pd.read_csv(CT_csv_path)
CT_df_shape = CT_df.shape
print("CT_df_shape:", CT_df_shape)
# ------------------------------------------------------------------------------------------------
# CT_Task 3 — Head + first 5 columns
assert 'CT_df' in globals(), "CT_df not found. Complete the previous task first."

CT_df_head = CT_df.head(5)
CT_df_first5cols = CT_df.columns[:5].tolist()
print("CT_df_first5cols:", CT_df_first5cols)
CT_df_head
# ------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

# 3.5 Define the model class (batch-first RNN → last hidden state → linear)
class RNNClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, num_classes=6, num_layers=1, nonlinearity="tanh"):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,   # (batch, seq, input)
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: logits (batch, num_classes)
        """
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.rnn(x)
        last_hidden = h_n[-1]          # (batch, hidden_size)
        logits = self.fc(last_hidden)   # (batch, num_classes)
        return logits

# Instantiate a small model for CPU-friendly training
model = RNNClassifier(input_size=9, hidden_size=32, num_classes=6, num_layers=1, nonlinearity="tanh")
print(model)
# ------------------------------------------------------------------------------------------------
# Parameter count (helps students connect capacity ↔ performance/compute)
def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters())

total_params = count_params(model)
print(f"Total parameters: {total_params:,}")
# ------------------------------------------------------------------------------------------------
# Dry-run forward pass on one real batch to verify shapes end-to-end
model.eval()
with torch.no_grad():
    xb, yb = next(iter(train_loader))     # from Section 2 DataLoader
    logits = model(xb)                    # (batch, 6)
    probs  = F.softmax(logits, dim=1)     # only for inspection
    pred   = probs.argmax(dim=1)          # predicted class indices

print("Input batch shape :", tuple(xb.shape))       # (B, 128, 9)
print("Logits shape      :", tuple(logits.shape))   # (B, 6)
print("Probs row sums    :", probs[0].sum().item()) # ~1.0
print("Targets shape     :", tuple(yb.shape))       # (B,)
print("Pred shape        :", tuple(pred.shape))     # (B,)
print("First 5 targets   :", yb[:5].tolist())
print("First 5 preds     :", pred[:5].tolist())
# ------------------------------------------------------------------------------------------------
from time import time

# 4.4 Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# (Optional) gradient clipping settings
USE_CLIP = True
MAX_NORM = 5.0
# ------------------------------------------------------------------------------------------------
# 4.5 Utility: accuracy on a batch (logits & targets)
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (batch, num_classes)
    targets: (batch,)
    returns: accuracy in [0,1]
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
# ------------------------------------------------------------------------------------------------
# Train / Eval loops

def train_one_epoch(model, loader, optimizer, criterion, clip=USE_CLIP, max_norm=MAX_NORM):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    for xb, yb in loader:
        # Forward
        logits = model(xb)            # (batch, 6)
        loss = criterion(logits, yb)  # scalar

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Running stats
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_examples += xb.size(0)
    
    avg_loss = total_loss / total_examples
    avg_acc  = total_correct / total_examples
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for xb, yb in loader:
        logits = model(xb)
        if criterion is not None:
            total_loss += criterion(logits, yb).item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_examples += xb.size(0)

    avg_loss = (total_loss / total_examples) if criterion is not None else None
    avg_acc  = total_correct / total_examples
    return avg_loss, avg_acc
# ------------------------------------------------------------------------------------------------
# Train for a few epochs (CPU-friendly)
EPOCHS = 10

history = {
    "train_loss": [],
    "train_acc":  [],
    "test_loss":  [],
    "test_acc":   [],
}

t0 = time()
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    te_loss, te_acc = evaluate(model, test_loader, criterion)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["test_loss"].append(te_loss)
    history["test_acc"].append(te_acc)

    print(f"Epoch {epoch:02d} | "
          f"train_loss={tr_loss:.4f} acc={tr_acc*100:5.1f}% | "
          f"test_loss={te_loss:.4f} acc={te_acc*100:5.1f}%")

t1 = time()
print(f"Total training time ~ {t1 - t0:.1f}s (CPU)")
# ------------------------------------------------------------------------------------------------
# Save the trained model weights
torch.save(model.state_dict(), "rnn_har.pth")
print("Model saved as rnn_har.pth")
# ------------------------------------------------------------------------------------------------
# Plot training curves (loss and accuracy)

import matplotlib.pyplot as plt

epochs = range(1, EPOCHS+1)

plt.figure(figsize=(12,4))
# Loss
plt.subplot(1,2,1)
plt.plot(epochs, history["train_loss"], label="Train loss")
plt.plot(epochs, history["test_loss"],  label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Loss over epochs")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(epochs, [a*100 for a in history["train_acc"]], label="Train acc")
plt.plot(epochs, [a*100 for a in history["test_acc"]],  label="Test acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over epochs")
plt.legend()

plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------------------------
CT_X_bf = torch.from_numpy(X_train_std).float()  # (B,T,F)
CT_y = torch.from_numpy(y_train_np).long()

CT_ds = TensorDataset(CT_X_bf, CT_y)
CT_loader = DataLoader(CT_ds, batch_size=32, shuffle=True)

CT_optimizer = torch.optim.Adam(list(CT_rnn_alt.parameters()) + list(CT_fc_alt.parameters()), lr=1e-3)
CT_criterion = nn.CrossEntropyLoss()

CT_train_losses = []
CT_NUM_EPOCHS = 10

for epoch in range(CT_NUM_EPOCHS):
    CT_rnn_alt.train()
    epoch_loss = 0.0
    total = 0
    for xb, yb in CT_loader:
        xb_tbf = xb.permute(1, 0, 2)          # (T,B,F)
        CT_optimizer.zero_grad()
        out_seq, h_n = CT_rnn_alt(xb_tbf)
        logits = CT_fc_alt(h_n.squeeze(0))
        loss = CT_criterion(logits, yb)
        loss.backward()
        CT_optimizer.step()
        epoch_loss += loss.item() * yb.size(0)
        total += yb.size(0)
    CT_train_losses.append(epoch_loss / total)

plt.figure(figsize=(6,3))
plt.plot(range(1, CT_NUM_EPOCHS+1), CT_train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CT_rnn_alt Training Loss (10 epochs)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# ------------------------------------------------------------------------------------------------
# Collect predictions on the test set

import torch
import numpy as np

model.eval()
all_logits = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)               # (batch, 6)
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())

all_logits  = torch.cat(all_logits, dim=0)     # (N_test, 6)
all_targets = torch.cat(all_targets, dim=0)    # (N_test,)
all_preds   = all_logits.argmax(dim=1)         # (N_test,)
# ------------------------------------------------------------------------------------------------
# Overall accuracy
test_acc = (all_preds == all_targets).float().mean().item()
print(f"Test accuracy: {test_acc*100:.2f}%")

# Show a few predictions
for i in range(5):
    t = int(all_targets[i])
    p = int(all_preds[i])
    print(f"Example {i}: true={ACTIVITY_MAP[t+1]:<20s} pred={ACTIVITY_MAP[p+1]:<20s}")
# ------------------------------------------------------------------------------------------------
# Confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

labels_idx = list(range(NUM_CLASSES))
labels_txt = [ACTIVITY_MAP[i+1] for i in labels_idx]

cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=labels_idx)
print("Confusion Matrix (raw counts):")
print(cm)

print("\nClassification Report:")
print(classification_report(all_targets.numpy(), all_preds.numpy(), target_names=labels_txt, digits=3))
# ------------------------------------------------------------------------------------------------
# Plot confusion matrix heatmap

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, interpolation='nearest', aspect='auto')
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=labels_txt,
    yticklabels=labels_txt,
    xlabel="Predicted label",
    ylabel="True label",
    title="Confusion Matrix — Vanilla RNN on HAR (Test)"
)

# Rotate x tick labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate cells with counts
thresh = cm.max() / 2.0 if cm.size > 0 else 0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------------------------
# Identify the most confused pairs (excluding the diagonal)

cm_np = cm.astype(np.int64)
cm_offdiag = cm_np.copy()
np.fill_diagonal(cm_offdiag, 0)

# Find top-k confusions
k = 5
flat_idx = np.argsort(cm_offdiag.ravel())[::-1][:k]
rows, cols = np.unravel_index(flat_idx, cm_offdiag.shape)

print("Top confusions (true -> predicted : count):")
for r, c in zip(rows, cols):
    if cm_offdiag[r, c] > 0:
        print(f"{labels_txt[r]:<20s} -> {labels_txt[c]:<20s} : {cm_offdiag[r, c]}")