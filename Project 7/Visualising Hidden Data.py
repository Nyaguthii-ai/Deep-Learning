# Loading the saved data in csv, npy and npz in Week 7/Vanilla RNN for Activity Recognition.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------------------------------
# CSV loader (default path)
def load_csv_dataset(train_path="har_train_standardized.csv",
                     test_path="har_test_standardized.csv",
                     timesteps=128, channels=9):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    def df_to_arrays(df):
        y = df["label"].values.astype(np.int64)              # (N,)
        subj = df["subject"].values.astype(np.int64)         # (N,)
        X_flat = df.drop(columns=["label","subject"]).values # (N, 1152)
        X = X_flat.reshape(len(df), timesteps, channels)     # (N, 128, 9)
        return X.astype(np.float32), y, subj

    X_train, y_train, subj_train = df_to_arrays(train_df)
    X_test,  y_test,  subj_test  = df_to_arrays(test_df)

    return X_train, y_train, X_test, y_test, subj_train, subj_test

# Load from CSV by default
X_train, y_train, X_test, y_test, subj_train, subj_test = load_csv_dataset()

# Wrap as PyTorch tensors (batch-first: (B, 128, 9))
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# DataLoaders: for NB03, small batches are fine; we’ll still provide standard loaders
BATCH_SIZE = 96
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t), batch_size=BATCH_SIZE, shuffle=False)

print("Loaded from CSV →",
      "X_train", X_train_t.shape, "| y_train", y_train_t.shape, 
      "| X_test", X_test_t.shape,  "| y_test",  y_test_t.shape)
# ------------------------------------------------------------------------------------------------------
#If we prefer speed and don’t need to read the CSVs, we can use the arrays saved in **`.npy`** directly.
# NPY loader (optional path)

def load_npy_dataset(
    x_train_path="X_train_std.npy",
    y_train_path="y_train.npy",
    x_test_path="X_test_std.npy",
    y_test_path="y_test.npy"
):
    X_train = np.load(x_train_path)  # (N, 128, 9)
    y_train = np.load(y_train_path)
    X_test  = np.load(x_test_path)
    y_test  = np.load(y_test_path)
    return X_train, y_train, X_test, y_test

# Example usage (commented out by default):
X_train, y_train, X_test, y_test = load_npy_dataset()
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t), batch_size=64, shuffle=False)
print("Loaded from NPY →", X_train_t.shape, y_train_t.shape, X_test_t.shape, y_test_t.shape)
# ------------------------------------------------------------------------------------------------------
# Return all hidden states
import torch.nn as nn

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
        returns:
          logits: (batch, num_classes)
          out_seq: (batch, seq_len, hidden_size)  # all hidden states
          h_n: (num_layers, batch, hidden_size)   # final hidden state(s)
        """
        out_seq, h_n = self.rnn(x)
        last_hidden = out_seq[:, -1, :]    # (batch, hidden)
        logits = self.fc(last_hidden)      # (batch, classes)
        return logits, out_seq, h_n
# ------------------------------------------------------------------------------------------------------
#Load trained weights (rnn_har.pth)
# Instantiate model and load checkpoint
model = RNNClassifier(input_size=9, hidden_size=32, num_classes=6, num_layers=1, nonlinearity="tanh")

ckpt_path = "rnn_har.pth"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()

print("Loaded trained weights from:", ckpt_path)
# ------------------------------------------------------------------------------------------------------
# Quick dry run to inspect shapes
import torch.nn.functional as F
xb, yb = next(iter(test_loader))   # or train_loader
logits, out_seq, h_n = model(xb)

print("xb shape      :", tuple(xb.shape))       # (B, 128, 9)
print("out_seq shape :", tuple(out_seq.shape))  # (B, 128, H)
print("h_n shape     :", tuple(h_n.shape))      # (1, B, H)
print("logits shape  :", tuple(logits.shape))   # (B, 6)

probs = F.softmax(logits, dim=1)
print("First row prob sum ~1.0:", float(probs[0].sum()))
# ------------------------------------------------------------------------------------------------------
# Activity index → readable name
ACTIVITY_MAP = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

def first_index_of_label(y_batch: torch.Tensor, label_idx: int):
    """Return first index in y_batch == label_idx, or None if not found."""
    y_np = y_batch.cpu().numpy()
    hits = np.where(y_np == label_idx)[0]
    return int(hits[0]) if hits.size > 0 else None

def pick_topk_dims_by_variance(hidden_seq: np.ndarray, k: int = 3):
    """Pick top-k hidden dimensions with highest variance across time."""
    var = hidden_seq.var(axis=0)  # (H,)
    return np.argsort(var)[::-1][:k]

def plot_hidden_dims_over_time(hidden_seq: np.ndarray, dims, title: str):
    """Plot selected hidden dims across timesteps."""
    T = hidden_seq.shape[0]
    t = np.arange(1, T+1)
    plt.figure(figsize=(10,4))
    for d in dims:
        plt.plot(t, hidden_seq[:, d], label=f"h[t, dim={d}]")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Hidden value")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_hidden_norm_over_time(hidden_seq: np.ndarray, title: str):
    """Plot the L2 norm of hidden vector across timesteps."""
    norms = np.linalg.norm(hidden_seq, axis=1)
    t = np.arange(1, hidden_seq.shape[0]+1)
    plt.figure(figsize=(10,3))
    plt.plot(t, norms)
    plt.xlabel("Timestep (t)")
    plt.ylabel("||h_t||_2")
    plt.title(title)
    plt.show()
# ------------------------------------------------------------------------------------------------------
# Forward one batch (B=96) and visualize one sequence (SITTING)

# NOTE: make sure your DataLoader was created with batch_size=96 earlier.
xb, yb = next(iter(test_loader))   # xb: (96,128,9), yb: (96,)

model.eval()
with torch.no_grad():
    logits, out_seq, h_n = model(xb)  # out_seq: (96,128,32)

B, T, H = out_seq.shape
print(f"Batch size B={B}, seq_len T={T}, hidden_size H={H}")

# Pick a sample with label = SITTING (class=3); fallback = first sample
target_label = 3
idx = first_index_of_label(yb, target_label)
if idx is None:
    idx = 0
    target_label = int(yb[idx])

print(f"Visualizing sample index {idx}, true label = {ACTIVITY_MAP[int(target_label)]}")

# Hidden states for that sample → (128,32)
hidden_seq = out_seq[idx].cpu().numpy()

# Strategy 1: top-variance dims
topk_dims = pick_topk_dims_by_variance(hidden_seq, k=3)
plot_hidden_dims_over_time(
    hidden_seq, topk_dims,
    title=f"Hidden dimensions (top-variance) — {ACTIVITY_MAP[int(target_label)]}"
)

# Strategy 2: hidden norm
plot_hidden_norm_over_time(
    hidden_seq,
    title=f"Hidden norm over time — {ACTIVITY_MAP[int(target_label)]}"
)
# ------------------------------------------------------------------------------------------------------
# Visualize a contrasting dynamic class (WALKING = 0) from the same batch

target_label_walk = 0
idx_walk = first_index_of_label(yb, target_label_walk)
if idx_walk is not None:
    hidden_seq_walk = out_seq[idx_walk].cpu().numpy()
    topk_dims_walk = pick_topk_dims_by_variance(hidden_seq_walk, k=3)

    plot_hidden_dims_over_time(
        hidden_seq_walk, topk_dims_walk,
        title=f"Hidden dimensions (top-variance) — {ACTIVITY_MAP[target_label_walk]}"
    )

    plot_hidden_norm_over_time(
        hidden_seq_walk,
        title=f"Hidden norm over time — {ACTIVITY_MAP[target_label_walk]}"
    )
else:
    print("No WALKING example found in this batch; re-run the cell to sample another batch.")
# ------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Collect hidden states from one batch
xb, yb = next(iter(test_loader))  # xb: (B,128,9)
with torch.no_grad():
    logits, out_seq, h_n = model(xb)  # out_seq: (B,128,H)

B, T, H = out_seq.shape
hidden_flat = out_seq.reshape(B*T, H).cpu().numpy()  # (B*T, H)
labels_rep = np.repeat(yb.cpu().numpy(), T)         # repeat each label 128 times

print("Hidden states shape for PCA/TSNE:", hidden_flat.shape)
# ------------------------------------------------------------------------------------------------------
# PCA projection to 2D
pca = PCA(n_components=2)
hidden_pca = pca.fit_transform(hidden_flat)

plt.figure(figsize=(8,6))
for label in np.unique(labels_rep):
    idxs = labels_rep == label
    plt.scatter(hidden_pca[idxs,0], hidden_pca[idxs,1], 
                s=5, alpha=0.5, label=ACTIVITY_MAP[label])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of hidden states (one batch)")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------
from sklearn.manifold import TSNE

# Flatten across batch & time
hidden_all = out_seq.reshape(-1, H)        # shape (B*T, H)
labels_all = yb.repeat_interleave(T)       # shape (B*T,)

# Subsample for t-SNE (because it’s slow)
n_samples = 2000
idx = np.random.choice(hidden_all.shape[0], n_samples, replace=False)
hidden_subset = hidden_all[idx]
labels_subset = labels_all[idx]

# Apply t-SNE
tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
hidden_2d = tsne.fit_transform(hidden_subset)

# Plot with colors by label
plt.figure(figsize=(8,6))
for lbl in np.unique(labels_subset):
    mask = labels_subset == lbl
    plt.scatter(hidden_2d[mask, 0], hidden_2d[mask, 1], s=10, alpha=0.7, label=ACTIVITY_MAP[int(lbl)])
plt.title("t-SNE of hidden states (subset, multiple activities)")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------
# CT_Task — PCA (4 comps): plot PC3 vs PC4
xb, yb = next(iter(test_loader))
with torch.no_grad():
    _, out_seq_tmp, _ = model(xb)

B, T, H = out_seq_tmp.shape
hidden_flat = out_seq_tmp.reshape(B*T, H).cpu().numpy()
labels_rep = np.repeat(yb.cpu().numpy(), T)

pca4 = PCA(n_components=4)
CT_hidden_pca4 = pca4.fit_transform(hidden_flat)

plt.figure(figsize=(8,6))
for lbl in np.unique(labels_rep):
    m = labels_rep == lbl
    plt.scatter(CT_hidden_pca4[m, 2], CT_hidden_pca4[m, 3], s=5, alpha=0.5, label=ACTIVITY_MAP[lbl])
plt.xlabel("PC3")
plt.ylabel("PC4")
plt.title("PCA (4 comps) of hidden states — PC3 vs PC4")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------
# Plot Hidden Trajectory
def plot_trajectory(hidden_seq, label, color="blue"):
    """
    hidden_seq: (T, H) numpy array of hidden states
    label: activity string
    """
    # PCA to 2D
    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_seq)  # (T, 2)

    # Plot trajectory with arrows
    plt.plot(hidden_2d[:,0], hidden_2d[:,1], marker="o", markersize=3,
             linewidth=1, color=color, alpha=0.8, label=label)

    # Add arrows for direction (every ~20 timesteps)
    for i in range(0, len(hidden_2d), 20):
        plt.arrow(hidden_2d[i,0], hidden_2d[i,1],
                  hidden_2d[i+1,0]-hidden_2d[i,0],
                  hidden_2d[i+1,1]-hidden_2d[i,1],
                  head_width=0.1, head_length=0.1, fc=color, ec=color)

    return hidden_2d

# --- Pick one WALKING and one SITTING sequence ---
xb, yb = next(iter(test_loader))   # (B,128,9), (B,)
model.eval()
with torch.no_grad():
    logits, out_seq, h_n = model(xb)  # out_seq: (B,128,H)

# Find indices
idx_walk = first_index_of_label(yb, 0)  # WALKING
idx_sit  = first_index_of_label(yb, 3)  # SITTING

plt.figure(figsize=(7,6))

if idx_walk is not None:
    hidden_walk = out_seq[idx_walk].cpu().numpy()
    plot_trajectory(hidden_walk, "WALKING", color="green")

if idx_sit is not None:
    hidden_sit = out_seq[idx_sit].cpu().numpy()
    plot_trajectory(hidden_sit, "SITTING", color="red")

plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.title("Hidden State Trajectories in PCA space")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------
