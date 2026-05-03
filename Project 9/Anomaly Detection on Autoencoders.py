import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from math import inf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
# -------------------------------------------------------------------------------------
#Import and initialize the FC_Autoencoder 
# Define the same architecture used in Notebook 1
class FC_Autoencoder(nn.Module):
    def __init__(self):
        super(FC_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(48 * 48, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 48 * 48),
            nn.Sigmoid()   # constrain outputs to [0, 1]
        )

    def forward(self, x):
        x = x.view(-1, 48 * 48)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 48, 48)
        return out

# Initialize model
model = FC_Autoencoder()

# Optional: Load trained weights if available
# model.load_state_dict(torch.load("fc_autoencoder.pth", map_location='cpu'))

# Confirm input–output shape consistency
dummy_input = torch.randn(1, 1, 48, 48)
dummy_output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")
# -------------------------------------------------------------------------------------
# Training for Anomaly Detection
# Load the downsampled dataset if not already in memory
try:
    X, y
except NameError:
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]

known_labels = {0, 3, 4, 5, 6}    # angry, happy, sad, surprise, neutral
unknown_labels = {1, 2}           # disgust, fear

# Filter known subset
mask_known = np.isin(y, list(known_labels))
X_known, y_known = X[mask_known], y[mask_known]

# Optional: balance known classes by undersampling to the minimum class count
counts = Counter(y_known.tolist())
min_count = min(counts.values())
idx_balanced = []

# collect min_count indices per class
for lbl in sorted(known_labels):
    idx_lbl = np.where(y_known == lbl)[0]
    choose = np.random.choice(idx_lbl, size=min_count, replace=False)
    idx_balanced.append(choose)

idx_balanced = np.concatenate(idx_balanced, axis=0)
np.random.shuffle(idx_balanced)

X_known_bal = X_known[idx_balanced]
y_known_bal = y_known[idx_balanced]

print("Balanced known class counts:",
      Counter(y_known_bal.tolist()))
print("X_known_bal shape:", X_known_bal.shape)
# ---------------------------------------------
#Now we split the known subset into train/test, stratified by emotion to preserve class balance.
# ---------------------------------------------
Xk_train, Xk_test, yk_train, yk_test = train_test_split(
    X_known_bal, y_known_bal, test_size=0.15, random_state=42, stratify=y_known_bal
)

# Convert to torch tensors with channel dimension (N, 1, 48, 48)
Xk_train_t = torch.tensor(Xk_train, dtype=torch.float32).unsqueeze(1)
Xk_test_t  = torch.tensor(Xk_test,  dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(Xk_train_t)   # labels not used for AE training
test_ds  = TensorDataset(Xk_test_t)

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(test_loader)
# ---------------------------------------------
# Define Loss and Optimizer
# ---------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # reuse FC_Autoencoder from Section 2

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ---------------------------------------------
# Train
# ---------------------------------------------
EPOCHS = 100
best_test = inf
best_state = None

for epoch in range(1, EPOCHS + 1):
    # ----- Train -----
    model.train()
    train_sum = 0.0
    for (xb,) in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        xhat = model(xb)
        loss = criterion(xhat, xb)
        loss.backward()
        optimizer.step()
        train_sum += loss.item() * xb.size(0)
    train_avg = train_sum / len(train_ds)

    # ----- Eval on held-out known data -----
    model.eval()
    test_sum = 0.0
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            xhat = model(xb)
            loss = criterion(xhat, xb)
            test_sum += loss.item() * xb.size(0)
    test_avg = test_sum / len(test_ds)

    print(f"Epoch {epoch:02d} | train MSE: {train_avg:.5f} | test MSE: {test_avg:.5f}")

    if test_avg < best_test:
        best_test = test_avg
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Restore best weights (recommended before anomaly evaluation)
if best_state is not None:
    model.load_state_dict(best_state)
# -------------------------------------------------------------------------------------
# Compute Reconstruction Errors (Known vs Unknown)
# Ensure we have: model (trained on known classes), device, and the full X, y arrays
try:
    X, y
except NameError:
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]

known_labels = {0, 3, 4, 5, 6}   # angry, happy, sad, surprise, neutral
unknown_labels = {1, 2}          # disgust, fear

# Build known/unknown tensors
idx_known_eval = np.isin(y, list(known_labels))
idx_unknown_eval = np.isin(y, list(unknown_labels))

X_known_eval = torch.tensor(X[idx_known_eval], dtype=torch.float32).unsqueeze(1)
X_unknown_eval = torch.tensor(X[idx_unknown_eval], dtype=torch.float32).unsqueeze(1)

def reconstruction_errors(batch_tensor):
    """
    Compute per-image MSE reconstruction error in [0, 1] space.
    Returns a 1D numpy array of length N.
    """
    errs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(batch_tensor), 256):
            xb = batch_tensor[i:i+256].to(device)
            xhat = model(xb)
            # Per-image MSE over HxW (not batch-averaged)
            e = (xhat - xb).pow(2).mean(dim=(1,2,3))  # (N,)
            errs.append(e.cpu().numpy())
    return np.concatenate(errs, axis=0)

errs_known = reconstruction_errors(X_known_eval)
errs_unknown = reconstruction_errors(X_unknown_eval)

print("Known eval count  :", len(errs_known))
print("Unknown eval count:", len(errs_unknown))
print("Known error stats :", np.mean(errs_known), np.median(errs_known))
print("Unknown error stats:", np.mean(errs_unknown), np.median(errs_unknown))
# ---------------------------------------------
# Histograms of Reconstruction Errors
# ---------------------------------------------
plt.figure(figsize=(7, 4))
bins = 40
plt.hist(errs_known, bins=bins, alpha=0.6, label="Known (train classes)", edgecolor="black")
plt.hist(errs_unknown, bins=bins, alpha=0.6, label="Unknown (anomalies)", edgecolor="black")
plt.title("Reconstruction Error Distributions")
plt.xlabel("Per-image MSE")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()
# ---------------------------------------------
# Thresholding and AUC
# ---------------------------------------------
# Simple threshold: 95th percentile of known errors
thr = np.percentile(errs_known, 95)
print("95th percentile threshold (known errors):", float(thr))

# Build labels and scores for ROC AUC
y_true = np.concatenate([np.zeros_like(errs_known), np.ones_like(errs_unknown)])  # 0=known, 1=unknown
y_score = np.concatenate([errs_known, errs_unknown])

auc = roc_auc_score(y_true, y_score)
print("ROC AUC (known vs unknown):", float(auc))

# Optional: ROC curve points
fpr, tpr, _ = roc_curve(y_true, y_score)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Anomaly Detection via Reconstruction Error")
plt.legend()
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------
# Side-by-Side Reconstructions (Known vs Unknown)
def _to_nchw_float32(x):
    """
    Accepts numpy or torch arrays in shape:
      - (N, 48, 48)  or
      - (N, 1, 48, 48)
    Returns torch.FloatTensor on model.device with shape (N, 1, 48, 48).
    """
    device = next(model.parameters()).device
    # If torch tensor, move to CPU for shape checks
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1:] == (48, 48):
        x = x[:, None, :, :]  # add channel dim -> (N,1,48,48)
    elif x.ndim == 4 and x.shape[1:] == (1, 48, 48):
        pass  # already (N,1,48,48)
    else:
        raise ValueError(f"Expected (N,48,48) or (N,1,48,48), got {x.shape}")

    xt = torch.tensor(x, dtype=torch.float32, device=device)
    return xt

def recon_pairs(XA, title_left, XB, title_right, n=5):
    """
    Robust side-by-side original vs reconstruction for two groups.
    Accepts XA/XB as numpy arrays or torch tensors with shape (N,48,48) or (N,1,48,48).
    """
    n = min(n, len(XA), len(XB))
    model.eval()

    A = _to_nchw_float32(XA[:n])
    B = _to_nchw_float32(XB[:n])

    with torch.no_grad():
        Ah = model(A).cpu().numpy()
        Bh = model(B).cpu().numpy()
        A  = A.cpu().numpy()
        B  = B.cpu().numpy()

    fig, axes = plt.subplots(n, 4, figsize=(8, 2*n))
    for i in range(n):
        axes[i, 0].imshow(A[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(f"{title_left} (orig)", fontsize=9); axes[i, 0].axis("off")

        axes[i, 1].imshow(Ah[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title(f"{title_left} (recon)", fontsize=9); axes[i, 1].axis("off")

        axes[i, 2].imshow(B[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title(f"{title_right} (orig)", fontsize=9); axes[i, 2].axis("off")

        axes[i, 3].imshow(Bh[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title(f"{title_right} (recon)", fontsize=9); axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

# If we already have torch tensors from earlier:
# X_known_eval and X_unknown_eval are (N,1,48,48) torch tensors
recon_pairs(X_known_eval, "Known", X_unknown_eval, "Unknown", n=5)

# If we have numpy arrays (e.g., from np.load):
# recon_pairs(X_known_np, "Known", X_unknown_np, "Unknown", n=5)
# -------------------------------------------------------------------------------------
# Error by Class (Bar Plot or Box Plot)
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def per_class_errors(X_all, y_all, max_per_class=400):
    errs_by_cls = {}
    for lbl in range(7):
        idx = np.where(y_all == lbl)[0]
        if len(idx) == 0:
            continue
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, size=max_per_class, replace=False)
        errs = reconstruction_errors(torch.tensor(X_all[idx], dtype=torch.float32).unsqueeze(1))
        errs_by_cls[lbl] = errs
    return errs_by_cls

errs_cls = per_class_errors(X, y, max_per_class=400)

# Bar plot of class-wise means
means = [errs_cls[k].mean() if k in errs_cls else np.nan for k in range(7)]
plt.figure(figsize=(7,4))
plt.bar(labels, means, edgecolor="black")
plt.xticks(rotation=30)
plt.ylabel("Mean reconstruction MSE")
plt.title("Reconstruction Error by Emotion Class")
plt.tight_layout()
plt.show()

# Optional: box plot to show spread
data_for_box = [errs_cls.get(k, np.array([])) for k in range(7)]
plt.figure(figsize=(8,4))
plt.boxplot([d for d in data_for_box if len(d)>0], labels=[l for l,d in zip(labels, data_for_box) if len(d)>0], showfliers=False)
plt.xticks(rotation=30)
plt.ylabel("Per-image MSE")
plt.title("Per-class Error Distribution (Known and Unknown Mixed)")
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------
# Building a Denoising Autoencoder
# If train/test splits are not in memory, recreate them from Section 4 defaults
try:
    Xk_train, Xk_test
except NameError:
    # Fallback: use the full dataset split if needed
    data = np.load("fer2013_small.npz")
    X_all, y_all = data["X"], data["y"]
    from sklearn.model_selection import train_test_split
    Xk_train, Xk_test, yk_train, yk_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )

# Add Gaussian noise to training inputs
noise_factor = 0.3
rng = np.random.default_rng(42)

X_noisy_train = Xk_train + noise_factor * rng.standard_normal(Xk_train.shape)
X_noisy_train = np.clip(X_noisy_train, 0.0, 1.0)

# Validation set can also be evaluated with noisy inputs
X_noisy_test = Xk_test + noise_factor * rng.standard_normal(Xk_test.shape)
X_noisy_test = np.clip(X_noisy_test, 0.0, 1.0)

# Tensors: inputs are noisy, targets are clean
X_tr_in  = torch.tensor(X_noisy_train, dtype=torch.float32).unsqueeze(1)
X_tr_tgt = torch.tensor(Xk_train,      dtype=torch.float32).unsqueeze(1)

X_te_in  = torch.tensor(X_noisy_test, dtype=torch.float32).unsqueeze(1)
X_te_tgt = torch.tensor(Xk_test,      dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_tr_in, X_tr_tgt)
test_ds  = TensorDataset(X_te_in, X_te_tgt)

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(test_loader)
# ------------------------------------------
# Reuse the Same Autoencoder Architecture
# ------------------------------------------
class FC_Autoencoder(nn.Module):
    def __init__(self, input_dim=48*48, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, 1, 48, 48)
        n = x.size(0)
        x_flat = x.view(n, -1)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z).view(n, 1, 48, 48)
        return x_hat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dae = FC_Autoencoder().to(device)
# ------------------------------------------
# Train With Noisy Inputs and Clean Targets
# ------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(dae.parameters(), lr=1e-3)

EPOCHS = 10
best_val = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    # Train
    dae.train()
    run_tr = 0.0
    for xb_in, xb_tgt in train_loader:
        xb_in  = xb_in.to(device)
        xb_tgt = xb_tgt.to(device)

        optimizer.zero_grad()
        xb_hat = dae(xb_in)
        loss = criterion(xb_hat, xb_tgt)  # target is CLEAN
        loss.backward()
        optimizer.step()

        run_tr += loss.item() * xb_in.size(0)

    tr_loss = run_tr / len(train_ds)

    # Validate
    dae.eval()
    run_va = 0.0
    with torch.no_grad():
        for xb_in, xb_tgt in test_loader:
            xb_in  = xb_in.to(device)
            xb_tgt = xb_tgt.to(device)
            xb_hat = dae(xb_in)
            loss = criterion(xb_hat, xb_tgt)
            run_va += loss.item() * xb_in.size(0)

    va_loss = run_va / len(test_ds)

    print(f"Epoch {epoch:02d} | train MSE: {tr_loss:.5f} | val MSE: {va_loss:.5f}")

    if va_loss < best_val:
        best_val = va_loss
        best_state = {k: v.cpu().clone() for k, v in dae.state_dict().items()}

# Restore best validation weights
if best_state is not None:
    dae.load_state_dict(best_state)
# -------------------------------------------------------------------------------------
# Prepare a Small Batch for Visualization
dae.eval()
with torch.no_grad():
    xb_noisy, xb_clean = next(iter(test_loader))
    xb_noisy = xb_noisy.to(device)
    xb_clean = xb_clean.to(device)
    xb_recon = dae(xb_noisy)

# Move tensors back to CPU for visualization
noisy_np = xb_noisy[:5].cpu().squeeze(1).numpy()
recon_np = xb_recon[:5].cpu().squeeze(1).numpy()
clean_np = xb_clean[:5].cpu().squeeze(1).numpy()

# Display Noisy → Reconstructed → Original
n_examples = 5
fig, axes = plt.subplots(n_examples, 3, figsize=(6, 10))
cols = ["Noisy Input", "Reconstructed", "Original"]

for i in range(n_examples):
    for j, img in enumerate([noisy_np[i], recon_np[i], clean_np[i]]):
        ax = axes[i, j]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title(cols[j], fontsize=11)
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------

