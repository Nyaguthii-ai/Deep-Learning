import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from math import inf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------------------
# Sampling with Reparameterization
# Suppose the encoder produced these:
mu = torch.tensor([0.0, 1.0, -1.0])
log_var = torch.tensor([-1.0, 0.0, 1.0])  # log(σ²)
sigma = torch.exp(0.5 * log_var)

# Sample epsilon ~ N(0, I)
epsilon = torch.randn_like(sigma)

# Reparameterization
z = mu + sigma * epsilon

print("mu      :", mu)
print("sigma   :", sigma)
print("epsilon :", epsilon)
print("z (sampled latent vector):", z)
# Output: 
# mu      : tensor([ 0.,  1., -1.])
# sigma   : tensor([0.6065, 1.0000, 1.6487])
# epsilon : tensor([ 0.3997,  1.4304, -1.1592])
# z (sampled latent vector): tensor([ 0.2424,  2.4304, -2.9112])
# ---------------------------------------------------------------------------------------

## VAE Architecture Implementation in PyTorch
class VAE(nn.Module):
    def __init__(self, input_dim=48*48, hidden_dim1=512, hidden_dim2=128, latent_dim=32):
        super(VAE, self).__init__()
        
        # ----- Encoder -----
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)       # Mean vector μ
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)   # Log-variance logσ²
        
        # ----- Decoder -----
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc5 = nn.Linear(hidden_dim1, input_dim)
        
    def encode(self, x):
        """Encode input into mean and log-variance."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Apply the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z):
        """Decode latent vector back to image."""
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        x_recon = torch.sigmoid(self.fc5(h))  # constrain to [0,1]
        return x_recon
    
    def forward(self, x):
        """Full forward pass: encode → reparam → decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
# -----------------------------------------   
    #Quick Sanity Check
# -----------------------------------------
model = VAE(latent_dim = 64) # default is 32
x_dummy = torch.randn(4, 48*48)  # batch of 4 random faces
x_recon, mu, logvar = model(x_dummy)

print("Input shape :", x_dummy.shape)
print("Reconstructed shape:", x_recon.shape)
print("Latent mean shape  :", mu.shape)
print("Latent logvar shape:", logvar.shape)
# ---------------------------------------------------------------------------------------
## VAE Loss Function
def vae_loss(x_recon, x, mu, logvar, recon_type="mse", beta=1.0, reduction="mean"):
    """
    x_recon: reconstructed images, shape (N, 2304) or (N, 1, 48, 48) after flattening
    x:       original images, same shape as x_recon
    mu:      latent mean, shape (N, latent_dim)
    logvar:  latent log-variance, shape (N, latent_dim)
    recon_type: "mse" or "bce"
    beta:    weighting factor for KL term (beta-VAE)
    reduction: "mean" or "sum" over batch

    returns: total_loss, recon_loss, kl_loss
    """

    # Ensure flat (N, 2304) for reconstruction computation
    if x_recon.dim() > 2:
        x_recon = x_recon.view(x_recon.size(0), -1)
    if x.dim() > 2:
        x = x.view(x.size(0), -1)

    # Reconstruction term
    if recon_type.lower() == "mse":
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)
    elif recon_type.lower() == "bce":
        # Avoid log(0); clamp is handled internally by F.binary_cross_entropy
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction=reduction)
    else:
        raise ValueError("recon_type must be 'mse' or 'bce'.")

    # KL divergence term:
    # L_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per sample
    # We average over batch by default (reduction == "mean")
    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    if reduction == "sum":
        kl_loss = -0.5 * torch.sum(kl_element)
    elif reduction == "mean":
        kl_loss = -0.5 * torch.mean(torch.sum(kl_element, dim=1))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
# ---------------------------------------------------------------------------------------
# Training the VAE
# Load the downsampled dataset prepared earlier
data = np.load("fer2013_small.npz")
X, y = data["X"], data["y"]          # X: (N, 48, 48) in [0,1], y: (N,)

# Train/validation split (stratified to preserve class proportions)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Convert to torch tensors and flatten to vectors of length 2304 for the VAE
def to_tensor_flat(arr):
    t = torch.tensor(arr, dtype=torch.float32)     # (N, 48, 48)
    t = t.view(t.size(0), -1)                      # (N, 2304)
    return t

X_train_t = to_tensor_flat(X_train)
X_val_t   = to_tensor_flat(X_val)

train_ds = TensorDataset(X_train_t)
val_ds   = TensorDataset(X_val_t)

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(val_loader)
# --------------------------------------
#Initialize the VAE and Optimizer
# --------------------------------------
# If the VAE class and vae_loss() are not yet in this runtime, rerun the cells from Sections 4–5 first.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example configuration: latent_dim=32 for a compact, sampleable latent space
model = VAE(input_dim=48*48, hidden_dim1=512, hidden_dim2=128, latent_dim=32).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ---------------------------------------------------------------------------------------
# Training Loop, Tracking Recon and KL
EPOCHS = 15
best_val = inf
best_state = None

def run_epoch(model, loader, train=True, recon_type="mse", beta=1.0):
    if train:
        model.train()
    else:
        model.eval()
    total_sum, recon_sum, kl_sum, count = 0.0, 0.0, 0.0, 0

    for (xb,) in loader:
        xb = xb.to(device)                # xb: (N, 2304)
        if train:
            optimizer.zero_grad()

        x_recon, mu, logvar = model(xb)   # x_recon: (N, 2304)
        total, recon, kl = vae_loss(x_recon, xb, mu, logvar,
                                    recon_type=recon_type, beta=beta, reduction="mean")

        if train:
            total.backward()
            optimizer.step()

        bs = xb.size(0)
        total_sum += total.item() * bs
        recon_sum += recon.item() * bs
        kl_sum    += kl.item() * bs
        count     += bs

    return total_sum / count, recon_sum / count, kl_sum / count
# ---------------------------------------------------------------------------------------
history = {"train_total":[], "train_recon":[], "train_kl":[],
           "val_total":[],   "val_recon":[],   "val_kl":[]}

def kl_weight(epoch, E=EPOCHS, start=0.0, end=1.0):
    # linear warm-up; try sigmoid if you prefer
    return start + (end - start) * (epoch-1)/(E-1)

for epoch in range(1, EPOCHS + 1):
    beta = kl_weight(epoch, EPOCHS, start=0.0, end=1.0)  # or end=0.5
    tr_tot, tr_rec, tr_kl = run_epoch(model, train_loader, train=True,  recon_type="bce", beta=beta)
    va_tot, va_rec, va_kl = run_epoch(model, val_loader,   train=False, recon_type="bce", beta=beta)

    history["train_total"].append(tr_tot); history["train_recon"].append(tr_rec); history["train_kl"].append(tr_kl)
    history["val_total"].append(va_tot);   history["val_recon"].append(va_rec);   history["val_kl"].append(va_kl)

    print(f"Epoch {epoch:02d} | "
          f"train total: {tr_tot:.5f} (recon {tr_rec:.5f}, KL {tr_kl:.5f}) | "
          f"val total: {va_tot:.5f} (recon {va_rec:.5f}, KL {va_kl:.5f})")

    if va_tot < best_val:
        best_val = va_tot
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Restore best validation weights
if best_state is not None:
    model.load_state_dict(best_state)
# --------------------------------------
# Save trained VAE
# --------------------------------------
try:
    torch.save(model.state_dict(), "vae_model.pth")
    print("✅ VAE model saved as vae_model.pth")
except Exception as e:
    print("⚠️ Could not save VAE:", e)

print(model)
# ---------------------------------------------------------------------------------------
# VAE: Original vs. Reconstructed Pairs
# --- Reuse validation data from Section 6; add a light fallback if needed ---
try:
    X_val_t
except NameError:
    import numpy as np
    from sklearn.model_selection import train_test_split
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).view(X_val.shape[0], -1)

device = next(model.parameters()).device  # VAE from Section 4–6
model.eval()

# Select a small batch
idx = torch.randperm(X_val_t.size(0))[:8]
xb = X_val_t[idx].to(device)                      # (N, 2304)
with torch.no_grad():
    x_recon, mu, logvar = model(xb)               # (N, 2304)
xb_img     = xb.view(-1, 48, 48).cpu().numpy()
xrecon_img = x_recon.view(-1, 48, 48).cpu().numpy()

# Plot original vs reconstructed (VAE)
n = xb_img.shape[0]
rows = n
fig, axes = plt.subplots(rows, 2, figsize=(5, 2*rows))
for i in range(rows):
    axes[i, 0].imshow(xb_img[i], cmap="gray", vmin=0.0, vmax=1.0)
    axes[i, 0].set_title("Original", fontsize=9); axes[i, 0].axis("off")

    axes[i, 1].imshow(xrecon_img[i], cmap="gray", vmin=0.0, vmax=1.0)

    axes[i, 1].set_title("VAE Reconstruction", fontsize=9); axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------------------------
# Side-by-Side Comparison: AE vs. VAE
# For reproducibility
torch.manual_seed(42)

# Flattened input size for 48×48 grayscale images
INPUT_DIM = 48 * 48         # 2304
LATENT_DIM = 128

class FC_Autoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder: 2304 -> 512 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )
        # Decoder: 128 -> 512 -> 2304
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # constrain output to [0, 1]
        )

    def forward(self, x):
        """
        x: (N, 1, 48, 48) or (N, 48, 48)
        returns: reconstruction with same shape as input
        """
        # Ensure shape is (N, 48, 48)
        if x.dim() == 4:        # (N, 1, 48, 48)
            x = x.squeeze(1)
        # Flatten to (N, 2304)
        n = x.size(0)
        x_flat = x.view(n, -1)
        # Encode -> Decode
        z = self.encoder(x_flat)
        x_hat_flat = self.decoder(z)
        # Reshape back to (N, 48, 48)
        x_hat = x_hat_flat.view(n, 48, 48)
        # Add channel dim back to be consistent with PyTorch image conventions
        x_hat = x_hat.unsqueeze(1)  # (N, 1, 48, 48)
        return x_hat
    
ae_model = FC_Autoencoder().to(device)
ae_model.load_state_dict(torch.load("ae_model.pth", map_location=device))
ae_model.eval()
# --------------------------------------------
# Optional AE comparison.
# Requirements:
# - ae_model: a trained AE on the same data distribution (flattened 2304 inputs -> 2304 outputs with Sigmoid)
# - If not available, this cell will skip gracefully.
# --------------------------------------------
try:
    _ = ae_model
    ae_model = ae_model.to(device).eval()

    with torch.no_grad():
        ae_recon = ae_model(xb)  # (N, 2304)

    ae_img = ae_recon.view(-1, 48, 48).detach().cpu().numpy()

    # Grid: Original | AE Recon | VAE Recon
    n = xb_img.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(7, 2*n))
    for i in range(n):
        axes[i, 0].imshow(xb_img[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title("Original", fontsize=9); axes[i, 0].axis("off")

        axes[i, 1].imshow(ae_img[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("AE Recon", fontsize=9); axes[i, 1].axis("off")

        axes[i, 2].imshow(xrecon_img[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("VAE Recon", fontsize=9); axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

except NameError:
    print("AE model `ae_model` not found. Skipping AE vs VAE comparison. "
          "Provide a trained AE (flattened 2304 → 2304 with Sigmoid) as `ae_model` to enable this plot.")

# ---------------------------------------------------------------------------------------
# Generating Synthetic Faces
# --------------------------------------- 
# Sample from {N}(0, I) and Decode
# ---------------------------------------
# We assume `model` is our trained VAE from Sections 4–6
device = next(model.parameters()).device
model.eval()
# set seed
torch.manual_seed(42)

# How many samples to generate (must be a square number for a tidy grid)
NUM = 16          # try 16 or 25
LATENT_DIM = model.fc_mu.out_features  # infer latent size from the encoder head

# Sample z ~ N(0, I)
with torch.no_grad():
    z = torch.randn(NUM, LATENT_DIM, device=device)
    x_gen = model.decode(z)                     # (NUM, 2304) if our VAE outputs flat vectors
    x_gen = x_gen.view(NUM, 48, 48).cpu().numpy()
# ---------------------------------------
# Visualize as a Grid of Synthetic Faces
# ---------------------------------------
# Display synthetic faces in an n x n grid
n = int(np.sqrt(NUM))
fig, axes = plt.subplots(n, n, figsize=(6, 6))
k = 0
for i in range(n):
    for j in range(n):
        axes[i, j].imshow(x_gen[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, j].axis("off")
        k += 1
plt.suptitle("Random Samples from VAE (z ~ N(0, I))", fontsize=12)
plt.tight_layout()
plt.show()
# ---------------------------------------
# Temperature (Variance) Scaling for Diversity
# ---------------------------------------
# 'temperature' > 1.0 increases diversity; < 1.0 makes samples more conservative
temperature = 0.7   # try 0.7, 1.0, 1.5
with torch.no_grad():
    z = torch.randn(NUM, LATENT_DIM, device=device) * temperature
    x_gen_t = model.decode(z).view(NUM, 48, 48).cpu().numpy()

fig, axes = plt.subplots(n, n, figsize=(6, 6))
k = 0
for i in range(n):
    for j in range(n):
        axes[i, j].imshow(x_gen_t[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, j].axis("off")
        k += 1
plt.suptitle(f"VAE Samples (temperature = {temperature})", fontsize=12)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------------------------
# Exploring the Latent Space
# ---------------------------------------
# Encode the Validation Set to Get μ Vectors
# ---------------------------------------
try:
    X_val, y_val
except NameError:
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

# -- Prepare data loader (flatten images for VAE)
X_val_t = torch.tensor(X_val, dtype=torch.float32).view(X_val.shape[0], -1)
val_ds = TensorDataset(X_val_t, torch.tensor(y_val))
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

device = next(model.parameters()).device
model.eval()

# -- Collect μ (latent means) for all validation images
mus = []
labs = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        mu, logvar = model.encode(xb)
        mus.append(mu.cpu().numpy())
        labs.append(yb.numpy())

Z_mu = np.concatenate(mus, axis=0)         # shape (N, latent_dim)
y_val_np = np.concatenate(labs, axis=0)

print("Encoded μ shape:", Z_mu.shape, "labels:", y_val_np.shape)
# ---------------------------------------
# 2D Projection (PCA or t-SNE) 
# PCA is fast and gives a linear projection.
# t-SNE can reveal non-linear structure but is slower; we often apply PCA to 30 dims first, then t-SNE.
# ---------------------------------------
# -- Emotion names
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -- Pairs to visualize
pairs = [
    (3, 5, "Happy vs Surprise"),
    (3, 4, "Happy vs Sad"),
]

for a, b, title in pairs:
    # Filter only the two chosen emotions
    mask = np.isin(y_val_np, [a, b])
    Z_pair = Z_mu[mask]
    y_pair = y_val_np[mask]

    # Re-label to binary (0,1) for consistent color
    y_binary = (y_pair == a).astype(int)  # 1 = Happy, 0 = other

    # Standardize is optional (useful for t-SNE stability)
    # --- PCA (2D) ---
    Z_pca = PCA(n_components=2, random_state=42).fit_transform(Z_pair)

    plt.figure(figsize=(6,5))
    plt.scatter(Z_pca[:,0], Z_pca[:,1], c=y_binary, cmap="bwr", s=15, alpha=0.85)

    # Custom legend
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="blue", markersize=8,
                   label=emotions[b]),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="red", markersize=8,
                   label=emotions[a]),
    ]
    plt.legend(handles=handles, title="Emotion", loc="best")
    plt.title(f"{title} (PCA Projection of μ)")
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()

    # --- t-SNE (optional) ---
    # First reduce to (at most) 30D for speed
    Z_30 = PCA(n_components=min(30, Z_pair.shape[1]), random_state=42).fit_transform(Z_pair)
    Z_tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                  init="pca", random_state=42).fit_transform(Z_30)

    plt.figure(figsize=(6,5))
    plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=y_binary, cmap="bwr", s=15, alpha=0.85)

    # Same legend
    plt.legend(handles=handles, title="Emotion", loc="best")
    plt.title(f"{title} (t-SNE Projection of μ)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------------------------------------
