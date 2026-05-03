import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------------------------------------------------------------
# Code: Load Trained Models
# --- Load FER2013-small dataset ---
data = np.load("fer2013_small.npz")
X, y = data["X"], data["y"]

print(f"Dataset loaded: {X.shape[0]} samples, shape {X.shape[1:]}")

# Convert to torch tensor for encoding later
X_t = torch.tensor(X, dtype=torch.float32).view(X.shape[0], -1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Define VAE model architecture ---
class VAE(nn.Module):
    def __init__(self, latent_dim=32):  # match NB03
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 2304)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        return self.sigmoid(self.fc5(h4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
# --- Load pre-trained model ---
vae_model = VAE(latent_dim=32).to(device)
vae_model.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae_model.eval()
print("✅ Correct VAE loaded successfully.")
# -------------------------------------------
# Setup: Reproducibility, Device
# -------------------------------------------
# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# -------------------------------------------
# Load FER2013-small and Prepare Validation Tensors
# -------------------------------------------
# Load the downsampled dataset saved earlier
data = np.load("fer2013_small.npz")
X, y = data["X"], data["y"]          # X: (N, 48, 48) in [0,1], y: (N,)

# Train/Validation split (stratified to preserve class proportions)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Torch tensors: we will encode flattened vectors for the (fully-connected) VAE
X_val_t = torch.tensor(X_val, dtype=torch.float32).view(X_val.shape[0], -1)  # (N, 2304)

print(f"✅ Data loaded. Train: {X_train.shape}, Val: {X_val.shape}")
# -------------------------------------------
# recreate the exact same class definition so vae_model.pth loads
# -------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):  # match NB03
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 2304)
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        return self.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
# -------------------------------------------
# Load Trained Models (VAE Mandatory, AE Optional)
# -------------------------------------------
# Load trained VAE (NB03 checkpoint)
vae_model = VAE(latent_dim=32).to(device)
vae_model.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae_model.eval()
print("✅ Loaded trained VAE (latent_dim=32).")

# Optional: load AE for later comparison if you saved it as ae_model.pth
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2304),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

try:
    ae_model = Autoencoder().to(device)
    ae_model.load_state_dict(torch.load("ae_model.pth", map_location=device))
    ae_model.eval()
    print("✅ Loaded trained AE (optional).")
except Exception as e:
    print("ℹ️ AE checkpoint not found; we will proceed with VAE-only analysis.")

# -----------------------------------------------------------------------------------
# CLASSIFICATION TASK (CT): Load Flattened Data into DataLoaders
# -----------------------------------------------------------------------------------
# 1) Load arrays (X: (N, 48, 48) in [0,1], y: (N,))
data = np.load("fer2013_small.npz")
X, y = data["X"], data["y"]

# 2) Stratified split
CT_X_train, CT_X_test, CT_y_train, CT_y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Flatten to (N, 2304)
def CT_flatten(arr):
    t = torch.tensor(arr, dtype=torch.float32)
    return t.view(t.size(0), -1)

CT_X_train_t = CT_flatten(CT_X_train)
CT_X_test_t  = CT_flatten(CT_X_test)

# 4) DataLoaders
CT_train_loader = DataLoader(TensorDataset(CT_X_train_t), batch_size=64, shuffle=True)
CT_test_loader  = DataLoader(TensorDataset(CT_X_test_t),  batch_size=64, shuffle=False)

xb = next(iter(CT_train_loader))[0]
print("CT_Task1 — train batch shape:", xb.shape)
# -----------------------------------------------------------------------------------
# Encode Validation Faces into Latent Mean Vectors
vae_model.eval()

all_mu = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(X_val_t), 128):
        batch = X_val_t[i:i+128].to(device)
        mu, logvar = vae_model.encode(batch)
        all_mu.append(mu.cpu())
        all_labels.append(torch.tensor(y_val[i:i+128]))

# Concatenate all batches
mu_all = torch.cat(all_mu)
labels_all = torch.cat(all_labels)

print(f"✅ Encoded {mu_all.shape[0]} faces into latent space.")
print(f"Latent vector shape: {mu_all.shape[1]} dimensions")

# -------------------------------------------
#Preview a Few Latent Vectors
# -------------------------------------------
# Display first 5 μ vectors
for i in range(5):
    print(f"Sample {i} | Emotion label: {labels_all[i].item()} | μ[:5]: {mu_all[i, :5].numpy()}")
# -------------------------------------------
# Last 5
CT_mu_last5 = mu_all[-5:].numpy()
print("CT_Task2 — CT_mu_last5 shape:", CT_mu_last5.shape)
print(CT_mu_last5)
# -----------------------------------------------------------------------------------
# Project μ to 2D via PCA or t-SNE
# Ensure we have mu_all (N, latent_dim) and labels_all (N,)
try:
    mu_all, labels_all
except NameError:
    # Fallback: compute μ for the validation set
    vae_model.eval()
    all_mu, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_val_t), 128):
            batch = X_val_t[i:i+128].to(device)
            mu, logvar = vae_model.encode(batch)
            all_mu.append(mu.cpu())
            all_labels.append(torch.tensor(y_val[i:i+128]))
    mu_all = torch.cat(all_mu)
    labels_all = torch.cat(all_labels)

Z = mu_all.numpy()
y_vis = labels_all.numpy()

# Choose projection: "pca" (fast) or "tsne" (nonlinear)
PROJECTION = "pca"  # change to "tsne" after confirming PCA works

if PROJECTION == "pca":
    Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
else:
    # Optional PCA pre-reduction to 30 dims for t-SNE speed/stability
    Z30 = PCA(n_components=min(30, Z.shape[1]), random_state=42).fit_transform(Z)
    Z2 = TSNE(
        n_components=2, perplexity=30, learning_rate="auto",
        init="pca", random_state=42
    ).fit_transform(Z30)
# -----------------------------------------------------------------------------------
# Scatter Plot Colored by Emotion
# FER2013 label order
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

plt.figure(figsize=(7,6))
sc = plt.scatter(Z2[:,0], Z2[:,1], c=y_vis, s=10, cmap="tab10", alpha=0.9)
# Build legend with fixed labels
handles = []
for k, name in enumerate(emotions):
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.tab10(k/10), label=name))
plt.legend(handles=handles, title="Emotion", loc="best", fontsize=9)
plt.title(f"VAE Latent Space (μ) — 2D via {PROJECTION.upper()}")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.tight_layout(); plt.show()
# -----------------------------------------------------------------------------------
# Optional: Overlay AE Latent Space for Contrast
try:
    # Encode with AE
    ae_model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(X_val_t), 128):
            batch = X_val_t[i:i+128].to(device)
            _, z_ae = ae_model(batch)      # our AE forward returns (x_hat, z)
            zs.append(z_ae.cpu())
    Z_AE = torch.cat(zs).numpy()

    # Project AE embeddings with the same method
    if PROJECTION == "pca":
        Z2_AE = PCA(n_components=2, random_state=42).fit_transform(Z_AE)
    else:
        Z30_AE = PCA(n_components=min(30, Z_AE.shape[1]), random_state=42).fit_transform(Z_AE)
        Z2_AE = TSNE(
            n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=42
        ).fit_transform(Z30_AE)

    # --- MINIMAL FIX: discrete cmap + norm + per-axes legends ---
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    # discrete colors from tab10, one per class id
    _base = plt.get_cmap("tab10").colors
    _colors = _base[:len(emotions)]
    _cmap = ListedColormap(_colors)
    # each integer label maps to exactly one color
    _norm = mpl.colors.BoundaryNorm(boundaries=np.arange(-0.5, len(emotions)+0.5, 1),
                                    ncolors=len(emotions))

    # Side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    # ---- VAE plot ----
    axes[0].scatter(Z2[:,0], Z2[:,1], c=y_vis, s=10, cmap=_cmap, norm=_norm, alpha=0.9)
    axes[0].set_title(f"VAE (μ) — {PROJECTION.upper()}")
    axes[0].set_xlabel("Dim 1"); axes[0].set_ylabel("Dim 2")
    # per-axes legend using the exact same colors
    handles0 = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=_colors[k],
                           markeredgecolor='none', label=emotions[k]) for k in range(len(emotions))]
    axes[0].legend(handles=handles0, title="Emotion", fontsize=8, loc="best")

    # ---- AE plot ----
    axes[1].scatter(Z2_AE[:,0], Z2_AE[:,1], c=y_vis, s=10, cmap=_cmap, norm=_norm, alpha=0.9)
    axes[1].set_title(f"AE (z) — {PROJECTION.upper()}")
    axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")
    handles1 = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=_colors[k],
                           markeredgecolor='none', label=emotions[k]) for k in range(len(emotions))]
    axes[1].legend(handles=handles1, title="Emotion", fontsize=8, loc="best")

    plt.tight_layout()
    plt.show()

except NameError:
    print("AE model not available — skipping AE vs VAE overlay.")

# -----------------------------------------------------------------------------------
# Plotting only 2 Emotions: Fear vs Neutral and Happy vs Sad
# --- Side-by-side PCA for selected pairs (VAE vs AE), using the SAME colors as before ---

import matplotlib as mpl
from matplotlib.colors import ListedColormap

# Reuse the same discrete cmap/norm if already defined; otherwise define them
try:
    _cmap, _norm, _colors
except NameError:
    base = plt.get_cmap("tab10").colors
    _colors = base[:len(emotions)]
    _cmap = ListedColormap(_colors)
    _norm = mpl.colors.BoundaryNorm(boundaries=np.arange(-0.5, len(emotions)+0.5, 1),
                                    ncolors=len(emotions))

# Define the pairs we want to visualize (using FER2013 indices)
pairs = [
    (2, 6, "Fear vs Neutral"),  # Fear=2, Neutral=6
    (0, 5, "Angry vs Surprise"),  # Angry=0, Surprise=5
    (2, 3, "Fear vs Happy"),      # Fear=2, Happy
]

try:
    for a, b, title in pairs:
        mask = np.isin(y_vis, [a, b])

        # Prepare per-axes legends (only the two classes in this pair)
        handles = [
            plt.Line2D([], [], marker="o", linestyle="",
                       markerfacecolor=_colors[a], markeredgecolor="none",
                       label=emotions[a]),
            plt.Line2D([], [], marker="o", linestyle="",
                       markerfacecolor=_colors[b], markeredgecolor="none",
                       label=emotions[b]),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

        # VAE (μ) — PCA subset
        axes[0].scatter(Z2[mask, 0], Z2[mask, 1], c=y_vis[mask], s=12, cmap=_cmap, norm=_norm, alpha=0.9)
        axes[0].set_title(f"VAE (μ) — {title} — PCA")
        axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
        axes[0].legend(handles=handles, title="Emotion", fontsize=8, loc="best")

        # AE (z) — PCA subset
        axes[1].scatter(Z2_AE[mask, 0], Z2_AE[mask, 1], c=y_vis[mask], s=12, cmap=_cmap, norm=_norm, alpha=0.9)
        axes[1].set_title(f"AE (z) — {title} — PCA")
        axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
        axes[1].legend(handles=handles, title="Emotion", fontsize=8, loc="best")

        plt.tight_layout()
        plt.show()

except NameError:
    print("AE model not available — skipping emotion pair comparisons.")

# -----------------------------------------------------------------------------------
# CLASSIFICATION TASK (CT): PCA on Test Set for Angry vs Neutral
# -----------------------------------------------------------------------------------

# Need full μ (test) and labels from our split:
CT_y_test = torch.tensor(CT_y_test)  # ensure tensor for masking

# Encode test set with VAE to get μ
vae_model.eval()
CT_all_mu = []
with torch.no_grad():
    for (xb,) in CT_test_loader:
        mu, logvar = vae_model.encode(xb.to(device))
        CT_all_mu.append(mu.cpu())
CT_all_mu = torch.cat(CT_all_mu, dim=0)

# If AE model exists, compute AE codes; else set to None
try:
    ae_model
    ae_model.eval()
    CT_all_z = []
    with torch.no_grad():
        for (xb,) in CT_test_loader:
            z = ae_model.encoder(xb.to(next(ae_model.parameters()).device))
            CT_all_z.append(z.cpu())
    CT_all_z = torch.cat(CT_all_z, dim=0).numpy()
except NameError:
    CT_all_z = None

# Mask for Angry (0) and Neutral (6)
# Ensure CT_y_test matches the size of CT_all_mu
N_samples = len(CT_all_mu)
CT_y_test_aligned = CT_y_test[:N_samples]
mask = ((CT_y_test_aligned == 0) | (CT_y_test_aligned == 6)).squeeze()

# PCA for VAE μ
pca = PCA(n_components=2, random_state=42)
CT_pca_vae_ang_neu = pca.fit_transform(CT_all_mu[mask].numpy())

# PCA for AE z (optional)
if CT_all_z is not None:
    CT_all_z_aligned = CT_all_z[:N_samples]
    pca_ae = PCA(n_components=2, random_state=42)
    CT_pca_ae_ang_neu = pca_ae.fit_transform(CT_all_z_aligned[mask.numpy()])
else:
    CT_pca_ae_ang_neu = None

# Plot
fig, axes = plt.subplots(1, 2 if CT_pca_ae_ang_neu is not None else 1, figsize=(10,4))
axes = np.atleast_1d(axes)
axes[0].scatter(CT_pca_vae_ang_neu[:,0], CT_pca_vae_ang_neu[:,1], c=CT_y_test_aligned.numpy()[mask.numpy()], s=10, cmap="tab10")
axes[0].set_title("VAE μ — Angry vs Neutral (PCA)")
if CT_pca_ae_ang_neu is not None:
    axes[1].scatter(CT_pca_ae_ang_neu[:,0], CT_pca_ae_ang_neu[:,1], c=CT_y_test_aligned.numpy()[mask.numpy()], s=10, cmap="tab10")
    axes[1].set_title("AE z — Angry vs Neutral (PCA)")
plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------------
# Selecting Two Faces with Distinct Emotions
# Helper: pick first example of each target emotion
def get_face_by_emotion(emotion_label, X, y):
    idx = np.where(y == emotion_label)[0][0]
    return X[idx]

# Emotion labels: ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
emotion_map = {i: e for i, e in enumerate(["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])}

happy_face = get_face_by_emotion(3, X_val, y_val)  # Happy
fear_face   = get_face_by_emotion(2, X_val, y_val)  # Fear

# Convert to tensors
x_happy = torch.tensor(happy_face, dtype=torch.float32).view(1, -1).to(device)
x_fear   = torch.tensor(fear_face, dtype=torch.float32).view(1, -1).to(device)

# Encode to latent means
vae_model.eval()
with torch.no_grad():
    mu_happy, _ = vae_model.encode(x_happy)
    mu_fear, _   = vae_model.encode(x_fear)

print("Happy μ:", mu_happy.shape, " Sad μ:", mu_fear.shape)
# ----------------------------------------------
# Interpolation coefficients
# ----------------------------------------------
alphas = np.linspace(0, 1, 10)  # 10 steps from happy → sad

interpolated_faces = []
vae_model.eval()
with torch.no_grad():
    for a in alphas:
        z = (1 - a) * mu_happy + a * mu_fear
        recon = vae_model.decode(z).cpu().view(48, 48)
        interpolated_faces.append(recon.numpy())
# ----------------------------------------------
# Visualizing the Emotion Transition
# ----------------------------------------------
# Plot the sequence from happy → sad
fig, axes = plt.subplots(1, len(interpolated_faces), figsize=(14, 2))
for i, img in enumerate(interpolated_faces):
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"{alphas[i]:.1f}")
plt.suptitle("Latent Interpolation: Happy → Fear", fontsize=12)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------
# AE Comparison (Optional)
try:
    ae_model.eval()
    with torch.no_grad():
        z_happy = ae_model.encoder(x_happy)
        z_fear = ae_model.encoder(x_fear)
        ae_interpolations = []
        for a in alphas:
            z = (1 - a) * z_happy + a * z_fear
            recon = ae_model.decoder(z).cpu().view(48, 48)
            ae_interpolations.append(recon.numpy())

    fig, axes = plt.subplots(2, len(alphas), figsize=(14, 4))
    for i, img in enumerate(interpolated_faces):
        axes[0, i].imshow(img, cmap="gray"); axes[0, i].axis("off")
    for i, img in enumerate(ae_interpolations):
        axes[1, i].imshow(img, cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("VAE", rotation=0, labelpad=20)
    axes[1, 0].set_ylabel("AE", rotation=0, labelpad=20)
    plt.suptitle("Latent Interpolation: Happy → Fear (VAE vs AE)", fontsize=12)
    plt.tight_layout()
    plt.show()

except NameError:
    print("AE model not loaded; skipping comparison.")
# -----------------------------------------------------------------------------------
# Comparing AE vs. VAE Representations
vae_model.eval()
try:
    ae_model.eval()
except NameError:
    ae_model = None

# Compute latent representations for both AE and VAE
vae_latents = []
ae_latents = []

with torch.no_grad():
    for i in range(0, len(X_val_t), 128):
        xb = X_val_t[i:i+128].to(device)
        mu, _ = vae_model.encode(xb)
        vae_latents.append(mu.cpu())
        if ae_model:
            _, z = ae_model(xb)
            ae_latents.append(z.cpu())

vae_latents = torch.cat(vae_latents).numpy()
if ae_model:
    ae_latents = torch.cat(ae_latents).numpy()

# Compute statistics
print(f"VAE latent mean: {vae_latents.mean():.4f}, std: {vae_latents.std():.4f}")
if ae_model:
    print(f"AE latent mean: {ae_latents.mean():.4f}, std: {ae_latents.std():.4f}")

# ----------------------------------------------
# Visualizing Latent Distributions
# ----------------------------------------------
fig, axes = plt.subplots(1, 2 if ae_model else 1, figsize=(10, 4), sharey=True)
axes = np.atleast_1d(axes)

axes[0].hist(vae_latents.flatten(), bins=50, color="steelblue", alpha=0.8)
axes[0].set_title("VAE Latent Distribution")
axes[0].set_xlabel("Latent Value"); axes[0].set_ylabel("Frequency")

if ae_model:
    axes[1].hist(ae_latents.flatten(), bins=50, color="darkorange", alpha=0.8)
    axes[1].set_title("AE Latent Distribution")
    axes[1].set_xlabel("Latent Value")

plt.tight_layout()
plt.show()
