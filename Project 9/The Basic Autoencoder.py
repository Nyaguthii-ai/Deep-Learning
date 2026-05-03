import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from math import inf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load and Inspect the Dataset for class balance and data quality
# Load the downsampled dataset
data = np.load("fer2013_small.npz")
X, y = data["X"], data["y"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Pixel value range:", X.min(), "to", X.max())
# -----------------------------------------------------------------------------------------------
# Visualizing Sample Images
# Define emotion labels for readability
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

plt.figure(figsize=(10, 10))

for i, label in enumerate(range(7)):
    # Pick 5 random samples from each emotion
    idx = np.where(y == label)[0]
    sample_idx = np.random.choice(idx, size=5, replace=False)

    for j, img_idx in enumerate(sample_idx):
        plt.subplot(7, 5, i * 5 + j + 1)
        plt.imshow(X[img_idx], cmap="gray")
        if j == 2:
            plt.title(emotions[label], fontsize=10)
        plt.axis("off")

plt.suptitle("Sample Faces from FER2013 by Emotion", fontsize=14)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# Check if the downsampled dataset remains balanced across the seven emotion categories.
# Compute class counts
unique, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(7, 4))
plt.bar(emotions, counts, color="gray", edgecolor="black")
plt.title("Class Distribution in FER2013 Subset")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.show()

print(dict(zip(emotions, counts)))
# -----------------------------------------------------------------------------------------------

# CT_Task 1 — one sample per emotion (0..6) and print shapes
import numpy as np

# Ensure X, y exist
try:
    X, y
except NameError:
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]  # X: (N,48,48), y: (N,)

CT_emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Build a dict: emotion_id -> one index in y
CT_one_idx_per_emotion = {}
for eid in range(7):
    idxs = np.where(y == eid)[0]
    # pick the first match for determinism
    CT_one_idx_per_emotion[eid] = idxs[0]  # e.g., idxs[0]

# Gather samples
CT_one_img_per_emotion = {eid: X[CT_one_idx_per_emotion[eid]] for eid in range(7)}

# Print shapes and labels
for eid in range(7):
    print(CT_emotions[eid], "-> index:", CT_one_idx_per_emotion[eid],
          " sample shape:", CT_one_img_per_emotion[eid].shape)
# -----------------------------------------------------------------------------------------------
# Building a Basic Autoencoder in PyTorch
# each face image is grayscale with size 48×48, we first flatten it into a vector of length 2,304. 
# The encoder compresses this vector into a 128-dimensional latent representation, 
# and the decoder expands it back to 2,304 values. 
# Finally, we reshape the output vector to recover the original 48×48 image.
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
        # Flatten to  (N, 2304)
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
# -----------------------------------------------------------------------------------------------
# Instantiate the Model and Verify Shapes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FC_Autoencoder().to(device)

# Sanity check with a dummy batch
with torch.no_grad():
    dummy = torch.rand(8, 1, 48, 48, device=device)  # batch of 8 images in [0, 1]
    out = model(dummy)
    print("Input shape :", tuple(dummy.shape))
    print("Output shape:", tuple(out.shape))
# --------------------------------------------
# Lightweight Model Summary
def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

print(model)  # architecture overview
print("Trainable parameters:", count_params(model))
# -----------------------------------------------------------------------------------------------
# CT_Task 3 — sanity batch pass (16)
# Ensure X is present
try:
    X
except NameError:
    data = np.load("fer2013_small.npz"); X, y = data["X"], data["y"]

# Make a 16-sample batch tensor in [0,1]
CT_x16 = torch.tensor(X[:16], dtype=torch.float32).unsqueeze(1)  # (16,1,48,48)

with torch.no_grad():
    CT_out16 = CT_ae(CT_x16)

print("CT_x16 shape:", CT_x16.shape, " | CT_out16 shape:", CT_out16.shape)
# -----------------------------------------------------------------------------------------------
# Prepare Train/Test Splits and DataLoaders
# If X, y are not already in memory, load them:
try:
    X, y
except NameError:
    data = np.load("fer2013_small.npz")
    X, y = data["X"], data["y"]

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Convert to torch tensors and add channel dimension (N, 1, 48, 48)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)

# Build datasets and dataloaders
train_ds = TensorDataset(X_train_t)  # labels not needed
test_ds  = TensorDataset(X_test_t, torch.tensor(y_test))

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

len(train_loader), len(test_loader)
# --------------------------------------------
# Define Loss and Optimizer
# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # reuse the model defined earlier

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# -----------------------------------------------------------------------------------------------
# Training Loop

EPOCHS = 100 
best_test_loss = inf

for epoch in range(1, EPOCHS + 1):
    # ----- Train -----
    model.train()
    running_train_loss = 0.0
    for (xb,) in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        xhat = model(xb)
        loss = criterion(xhat, xb)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * xb.size(0)

    avg_train_loss = running_train_loss / len(train_ds)

    # ----- Evaluate (reconstruction on held-out set) -----
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            xhat = model(xb)
            loss = criterion(xhat, xb)
            running_test_loss += loss.item() * xb.size(0)

    avg_test_loss = running_test_loss / len(test_ds)

    print(f"Epoch {epoch:02d} | train MSE: {avg_train_loss:.5f} | test MSE: {avg_test_loss:.5f}")

    # Optionally keep best weights
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
# --------------------------------------------
# Restore best test-loss weights (optional but recommended)
# --------------------------------------------
if 'best_state' in globals():
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "ae_model.pth")
# -----------------------------------------------------------------------------------------------
# **Show Original vs. Reconstructed Pairs**
# We assume `model` and `test_loader` already exist from previous sections.
model.eval()

# Get a single batch from the test loader
xb = next(iter(test_loader))[0].to(next(model.parameters()).device)

with torch.no_grad():
    xhat = model(xb)

# Move to CPU and convert to numpy for plotting
orig = xb.detach().cpu().numpy()      # (N, 1, 48, 48)
reco = xhat.detach().cpu().numpy()    # (N, 1, 48, 48)

# Select up to 5 examples to display
n_rows = min(5, orig.shape[0])

fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2*n_rows))
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)  # ensure 2D indexing

for i in range(n_rows):
    # Original
    axes[i, 0].imshow(orig[i, 0, :, :], cmap="gray", vmin=0.0, vmax=1.0)
    axes[i, 0].set_title("Original", fontsize=10)
    axes[i, 0].axis("off")

    # Reconstruction
    axes[i, 1].imshow(reco[i, 0, :, :], cmap="gray", vmin=0.0, vmax=1.0)
    axes[i, 1].set_title("Reconstruction", fontsize=10)
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# Can Emotions Be Separated in Latent Space? we will project the 128-D latent vectors into 2D using:
# PCA – linear projection (captures global variance)
# t-SNE – nonlinear projection (captures local structure)
model.eval()
latents, labels = [], []

with torch.no_grad():
    for xb, in test_loader:
        xb = xb.to(next(model.parameters()).device)
        n = xb.size(0)
        z = model.encoder(xb.view(n, -1))  # (N, 128)
        latents.append(z.cpu().numpy())

latents = np.concatenate(latents, axis=0)
labels = y_test  # from earlier split

# Emotion indices
HAPPY = 3
SURPRISE = 5
DISGUST = 1

emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

pairs = [
    (HAPPY, SURPRISE, f"Happy vs Surprise"),
    (HAPPY, DISGUST, f"Happy vs Disgust"),
]

for a, b, title in pairs:
    # Filter the two classes
    mask = np.isin(labels, [a, b])
    L = latents[mask]
    y_pair = labels[mask]

    # Remap to 0 and 1 for consistent coloring
    y_binary = (y_pair == a).astype(int)  # 1 = Happy, 0 = Other emotion

    # Standardize latent vectors
    L_std = StandardScaler().fit_transform(L)

    # ----- PCA (2D) -----
    L_pca = PCA(n_components=2, random_state=42).fit_transform(L_std)

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(L_pca[:,0], L_pca[:,1],
                          c=y_binary, cmap="bwr", s=15, alpha=0.8)

    # Legend
    labels_legend = [emotion_names[a], emotion_names[b]]
    handles = [
        plt.Line2D([], [], marker="o", color="w",
                   markerfacecolor="blue", markersize=8, label=labels_legend[1]),  # 0 class
        plt.Line2D([], [], marker="o", color="w",
                   markerfacecolor="red", markersize=8, label=labels_legend[0])   # 1 class (Happy)
    ]
    plt.legend(handles=handles, title="EEmotions", loc="best")

    plt.title(f"{title} (PCA Projection)")
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.show()

    # ----- t-SNE (with PCA-30 init) -----
    L_pca_30 = PCA(n_components=30, random_state=42).fit_transform(L_std)
    L_tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    ).fit_transform(L_pca_30)

    plt.figure(figsize=(6,5))
    plt.scatter(L_tsne[:,0], L_tsne[:,1],
                c=y_binary, cmap="bwr", s=15, alpha=0.8)

    # Legend again
    plt.legend(handles=handles, title="Emotions", loc="best")

    plt.title(f"{title} (t-SNE Projection)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.show()

# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
