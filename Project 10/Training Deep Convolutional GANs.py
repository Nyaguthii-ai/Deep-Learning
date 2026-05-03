import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
# ----------------------------------------------------------------------------------------
# Define transforms for DCGAN training
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1, 1]
])

# Load the dataset
data_path = "WikiartSubsetDataset/WikiartSubsetDataset/train"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Create DataLoader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Print basic info
print(f"Total images: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")
print(f"Classes: {dataset.classes}")
# ------------------------------------------------
# Function to unnormalize and display a grid of images
# ------------------------------------------------
def imshow_grid(images, n=8):
    grid = images[:n*n]
    grid = grid * 0.5 + 0.5  # unnormalize from [-1,1] to [0,1]
    grid = torch.clamp(grid, 0, 1)
    grid = torchvision.utils.make_grid(grid, nrow=n)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Sample WikiArt Subset Images (64x64)")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

# Get one batch
import torchvision
data_iter = iter(dataloader)
images, labels = next(data_iter)
imshow_grid(images)
# ----------------------------------------------------------------------------------------
# CT_Task 1: Find "Realism" class and plot 4 samples
# ------------------------------------------------
# 1) Find class id (case-insensitive)
CT_realism_class_name = None
for c in dataset.classes:
    if c.lower() == "realism":
        CT_realism_class_name = c
        break
assert CT_realism_class_name is not None, "❌ 'Realism' class not found in dataset.classes"

CT_realism_class_id = dataset.classes.index(CT_realism_class_name)

# 2) Gather indices from this class
CT_realism_indices = []
for i in range(len(dataset)):
    _, lbl = dataset[i]
    if lbl == CT_realism_class_id:
        CT_realism_indices.append(i)

# Keep only first 4
CT_realism_indices = CT_realism_indices[:4]

# 3) Visualize in one row
imgs = []
for idx in CT_realism_indices:
    x, _ = dataset[idx]               # normalized to [-1,1]
    x = (x * 0.5) + 0.5               # denorm to [0,1] for display
    imgs.append(x)

grid = torchvision.utils.make_grid(imgs, nrow=4, padding=2)
plt.figure(figsize=(10,3))
plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
plt.title("CT_Task 1 — 4 Realism Samples")
plt.axis("off")
plt.show()

print("CT_realism_indices:", CT_realism_indices)

# ----------------------------------------------------------------------------------------
# Generator
class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_maps=64):
        """
        z_dim: size of latent vector z
        img_channels: 3 for RGB
        feature_maps: base channel multiplier (64 is DCGAN default)
        """
        super().__init__()
        # We start from (N, z_dim, 1, 1) and grow to (N, 3, 64, 64)
        self.net = nn.Sequential(
            # Input: (N, z_dim, 1, 1) → (N, 256, 4, 4)
            nn.ConvTranspose2d(z_dim, feature_maps*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),

            # (N, 256, 4, 4) → (N, 128, 8, 8)
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),

            # (N, 128, 8, 8) → (N, 64, 16, 16)
            nn.ConvTranspose2d(feature_maps*2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # (N, 64, 16, 16) → (N, 32, 32, 32)
            nn.ConvTranspose2d(feature_maps, feature_maps//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps//2),
            nn.ReLU(True),

            # (N, 32, 32, 32) → (N, 3, 64, 64)
            nn.ConvTranspose2d(feature_maps//2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # Expect z of shape (N, z_dim); reshape to (N, z_dim, 1, 1)
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

# Sanity check: instantiate and print summary-like info
device = "cuda" if torch.cuda.is_available() else "cpu"
G_dcgan = GeneratorDCGAN(z_dim=100, img_channels=3, feature_maps=64).to(device)
print(G_dcgan)

# Test a forward pass to verify shapes
with torch.no_grad():
    z = torch.randn(4, 100, device=device)       # 4 latent vectors
    fake_imgs = G_dcgan(z)                        # (4, 3, 64, 64)
print("Fake image batch shape:", tuple(fake_imgs.shape))

# ----------------------------------------------------------------------------------------
# Discriminator
class DiscriminatorDCGAN(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        """
        img_channels: 3 for RGB
        feature_maps: base channel multiplier (64 is DCGAN default)
        """
        super().__init__()
        self.net = nn.Sequential(
            # (N, 3, 64, 64) -> (N, 64, 32, 32)
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 64, 32, 32) -> (N, 128, 16, 16)
            nn.Conv2d(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 128, 16, 16) -> (N, 256, 8, 8)
            nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 256, 8, 8) -> (N, 512, 4, 4)
            nn.Conv2d(feature_maps*4, feature_maps*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps*8*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.net(x)
        out = self.classifier(h)
        return out

# Instantiate and print summary-like info
device = "cuda" if torch.cuda.is_available() else "cpu"
D_dcgan = DiscriminatorDCGAN(img_channels=3, feature_maps=64).to(device)
print(D_dcgan)

# Shape check with a dummy batch
with torch.no_grad():
    dummy = torch.randn(4, 3, 64, 64, device=device)
    prob = D_dcgan(dummy)
print("Discriminator output shape:", tuple(prob.shape))
# ----------------------------------------------------------------------------------------
# Safety checks: expect G_dcgan and D_dcgan from previous sections
assert 'G_dcgan' in globals() and 'D_dcgan' in globals(), "Please run Sections 3 and 4 to define G_dcgan and D_dcgan first."
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# DCGAN-style weight initialization
# -----------------------------
def dcgan_init(m: nn.Module):
    """
    Initialize Conv/ConvTranspose weights ~ N(0, 0.02), bias = 0
    Initialize BatchNorm weight (gamma) ~ N(1, 0.02), bias (beta) = 0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        # Linear layers are less common in canonical DCGAN, but keep consistent init
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Apply initialization to both networks
G_dcgan.apply(dcgan_init)
D_dcgan.apply(dcgan_init)

print("Initialized weights for G_dcgan and D_dcgan (DCGAN recipe).")
# -----------------------------
# Loss function: Binary Cross-Entropy
# -----------------------------
criterion = nn.BCELoss()

# -----------------------------
# Optimizers: Adam with (lr=2e-4, betas=(0.5, 0.999))
# -----------------------------
lr = 2e-4
betas = (0.5, 0.999)
opt_G = optim.Adam(G_dcgan.parameters(), lr=lr, betas=betas)
opt_D = optim.Adam(D_dcgan.parameters(), lr=lr, betas=betas)

print("Optimizers ready: Adam with lr=0.0002, betas=(0.5, 0.999).")
# -----------------------------
# Fixed noise for consistent visualization across epochs
# -----------------------------
latent_dim = 100  # ensure this matches your Generator z_dim
fixed_batch = 64  # number of images to visualize per snapshot (8x8 grid works nicely)
fixed_z = torch.randn(fixed_batch, latent_dim, device=device).view(fixed_batch, latent_dim, 1, 1)

print("Fixed noise prepared:", fixed_z.shape)
# ----------------------------------------------------------------------------------------
# CT_Task 2: Training with z_dim=64 for 3 epochs
# ------------------------------------------------
# Define latent dim different from 100
CT_z_dim_train = 64

# DataLoader from the existing `dataset`
CT_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

def CT_init_weights(m):
    name = m.__class__.__name__
    if "Conv" in name or "Linear" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)

# Instantiate models
CT_G_train = GeneratorDCGAN(z_dim=CT_z_dim_train).to(device)
CT_D_train = DiscriminatorDCGAN().to(device)
CT_G_train.apply(CT_init_weights)
CT_D_train.apply(CT_init_weights)

criterion = nn.BCELoss()
optD = optim.Adam(CT_D_train.parameters(), lr=2e-4, betas=(0.5, 0.999))
optG = optim.Adam(CT_G_train.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Fixed noise for monitoring
CT_fixed_z64 = torch.randn(16, CT_z_dim_train, device=device)

CT_D_losses_3ep, CT_G_losses_3ep = [], []
CT_samples_after_epochs = []

epochs = 3
for ep in range(1, epochs+1):
    CT_G_train.train(); CT_D_train.train()
    # one batch per epoch
    real_batch, _ = next(iter(CT_loader))
    real = real_batch.to(device)
    bsz = real.size(0)

    # --- Discriminator step ---
    CT_D_train.zero_grad()
    lab_real = torch.ones(bsz, 1, device=device)
    out_real = CT_D_train(real)
    loss_real = criterion(out_real, lab_real)

    z = torch.randn(bsz, CT_z_dim_train, device=device)
    fake_detached = CT_G_train(z).detach()
    lab_fake = torch.zeros(bsz, 1, device=device)
    out_fake = CT_D_train(fake_detached)
    loss_fake = criterion(out_fake, lab_fake)

    loss_D = loss_real + loss_fake
    loss_D.backward()
    optD.step()

    # --- Generator step ---
    CT_G_train.zero_grad()
    z = torch.randn(bsz, CT_z_dim_train, device=device)
    gen = CT_G_train(z)
    out = CT_D_train(gen)
    target = torch.ones(bsz, 1, device=device)
    loss_G = criterion(out, target)
    loss_G.backward()
    optG.step()

    CT_D_losses_3ep.append(loss_D.item())
    CT_G_losses_3ep.append(loss_G.item())
    
    print(f"Epoch {ep} - D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

print("Training complete!")
print("CT_D_losses_3ep:", CT_D_losses_3ep)
print("CT_G_losses_3ep:", CT_G_losses_3ep)
# ----------------------------------------------------------------------------------------
# Safety checks for previously defined objects
assert 'G_dcgan' in globals() and 'D_dcgan' in globals(), "Run Sections 3–5 to define and init G_dcgan/D_dcgan."
assert 'dataloader' in globals(), "Run Section 2 to create dataloader."
assert 'criterion' in globals() and 'opt_G' in globals() and 'opt_D' in globals(), "Run Section 5 to set loss/optimizers."
assert 'fixed_z' in globals(), "Run Section 5 to prepare fixed_z."

device = "cuda" if torch.cuda.is_available() else "cpu"
G_dcgan.train()
D_dcgan.train()

epochs = 3                 # keep it short
log_every = 50             # print frequency
real_label = 1.0
fake_label = 0.0

def denorm(x):
    # [-1,1] -> [0,1]
    return (x + 1) / 2

G_losses, D_losses = [], []

for epoch in range(1, epochs + 1):
    for i, (real_imgs, _) in enumerate(dataloader, start=1):
        bsz = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # ---------------------------
        # 1) Update Discriminator
        # ---------------------------
        opt_D.zero_grad(set_to_none=True)

        # Real batch
        labels_real = torch.full((bsz, 1), real_label, dtype=torch.float, device=device)
        out_real = D_dcgan(real_imgs)
        loss_real = criterion(out_real, labels_real)

        # Fake batch
        z = torch.randn(bsz, 100, device=device).view(bsz, 100, 1, 1)
        fake_imgs = G_dcgan(z).detach()  # detach so G is not updated here
        labels_fake = torch.full((bsz, 1), fake_label, dtype=torch.float, device=device)
        out_fake = D_dcgan(fake_imgs)
        loss_fake = criterion(out_fake, labels_fake)

        # Combine and step D
        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ---------------------------
        # 2) Update Generator
        # ---------------------------
        opt_G.zero_grad(set_to_none=True)
        z = torch.randn(bsz, 100, device=device).view(bsz, 100, 1, 1)
        gen_imgs = G_dcgan(z)
        # For G, pretend fakes are real (target=1)
        labels_gen = torch.full((bsz, 1), real_label, dtype=torch.float, device=device)
        out = D_dcgan(gen_imgs)
        loss_G = criterion(out, labels_gen)
        loss_G.backward()
        opt_G.step()

        # Logging
        if i % log_every == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())

    # ---------------------------
    # Visualize generated samples after each epoch
    # ---------------------------
    G_dcgan.eval()
    with torch.no_grad():
        samples = G_dcgan(fixed_z).cpu()  # (fixed batch)
    G_dcgan.train()

    grid = torchvision.utils.make_grid(denorm(samples), nrow=8, padding=2)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(f"DCGAN — Generated Samples after Epoch {epoch}")
    plt.axis("off")
    plt.show()

# ----------------------------------------------------------------------------------------
