import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn

# ----------------------------------------------------------------------------------------
# The dataset folder WikiartSubsetDataset/ already contains a balanced, 
# preprocessed subset of the original WikiArt collection.

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
data_dir = "WikiartSubsetDataset/WikiartSubsetDataset/train"
wikiart_data = datasets.ImageFolder(root=data_dir, transform=transform)

# Create DataLoader
data_loader = DataLoader(wikiart_data, batch_size=32, shuffle=True)

# Print class names and sample counts
class_names = wikiart_data.classes
class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in class_names}

print("Classes:", class_names)
print("Sample counts per class:")
for c, n in class_counts.items():
    print(f"  {c}: {n}")

#V isualize a few sample images

def imshow_grid(images, labels, classes, num_images=16):
    plt.figure(figsize=(8,8))
    images = images[:num_images]
    labels = labels[:num_images]
    grid_img = np.transpose(torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=True), (1, 2, 0))
    plt.imshow(grid_img)
    plt.title("Sample Images from WikiArt Subset")
    plt.axis("off")
    plt.show()

# Fetch a batch
images, labels = next(iter(data_loader))
imshow_grid(images, labels, class_names)
# ----------------------------------------------------------------------------------------
# CT_Task 1 — Sample 4 Impressionism images (store indices + plot)

# 1) Find the class id (case-insensitive search)
cls_names = getattr(wikiart_data, "classes", [])
CT_impr_class_name = None
for c in cls_names:
    if c.lower() == "impressionism":
        CT_impr_class_name = c
        break
assert CT_impr_class_name is not None, "Impressionism class not found."

CT_impr_class_id = cls_names.index(CT_impr_class_name) # index of "Impressionism" within cls_names

# 2) Collect indices for this class
CT_impr_indices = []
for idx in range(len(wikiart_data)):
    _, lbl = wikiart_data[idx]
    if lbl == CT_impr_class_id:
        CT_impr_indices.append(idx)
# Keep only the first 4
CT_impr_indices = CT_impr_indices[:4]

# 3) Plot in one row
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axes):
    img, _ = wikiart_data[CT_impr_indices[i]]
    # tensor (C,H,W) -> (H,W,C)
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    ax.set_title(CT_impr_class_name)
    ax.axis("off")
plt.suptitle("CT_Task 1 — 4 Impressionism Samples (Unnormalized)")
plt.show()

print("CT_impr_indices:", CT_impr_indices)
# ----------------------------------------------------------------------------------------
# CT_Task 2 — Normalized pipeline views of the same 4 samples

def CT_denorm(x):
    # from [-1,1] back to [0,1]
    return (x * 0.5) + 0.5

# Fetch tensors from wikiart_dataset at the same indices
CT_norm_imgs = []
for idx in CT_impr_indices:
     img, _ = wikiart_data[idx]
     CT_norm_imgs.append(img)

CT_norm_imgs = torch.stack(CT_norm_imgs, dim=0)  # (4,3,64,64) typically

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
grid = torchvision.utils.make_grid(CT_denorm(CT_norm_imgs), nrow=4, padding=2)
ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
ax.set_title("CT_Task 2 — Same 4 Samples (Normalized Pipeline)")
ax.axis("off")
plt.show()

print("CT_norm_imgs shape:", CT_norm_imgs.shape)
# ----------------------------------------------------------------------------------------

# Step 2: Define image transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scales pixels to [-1, 1]
])

# Step 3: Load the dataset with the new transform
data_dir = "WikiartSubsetDataset"
wikiart_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Step 4: Define a DataLoader
batch_size = 32
data_loader = DataLoader(wikiart_dataset, batch_size=batch_size, shuffle=True)

# Step 5: Verify batch shape
images, labels = next(iter(data_loader))
print(f"Batch shape: {images.shape}")
print(f"Pixel range after normalization: [{images.min():.2f}, {images.max():.2f}]")

# Step 6: Visualize a few normalized images

def denorm(img_tensors):
    """Convert normalized tensors back to [0, 1] for visualization."""
    return img_tensors * 0.5 + 0.5

def show_images(images, num=16):
    plt.figure(figsize=(8,8))
    grid = torchvision.utils.make_grid(denorm(images[:num]), nrow=4)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title("Normalized Training Images")
    plt.axis("off")
    plt.show()

show_images(images)

# Step 7: Define latent space dimensionality
latent_dim = 100

# Sample a few random noise vectors
z = torch.randn(5, latent_dim)
print(f"Shape of latent noise vector z: {z.shape}")
# ----------------------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100  # z ~ N(0,1)^100

# ---------------------------
# Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(z_dim, 128*8*8),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16→32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32→64
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)                    # (N, 128*8*8)
        x = x.view(z.size(0), 128, 8, 8)       # (N, 128, 8, 8)
        x = self.upsample(x)                   # (N, 3, 64, 64)
        return x

# ---------------------------
# Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32→16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16→8
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.features(x)
        out = self.classifier(h)
        return out

# ------------------------------------------
# Weight initialization (DCGAN-style)
# ------------------------------------------
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
G.apply(init_weights)
D.apply(init_weights)

print(G)
print(D)
# ----------------------------------------------------------------------------------------
# Training GAN
# Safety checks for variables defined in earlier sections
assert 'G' in globals() and 'D' in globals(), "Please run the model definition cell (Section 4) first."
assert 'data_loader' in globals(), "Please run the data loading cell (Section 2/3) first."
assert 'latent_dim' in globals(), "Please define latent_dim (e.g., latent_dim = 100) first."
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loss and optimizers (DCGAN defaults for Adam)
criterion = nn.BCELoss()
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

# Fixed noise for visualization across epochs
fixed_z = torch.randn(32, latent_dim, device=device)

# Utility: denormalize from [-1, 1] to [0, 1] for display
def denorm(x):
    return (x + 1) / 2

# Training parameters
epochs = 1
log_every = 50  # print frequency (batches)

G_losses, D_losses = [], []

for epoch in range(1, epochs + 1):
    G.train(); D.train()
    for i, (imgs, _) in enumerate(data_loader, start=1):
        # -------------------------
        # 1) Update Discriminator
        # -------------------------
        D.zero_grad()

        # Real images
        real = imgs.to(device)
        bsz = real.size(0)
        labels_real = torch.ones(bsz, 1, device=device)
        out_real = D(real)
        loss_real = criterion(out_real, labels_real)

        # Fake images (detach so gradients don't flow into G on D step)
        z = torch.randn(bsz, latent_dim, device=device)
        fake = G(z).detach()
        labels_fake = torch.zeros(bsz, 1, device=device)
        out_fake = D(fake)
        loss_fake = criterion(out_fake, labels_fake)

        # Total D loss and step
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # -------------------------
        # 2) Update Generator
        # -------------------------
        G.zero_grad()

        z = torch.randn(bsz, latent_dim, device=device)
        gen = G(z)
        out = D(gen)
        # For G, we want D(gen) -> 1 (i.e., look real)
        labels_for_G = torch.ones(bsz, 1, device=device)
        loss_G = criterion(out, labels_for_G)
        loss_G.backward()
        optimizerG.step()

        # Logging
        if i % log_every == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(data_loader)}] "
                  f"D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}")

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    # -------------------------
    # Visualize after each epoch
    # -------------------------
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z).cpu()
    grid = torchvision.utils.make_grid(denorm(samples), nrow=8, padding=2)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(f"Generated Samples after Epoch {epoch}")
    plt.axis("off")
    plt.show()

# Save the model for reuse later (eg in NB02)
torch.save(G.state_dict(), "dcgan_min_generator_final.pth")
torch.save(D.state_dict(), "dcgan_min_discriminator_final.pth")
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
