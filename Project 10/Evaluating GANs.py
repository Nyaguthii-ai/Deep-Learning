# --- Imports and setup ---
import torch
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import pandas as pd
import random
# ------------------------------------------------------------------------------
# Pretrained DCGAN Model
# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transformations for dataset ---
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Load the WikiArt subset (already balanced & downsampled) ---
data_path = "WikiartSubsetDataset/WikiartSubsetDataset/train"  # adjust path if needed
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# --- Define DCGAN Generator architecture ---
class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=100, feature_g=64, channels_img=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# --- Load pretrained model (match channels to checkpoint) ---
G_pretrained = GeneratorDCGAN(z_dim=100, feature_g=32, channels_img=3).to(device)
state = torch.load("dcgan_pretrained.pth", map_location=device)
G_pretrained.load_state_dict(state, strict=True)
G_pretrained.eval()
print("✅ Pretrained DCGAN Generator loaded successfully with feature_g=32.")


# --- Generate random art samples ---
z = torch.randn(25, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = G_pretrained(z).detach().cpu()

# --- Denormalize and visualize ---
grid = utils.make_grid(fake_images, nrow=5, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.axis("off")
plt.title("Generated Art Samples (Pretrained DCGAN)")
plt.show()

# ------------------------------------------------------------------------------
# --- Helper: visualize a batch as grid ---
def show_grid(images, title, nrow=5):
    grid = utils.make_grid(images, nrow=nrow, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.title(title)
    plt.show()

# --- Fixed latent vectors for consistent comparison ---
torch.manual_seed(42)
fixed_z = torch.randn(25, 100, 1, 1, device=device)

# --- Function to load a checkpoint and generate images ---
def generate_from_checkpoint(ckpt_path):
    model = GeneratorDCGAN(z_dim=100, feature_g=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        imgs = model(fixed_z).detach().cpu()
    return imgs

# --- Compare across epochs ---
checkpoints = {
    "Epoch 1 (Early)": "dcgan_epoch1.pth",
    "Epoch 5 (Mid)": "dcgan_epoch5.pth",
    "Epoch 20 (More Matured)": "dcgan_epoch20.pth",
    "Epoch 33 (Final)": "dcgan_pretrained.pth"
}

for stage, path in checkpoints.items():
    imgs = generate_from_checkpoint(path)
    show_grid(imgs, f"Generated Art — {stage}")
# ------------------------------------------------------------------------------
# Code Task — Load training log CSV and plot losses
CT_log_path = "training_log.csv"   # adjust if needed

# 1) Read CSV with expected columns: epoch,d_loss,g_loss
CT_df = pd.read_csv(CT_log_path)

# 2) Select the 3 columns explicitly (and sanity-check length)
CT_epochs = CT_df["epoch"].values
CT_d_loss = CT_df["d_loss"].values
CT_g_loss = CT_df["g_loss"].values

# 3) Plot losses vs epoch
plt.figure(figsize=(7,4))
plt.plot(CT_epochs, CT_d_loss, label="D loss")
plt.plot(CT_epochs, CT_g_loss, label="G loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CT_Task 1 — DCGAN Loss Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print("CT_Task 1:", len(CT_epochs), "epochs loaded.")

# ------------------------------------------------------------------------------
# We assume a trained/pretrained generator is available as G (or G_pretrained).
# Prefer the pretrained one for a strong baseline; replace if needed.
assert 'G_pretrained' in globals() or 'G' in globals(), "Load a generator first (e.g., Section 2)."
G_eval = G_pretrained if 'G_pretrained' in globals() else G
G_eval.eval()

device = next(G_eval.parameters()).device
latent_dim = 100

# Helper: denormalize from [-1,1] to [0,1]
def denorm(x): 
    return (x + 1) / 2

# 1) VISUAL CHECK — generate multiple batches of samples
def show_batch(title, imgs, nrow=8):
    grid = torchvision.utils.make_grid(denorm(imgs).cpu(), nrow=nrow, padding=2, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.show()

with torch.no_grad():
    # Generate 3 different batches from different z’s
    batch_sizes = [64, 64, 64]
    z1 = torch.randn(batch_sizes[0], latent_dim, 1, 1, device=device)
    z2 = torch.randn(batch_sizes[1], latent_dim, 1, 1, device=device)
    z3 = torch.randn(batch_sizes[2], latent_dim, 1, 1, device=device)

    f1 = G_eval(z1)
    f2 = G_eval(z2)
    f3 = G_eval(z3)

show_batch("Batch A (random z)", f1, nrow=8)
show_batch("Batch B (random z)", f2, nrow=8)
show_batch("Batch C (random z)", f3, nrow=8)

# 2) SIMPLE DIVERSITY HEURISTICS (no heavy dependencies)
#    - Pixelwise std across the batch (higher is better)
#    - Mean pairwise cosine similarity between flattened images (lower is better)

def batch_diversity_scores(imgs):
    # imgs: (N, 3, 64, 64) in [-1,1]
    x = denorm(imgs).clamp(0,1)                      # (N,3,64,64) in [0,1]
    x_np = x.cpu().numpy()
    N = x_np.shape[0]

    # Pixelwise std aggregated
    std_per_pixel = x_np.std(axis=0)                 # (3,64,64)
    mean_pixel_std = std_per_pixel.mean()            # scalar

    # Mean pairwise cosine similarity on flattened vectors
    X = x.view(N, -1)                                # (N, 3*64*64)
    X = X - X.mean(dim=1, keepdim=True)              # zero-center each sample
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)     # l2-normalize
    sims = []
    for i in range(N):
        # compare with 10 random others to keep it light
        idxs = torch.randint(low=0, high=N, size=(10,))
        dot = (X[i, None, :] * X[idxs, :]).sum(dim=1)  # cosine similarity
        sims.append(dot.mean().item())
    mean_cos_sim = float(np.mean(sims))

    return mean_pixel_std, mean_cos_sim

with torch.no_grad():
    s1 = batch_diversity_scores(f1)
    s2 = batch_diversity_scores(f2)
    s3 = batch_diversity_scores(f3)

print("Diversity heuristics (higher std, lower cosine similarity are better):")
print(f"Batch A: pixel_std={s1[0]:.4f}, mean_cos_sim={s1[1]:.4f}")
print(f"Batch B: pixel_std={s2[0]:.4f}, mean_cos_sim={s2[1]:.4f}")
print(f"Batch C: pixel_std={s3[0]:.4f}, mean_cos_sim={s3[1]:.4f}")
# ------------------------------------------------------------------------------
# Load a small batch of real images from the WikiArt subset (validation/test split)
# Ensure same preprocessing as training
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Assuming 'wikiart_test_dir' points to test/validation subset
wikiart_test_dir = "WikiartSubsetDataset/WikiartSubsetDataset/train"  # adjust path if needed
real_dataset = torchvision.datasets.ImageFolder(root=wikiart_test_dir, transform=transform)
real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=16, shuffle=True)

real_imgs, _ = next(iter(real_loader))
print(f"Loaded {real_imgs.size(0)} real samples for comparison.")
# -------------------------------------------
# Generate fake samples (same batch size as real images)
# -------------------------------------------
torch.manual_seed(42)
z = torch.randn(real_imgs.size(0), 100, 1, 1, device=device)
fake_imgs = G_pretrained(z).detach().cpu()
# -------------------------------------------
# Helper for visualization
# -------------------------------------------
def show_comparison_grid(real, fake, nrow=8):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    grid_real = torchvision.utils.make_grid((real + 1)/2, nrow=nrow, padding=2)
    grid_fake = torchvision.utils.make_grid((fake + 1)/2, nrow=nrow, padding=2)

    axes[0].imshow(np.transpose(grid_real.numpy(), (1, 2, 0)))
    axes[0].set_title("Real WikiArt Paintings")
    axes[0].axis("off")

    axes[1].imshow(np.transpose(grid_fake.numpy(), (1, 2, 0)))
    axes[1].set_title("Generated Paintings (DCGAN)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

show_comparison_grid(real_imgs[:16], fake_imgs[:16])
# -------------------------------------------
# Optional: Shuffle real + fake images
# -------------------------------------------
all_imgs = torch.cat([real_imgs[:8], fake_imgs[:8]], dim=0)
labels = ["Real"] * 8 + ["Fake"] * 8

# Shuffle jointly
combined = list(zip(all_imgs, labels))
random.shuffle(combined)
shuffled_imgs, shuffled_labels = zip(*combined)

grid_mix = torchvision.utils.make_grid((torch.stack(shuffled_imgs) + 1)/2, nrow=8, padding=2)
plt.figure(figsize=(10, 5))
plt.imshow(np.transpose(grid_mix.numpy(), (1, 2, 0)))
plt.axis("off")
plt.title("Mixed Real + Generated Paintings — Can You Tell Which Are Which?")
plt.show()

# Display label key (for instructor reference)
print("Label order (for instructor validation):")
print(shuffled_labels)
# ------------------------------------------------------------------------------
# CT_Task 4 — Generate fake images matching the real batch size

# Reuse CT_G_epoch20 (loaded in Task 2), CT_z_dim, and CT_real_imgs (from Task 3)
assert 'CT_G_epoch20' in globals(), "CT_G_epoch20 must be loaded (Task 2)."
assert 'CT_z_dim' in globals(), "CT_z_dim must be defined as in Task 2."
assert 'CT_real_imgs' in globals(), "CT_real_imgs must be loaded (Task 3)."

CT_batch_size = CT_real_imgs.shape[0]
CT_device = "cpu"

CT_G_epoch20.eval()
with torch.no_grad():
    CT_z = torch.randn(16, CT_z_dim, 1, 1, device=CT_device)
    CT_fake_batch = CT_G_epoch20(CT_z)  # pass noise through the generator (keep in [-1, 1])

print("CT_Task 4 — CT_fake_batch shape:", tuple(CT_fake_batch.shape))
# ------------------------------------------------------------------------------
# CT_Task 5 — Side-by-side grids: real vs fake
def CT_denorm_01(x):
    # x in [-1,1] -> [0,1]
    return (x + 1) / 2

# Make grids (same nrow)
CT_nrow = 8
CT_comparison_grid_real = torchvision.utils.make_grid(CT_denorm_01(CT_real_imgs[:16]), nrow=CT_nrow, padding=2)
CT_comparison_grid_fake = torchvision.utils.make_grid(CT_denorm_01(CT_fake_batch), nrow=CT_nrow, padding=2)
# ------------------------------------------------------------------------------

