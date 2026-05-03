import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Device setup (CPU-friendly, as in your NB04)
device = torch.device("cpu")
print("Device in use:", device)

#----------------------------------------------------------------------------------------------------
# === EXACT SAME GENERATOR AS IN PRETRAINING (self.net, *4 plan) ===
class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_maps=64):
        super().__init__()
        self.net = nn.Sequential(
            # (N, z_dim, 1, 1) -> (N, feature_maps*4, 4, 4)
            nn.ConvTranspose2d(z_dim, feature_maps*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps*2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps, feature_maps//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps//2),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps//2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)
#------------------------------------------------
# === Load pretrained generator (matches your training script exactly) ===
#------------------------------------------------
G_pretrained = GeneratorDCGAN(z_dim=100, img_channels=3, feature_maps=64).to(device)
state = torch.load("dcgan_pretrained.pth", map_location=device)
G_pretrained.load_state_dict(state, strict=True)
G_pretrained.eval()
print("✅ Pretrained DCGAN Generator loaded successfully (feature_maps=64, self.net).")
#----------------------------------------------------------------------------------------------------
# Setup: we expect a pretrained generator named G_pretrained (from Section 2)

assert 'G_pretrained' in globals(), "Please run Section 2 to load `G_pretrained`."
G_eval = G_pretrained.eval()
device = next(G_eval.parameters()).device

latent_dim = 100

def denorm(x):
    return (x + 1) / 2

@torch.no_grad()
def decode_latents(Z):
    """
    Z: (N, latent_dim, 1, 1)
    Returns: images tensor in [-1, 1]
    """
    return G_eval(Z).cpu()

def show_grid(imgs, nrow, title):
    grid = torchvision.utils.make_grid(denorm(imgs).clamp(0,1), nrow=nrow, padding=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis("off")
    plt.title(title)
    plt.show()
#------------------------------------------------
# 1D traversal: vary a single coordinate k and keep others fixed
#------------------------------------------------
torch.manual_seed(7)
z0 = torch.randn(1, latent_dim, 1, 1, device=device)

k = 7                         # which dimension to vary
values = torch.linspace(-2.0, 2.0, steps=11).to(device)

Z = []
for v in values:
    z = z0.clone()
    z[:, k, :, :] = v
    Z.append(z)
Z = torch.cat(Z, dim=0)       # (11, latent_dim, 1, 1)

imgs_1d = decode_latents(Z)
show_grid(imgs_1d, nrow=11, title=f"1D Traversal on z[{k}] from -2 → 2")
#----------------------------------------------------------------------------------------------------
#------------------------------------------------
# Task: 1D traversal on a different coordinate and different seed
#------------------------------------------------
assert 'G_pretrained' in globals(), "Please load G_pretrained earlier."
G_pretrained.eval()
CT_device = next(G_pretrained.parameters()).device
CT_latent_dim = 100

def CT_denorm(x): return (x + 1)/2

# Different seed and z0 than NB example
torch.manual_seed(123)
CT_z0_1d = torch.randn(1, CT_latent_dim, 1, 1, device=CT_device)

CT_k1d = 11                    # pick another latent index
CT_values1d = torch.linspace(-2.0, 2.0, steps=10).to(CT_device)  # 10 steps

CT_Z_list = []
for v in CT_values1d:
    z = z0.clone()
    z[:, CT_k1d, :, :] = v
    CT_Z_list.append(z)

CT_Z_1d = torch.cat(CT_Z_list, dim=0)

with torch.no_grad():
    CT_imgs_trav1d = G_pretrained(CT_Z_1d).cpu()

# Plot in one row
grid = torchvision.utils.make_grid(CT_denorm(CT_imgs_trav1d).clamp(0,1), nrow=10, padding=2)
plt.figure(figsize=(12, 2.5))
plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
plt.title(f"CT_Task 1 — 1D Traversal on z[{CT_k1d}]")
plt.axis("off")
plt.show()

print("CT_Task 1 done: imgs =", tuple(CT_imgs_trav1d.shape))
#------------------------------------------------
# 2D traversal: vary two coordinates (i, j) across a grid
#------------------------------------------------
torch.manual_seed(13)
z0 = torch.randn(1, latent_dim, 1, 1, device=device)

i, j = 3, 17                     # two latent dimensions to sweep
grid_lin = torch.linspace(-2.0, 2.0, steps=7).to(device)  # 7x7 grid

Z_list = []
for a in grid_lin:               # vertical axis (row)
    for b in grid_lin:           # horizontal axis (col)
        z = z0.clone()
        z[:, i, :, :] = a
        z[:, j, :, :] = b
        Z_list.append(z)

Z_grid = torch.cat(Z_list, dim=0)  # (49, latent_dim, 1, 1)
imgs_2d = decode_latents(Z_grid)
show_grid(imgs_2d, nrow=len(grid_lin), title=f"2D Traversal on (z[{i}], z[{j}]) over [-2, 2]^2")
#------------------------------------------------
# Partial-fix traversal: vary a small set of dims jointly, others fixed
#------------------------------------------------
torch.manual_seed(21)
z0 = torch.randn(1, latent_dim, 1, 1, device=device)

vary_idx = torch.tensor([2, 5, 9, 12, 27], device=device)  # a few coords to vary
steps = 8
alphas = torch.linspace(-1.5, 1.5, steps=steps).to(device)

Z_list = []
for a in alphas:
    z = z0.clone()
    z[:, vary_idx, :, :] = a
    Z_list.append(z)
Z_line = torch.cat(Z_list, dim=0)  # (steps, latent_dim, 1, 1)

imgs_partial = decode_latents(Z_line)
show_grid(imgs_partial, nrow=steps, title=f"Partial-Fix Traversal on dims {vary_idx.tolist()} (-1.5 → 1.5)")
#----------------------------------------------------------------------------------------------------
# --- Ensure pretrained generator G is loaded ---
assert 'G_pretrained' in globals(), "Please run Section 3 to load pretrained generator as 'G_pretrained'."
G_pretrained.eval()
device = next(G_pretrained.parameters()).device

# --- Linear interpolation function ---
@torch.no_grad()
def interpolate_linearly(z1, z2, steps=10):
    """
    Linear interpolation between z1 and z2 in latent space.
    z1, z2: tensors of shape (1, z_dim, 1, 1)
    returns: tensor of shape (steps, z_dim, 1, 1)
    """
    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device)
    Z = [(1 - a) * z1 + a * z2 for a in alphas]
    return torch.cat(Z, dim=0)

# --- Sample latent endpoints ---
torch.manual_seed(42)
z_dim = 100
z1 = torch.randn(1, z_dim, 1, 1, device=device)
z2 = torch.randn(1, z_dim, 1, 1, device=device)

# --- Generate interpolation sequence ---
Z_line = interpolate_linearly(z1, z2, steps=10)
with torch.no_grad():
    imgs = G_pretrained(Z_line).cpu()
imgs = (imgs + 1) / 2  # denormalize to [0,1]

# --- Plot horizontal gallery ---
grid = make_grid(imgs, nrow=10, padding=2)
plt.figure(figsize=(14, 2.2))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.axis("off")
plt.title("Latent Interpolation — $z_1 \\rightarrow z_2$ (10 steps)")
#----------------------------------------------------------------------------------------------------

# --- Ensure pretrained generator G is loaded ---
assert 'G' in globals(), "Please run Section 3 to load pretrained generator as 'G'."
G.eval()
device = next(G.parameters()).device
z_dim = 100

@torch.no_grad()
def blend_latent_styles(zA, zB, split_index=50):
    """
    Create a hybrid latent vector by combining the first part of zA and the rest of zB.
    """
    return torch.cat([zA[:, :split_index, :, :], zB[:, split_index:, :, :]], dim=1)

# --- Random seeds for reproducibility ---
torch.manual_seed(100)
zA = torch.randn(1, z_dim, 1, 1, device=device)
zB = torch.randn(1, z_dim, 1, 1, device=device)
z_mix = blend_latent_styles(zA, zB, split_index=50)

# --- Generate the three paintings ---
with torch.no_grad():
    imgs = G(torch.cat([zA, z_mix, zB], dim=0)).cpu()

# --- Normalize and display ---
imgs = (imgs + 1) / 2
titles = ["Style A", "Mixed (A+B)", "Style B"]

grid = make_grid(imgs, nrow=3, padding=5)
plt.figure(figsize=(9, 3))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.axis("off")

plt.title("Style Blending in Latent Space (Partial Vector Mixing)")
plt.show()
#----------------------------------------------------------------------------------------------------
