import torch, json
from pathlib import Path
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------
# Load class mapping → stable label order
class_to_idx = torch.load(MAP_PATH, weights_only=False)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)
print(f"[info] Classes ({num_classes}): {class_names}")

# Transforms (ImageNet-compatible) — same as NB01/NB02
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Datasets & DataLoaders
train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=transform)
val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=transform)
test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=transform)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)
# -------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Start from the same architecture
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 2) Resize the classifier head to our classes (same as NB02)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# 3) Load NB02 checkpoint (feature extraction result: frozen backbone + trained head)
state = torch.load(FT_LOAD_PATH, weights_only=False, map_location="cpu")
model.load_state_dict(state, strict=True)

# 4) Freeze everything first
for p in model.parameters():
    p.requires_grad = False

# 5) Unfreeze only the final residual block (layer4) and the classifier head
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

model = model.to(device)
model.eval()

print("[info] Loaded NB02 checkpoint and set trainable layers: layer4 + fc")
# -----------------------------------------
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params

trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

# Safety: ensure only layer4 and fc are trainable
assert all(n.startswith("layer4.") or n.startswith("fc.") for n in trainable_names), \
    "Unexpected trainable layers detected (only layer4 and fc should be trainable)."

print("[check] Trainability looks correct (only layer4 + fc).")
# -------------------------------------------------------------------------------------------
# Loss function (same as NB02)
criterion = nn.CrossEntropyLoss()

# Differential learning rates:
# - layer4 (pretrained, now unfrozen): small LR
# - fc (head): larger LR
param_groups = [
    {"params": model.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3, "weight_decay": 1e-4},
]

optimizer = optim.Adam(param_groups)

# Optional scheduler for gentle LR decay
use_scheduler = True
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5) if use_scheduler else None

# Small helper to print current learning rates per group
def get_lrs(optim_obj):
    return [group["lr"] for group in optim_obj.param_groups]

print("[ok] Optimizer set with differential LRs.")
print("     LRs per group (layer4, fc):", get_lrs(optimizer))
# -------------------------------------------------------------------------------------------
# Train/Eval helpers (fine-tuning)
def ft_train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # Forward
        logits = model(xb)
        loss = criterion(logits, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        bs = xb.size(0)
        running_loss += loss.item() * bs
        total += bs

    return running_loss / max(total, 1)


@torch.no_grad()
def ft_evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        running_loss += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return running_loss / max(total, 1), correct / max(total, 1)
# ----------------------------------------------
# Re-fine tuning
# ----------------------------------------------
num_epochs_ft = 5  
ft_history = {"train_loss": [], "val_loss": [], "val_acc": []}

print(f"[start] Fine-tuning layer4 + head for {num_epochs_ft} epochs...")
t_start = time.time()

for epoch in range(1, num_epochs_ft + 1):
    t0 = time.time()

    tr_loss = ft_train_one_epoch(model, train_loader, optimizer, criterion, device)
    va_loss, va_acc = ft_evaluate(model, val_loader, criterion, device)

    if scheduler is not None:
        scheduler.step()

    ft_history["train_loss"].append(tr_loss)
    ft_history["val_loss"].append(va_loss)
    ft_history["val_acc"].append(va_acc)

    dt = time.time() - t0
    print(f"Epoch {epoch:02d} | "
          f"train_loss={tr_loss:.4f} | "
          f"val_loss={va_loss:.4f} | "
          f"val_acc={va_acc:.3f} | "
          f"epoch_time={dt:.1f}s | LRs={ [g['lr'] for g in optimizer.param_groups] }")

print(f"[done] Finished fine-tuning in {time.time() - t_start:.1f}s")
# -------------------------------------------------------------------------------------------
# Evaluate the fine-tuned model on the test set
# Current model = fine-tuned (layer4 + head) from previous section
model.eval()

y_true, y_pred_ft = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
        y_pred_ft.extend(preds.tolist())
        y_true.extend(yb.numpy().tolist())

y_true = np.array(y_true)
y_pred_ft = np.array(y_pred_ft)

acc_ft = accuracy_score(y_true, y_pred_ft)
f1_ft_macro = f1_score(y_true, y_pred_ft, average="macro")
print(f"[FT] Test accuracy: {acc_ft:.4f} | Macro F1: {f1_ft_macro:.4f}")

report_ft = classification_report(
    y_true, y_pred_ft, target_names=class_names, digits=3, zero_division=0
)
print("\n[FINE-TUNING] Classification report\n")
print(report_ft)

cm_ft = confusion_matrix(y_true, y_pred_ft, labels=list(range(len(class_names))))
# -------------------------------------------------------------------------------------------
# Load the FE baseline (NB02 checkpoint) and evaluate
fe_ckpt_path = Path("./checkpoints/resnet18_feature_extraction.pth")
assert fe_ckpt_path.exists(), "NB02 checkpoint not found: ./checkpoints/resnet18_feature_extraction.pth"

# Recreate the FE model (frozen backbone, trained head)
model_fe = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model_fe.fc.in_features
model_fe.fc = nn.Linear(in_features, len(class_names))

state_fe = torch.load(fe_ckpt_path, weights_only=False, map_location="cpu")
model_fe.load_state_dict(state_fe, strict=True)

# Freeze backbone as in NB02 (not strictly required for inference, but explicit)
for p in model_fe.parameters():
    p.requires_grad = False

model_fe = model_fe.to(device)
model_fe.eval()

y_pred_fe = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model_fe(xb.to(device))
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
        y_pred_fe.extend(preds.tolist())

y_pred_fe = np.array(y_pred_fe)

acc_fe = accuracy_score(y_true, y_pred_fe)
f1_fe_macro = f1_score(y_true, y_pred_fe, average="macro")
print(f"[FE] Test accuracy: {acc_fe:.4f} | Macro F1: {f1_fe_macro:.4f}")

report_fe = classification_report(
    y_true, y_pred_fe, target_names=class_names, digits=3, zero_division=0
)
print("\n[FEATURE EXTRACTION] Classification report\n")
print(report_fe)

cm_fe = confusion_matrix(y_true, y_pred_fe, labels=list(range(len(class_names))))
# ------------------------------------------
#Compact metric comparison table
# ------------------------------------------
compare_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro F1"],
    "Feature Extraction (NB02)": [acc_fe, f1_fe_macro],
    "Fine-Tuning (NB03)": [acc_ft, f1_ft_macro],
})
print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# ------------------------------------------
#Confusion matrices side-by-side 
# ------------------------------------------
# Row-normalize
cm_fe_norm = cm_fe.astype(float) / cm_fe.sum(axis=1, keepdims=True).clip(min=1)
cm_ft_norm = cm_ft.astype(float) / cm_ft.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_fe_norm, cmap="Blues", vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Feature Extraction (NB02)\nConfusion Matrix (row-normalized)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.heatmap(cm_ft_norm, cmap="Greens", vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Fine-Tuning (NB03)\nConfusion Matrix (row-normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
# ------------------------------------------
#Which classes improved the most? 
# ------------------------------------------
prec_fe, rec_fe, f1_fe, _ = precision_recall_fscore_support(y_true, y_pred_fe, labels=list(range(len(class_names))), zero_division=0)
prec_ft, rec_ft, f1_ft, _ = precision_recall_fscore_support(y_true, y_pred_ft, labels=list(range(len(class_names))), zero_division=0)

delta = f1_ft - f1_fe
per_class = pd.DataFrame({
    "class": class_names,
    "F1_FE": f1_fe,
    "F1_FT": f1_ft,
    "ΔF1": delta
}).sort_values("ΔF1", ascending=False)

print("[per-class ΔF1] Positive values mean fine-tuning improved that class.")
display(per_class)
# -------------------------------------------------------------------------------------------
# Pick one validation image (fixed for reproducibility)
sample_path = val_ds.samples[0][0]  # (path, label)
sample_img = Image.open(sample_path).convert("RGB")

prep = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

x = prep(sample_img).unsqueeze(0).to(device)

# --- Hook helpers ---
acts_fe, acts_ft = {}, {}

def hook_capturer_acts(store_dict, name):
    def hook(_m, _i, o):
        store_dict[name] = o.detach().cpu()
    return hook

# Register hooks on the LAST conv in layer4 (good Grad-CAM target too)
handle_fe = model_fe.layer4.register_forward_hook(hook_capturer_acts(acts_fe, "layer4"))
handle_ft = model.layer4.register_forward_hook(hook_capturer_acts(acts_ft, "layer4"))

# Forward both models
with torch.no_grad():
    _ = model_fe(x)
    _ = model(x)

# Remove hooks
handle_fe.remove()
handle_ft.remove()

# Reduce channel dimension with mean to a single 2D map for each model
# fe_map = acts_fe["layer4"][0].mean(dim=1).squeeze(0)   # [H, W]
# ft_map = acts_ft["layer4"][0].mean(dim=1).squeeze(0)   # [H, W]

# acts_fe["layer4"]: [B, C, H, W]
fe_map = acts_fe["layer4"].mean(dim=1).squeeze(0)  # [H, W]
ft_map = acts_ft["layer4"].mean(dim=1).squeeze(0)  # [H, W]


# Normalize for display
def norm01(t):
    t = t - t.min()
    return t / (t.max().clamp(min=1e-8))

fe_disp = norm01(fe_map).numpy()
ft_disp = norm01(ft_map).numpy()

# Show original + activation maps
plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.imshow(sample_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(fe_disp, cmap="viridis")
plt.title("FE layer4 mean-activation")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(ft_disp, cmap="viridis")
plt.title("FT layer4 mean-activation")
plt.axis("off")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------
# ==== 1. Enable grads where we need them for Grad-CAM ====

# For feature-extraction model: everything was frozen; we now allow
# layer4 + fc to have grads so we can compute Grad-CAM.
for p in model_fe.parameters():
    p.requires_grad = False
for p in model_fe.layer4.parameters():
    p.requires_grad = True
for p in model_fe.fc.parameters():
    p.requires_grad = True

# For fine-tuned model: layer4 + fc are already trainable, but we
# explicitly ensure that (harmless if already True).
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

model_fe.eval()
model.eval()

# ==== 2. Grad-CAM helper ====

def gradcam_for_model(model_ref, x_tensor, target_layer):
    """
    Compute Grad-CAM heatmap [H,W] in 0..1 for the model's predicted class.
    """
    model_ref.eval()
    acts = {}
    grads = {}

    # forward hook: save activations
    def fwd_hook(_m, _i, o):
        acts["a"] = o

    # backward hook: save gradients w.r.t. those activations
    def bwd_hook(_m, grad_in, grad_out):
        grads["g"] = grad_out[0]

    # register hooks
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    # grad-trackable input
    x_local = x_tensor.clone().detach().requires_grad_(True)

    # forward
    logits = model_ref(x_local)             # [1, num_classes]
    pred_idx = logits.argmax(dim=1).item()  # scalar int

    # backward w.r.t. predicted logit
    score = logits[0, pred_idx]
    model_ref.zero_grad(set_to_none=True)
    score.backward()

    # remove hooks
    h1.remove()
    h2.remove()

    # activations & grads: [C,H,W]
    A = acts["a"][0]
    G = grads["g"][0]

    # channel weights = GAP over spatial dims of gradients
    weights = G.mean(dim=(1, 2), keepdim=True)    # [C,1,1]
    cam = (weights * A).sum(dim=0)                # [H,W]

    # ReLU (standard Grad-CAM)
    cam = torch.relu(cam)

    # normalize 0..1
    cam = cam.detach().cpu()
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)

    return cam.numpy(), pred_idx

# ==== 3. Run Grad-CAM for both models ====

cam_fe, pred_fe = gradcam_for_model(model_fe, x, model_fe.layer4)
cam_ft, pred_ft = gradcam_for_model(model,    x, model.layer4)

# ==== 4. Prepare original (unnormalized) image ====

def unnorm_for_vis(xb):
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std  = np.array([0.229, 0.224, 0.225])[None, None, :]
    img = xb[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img

base_rgb = unnorm_for_vis(x)  # shape: (H, W, 3)

# ==== 5. Overlay utility (with CAM upsampling) ====
def overlay_cam(img, cam, alpha=0.35, cmap="jet"):
    """
    img: (H, W, 3), cam: (h, w) in [0,1].
    We resize cam to (H, W) before blending.
    """
    H, W = img.shape[:2]

    if cam.shape != (H, W):
        # upsample CAM using bilinear interpolation
        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        cam_t = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam_t.squeeze().cpu().numpy()  # [H,W]

    heat = plt.get_cmap(cmap)(cam)[..., :3]   # RGBA -> RGB
    return (1 - alpha) * img + alpha * heat

ovl_fe = overlay_cam(base_rgb, cam_fe)
ovl_ft = overlay_cam(base_rgb, cam_ft)

# ==== 6. Plot ====

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(base_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ovl_fe)
plt.title(f"Grad-CAM FE (pred={class_names[pred_fe]})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ovl_ft)
plt.title(f"Grad-CAM FT (pred={class_names[pred_ft]})")
plt.axis("off")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------
