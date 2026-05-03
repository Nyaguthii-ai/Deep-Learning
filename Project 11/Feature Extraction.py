# --- Basic Imports and Setup ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os, json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import time
from PIL import Image
from pathlib import Path

# Device configuration (CPU-friendly)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------------------------
# Load pretrained model (ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 10-class Caltech subset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Verify which layers are frozen vs. trainable
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
# -------------------------------------------------------------------------------------------
# --- Paths (must match NB01 saves) ---
DATA_ROOT = Path("./data/caltech101_10")
CKPT_DIR  = Path("./checkpoints")
CKPT_PATH = CKPT_DIR / "resnet18_frozen_init.pth"
MAP_PATH  = CKPT_DIR / "class_to_idx.pth"

assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"
assert CKPT_PATH.exists(), f"Checkpoint not found at {CKPT_PATH} (run NB01 saves first)"
assert MAP_PATH.exists(), f"class_to_idx mapping not found at {MAP_PATH}"

# --- Load class mapping and derive ordered class list ---
class_to_idx = torch.load(MAP_PATH, weights_only = False)
# Build list of class names ordered by their index (0..K-1)
idx_to_class = {v:k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)
print(f"[info] Classes ({num_classes}): {class_names}")

# --- Preprocessing pipeline (must match NB01) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# --- Datasets and DataLoaders (same split folders) ---
train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=transform)
val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=transform)
test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

print(f"[info] Dataset sizes → train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# --- Recreate model (same architecture as NB01) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone
for p in model.parameters():
    p.requires_grad = False

# Replace head sized to our classes (same as NB01) and load the saved state
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only = False)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

print("[info] Loaded frozen backbone + new head from checkpoint.")
# ---------------------------------------------
# Verify trainability and print a compact summary
# ---------------------------------------------
# Count params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# List trainable tensors (should only be 'fc.*')
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

print(f"[summary] Total parameters:     {total_params:,}")
print(f"[summary] Trainable parameters: {trainable_params:,}")
print("[summary] Trainable tensors:")
for n in trainable_names:
    print("  -", n)

# Safety check
assert all(n.startswith("fc.") for n in trainable_names), \
       "Unexpected trainable parameters outside the classifier head."

print("[check] Only the classifier head is trainable — setup is correct.")
# -------------------------------------------------------------------------------------------
# ---------------------------------------------
# Define criterion, optimizer (head-only), and optional scheduler
# ---------------------------------------------
# 1) Loss function: multi-class cross-entropy
criterion = nn.CrossEntropyLoss()

# 2) Optimizer: Adam on the classifier head only
#    (Backbone params are frozen, so they won't appear here.)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

# 3) Optional scheduler: gently decay LR every few epochs
use_scheduler = True
if use_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
else:
    scheduler = None

print("[ok] Loss, optimizer (head-only), and scheduler are set.")
# ---------------------------------------------
# Helper functions: train one epoch & evaluate
# ---------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # enables dropout/bn if present (not critical here but good practice)
    running_loss = 0.0
    total = 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # Forward
        logits = model(xb)
        loss = criterion(logits, yb)

        # Backward (only classifier head has requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        # Accumulate loss
        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        # Compute accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc
# ------------------------------------------
# Epoch loop with clean logging (CPU-friendly)
# ------------------------------------------
num_epochs = 5  # keep small (5-10) for ~few minutes on CPU
history = {"train_loss": [], "val_loss": [], "val_acc": []}

print(f"[start] Training head-only for {num_epochs} epochs...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    t0 = time.time()

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Step the scheduler (if used)
    if scheduler is not None:
        scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    dt = time.time() - t0
    print(f"Epoch {epoch:02d} | "
          f"train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | "
          f"val_acc={val_acc:.3f} | "
          f"epoch_time={dt:.1f}s")

total_time = time.time() - start_time
print(f"[done] Finished {num_epochs} epochs in {total_time:.1f}s")
# -------------------------------------------------------------------------------------------
# Run inference and collect predictions
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(yb.numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

test_acc = accuracy_score(y_true, y_pred)
print(f"[test] Overall accuracy: {test_acc:.4f}")
# --------------------------------------------
# Per-class precision/recall/F1 and macro/micro averages
# --------------------------------------------
# Build a label list in the correct index order for the report
target_names = [class_names[i] for i in range(len(class_names))]

report = classification_report(
    y_true, y_pred, target_names=target_names, digits=3, zero_division=0
)
print(report)
# --------------------------------------------
# Confusion matrix (raw and normalized) + heatmaps
# --------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(cm_norm, annot=False, cmap="Greens", vmin=0.0, vmax=1.0,
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix (row-normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
# --------------------------------------------
# Inspect the most confused class pairs (optional)
# --------------------------------------------
# Exclude diagonal to find top confusions (by normalized rate)
cm_offdiag = cm_norm.copy()
np.fill_diagonal(cm_offdiag, 0.0)

# Get top-K confused pairs
K = 5
pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j:
            pairs.append((cm_offdiag[i, j], i, j))
pairs.sort(reverse=True)
top_pairs = pairs[:K]

print("[insight] Top confused class pairs (normalized rates):")
for val, i, j in top_pairs:
    print(f"  True='{class_names[i]}' → Pred='{class_names[j]}' : {val:.3f}")
# --------------------------------------------
# Visualize a few misclassified images (optional)
# --------------------------------------------
# Rebuild a flat list of filepaths in the same order as test_loader iteration
# so we can display misclassified examples
test_paths = []
for paths, _ in [( [p for p,_ in test_loader.dataset.samples], test_loader.dataset.targets )]:
    test_paths = paths

# If the above shortcut is unclear on some torch/torchvision versions,
# we can rebuild from the dataset directly:
if not test_paths:
    test_paths = [p for p, _ in test_loader.dataset.samples]

mis_idx = np.where(y_true != y_pred)[0]
k = min(8, len(mis_idx))
if k > 0:
    sel = np.random.choice(mis_idx, size=k, replace=False)
    plt.figure(figsize=(12, 3))
    for t, idx in enumerate(sel):
        # Recover image path; idx corresponds to dataset order if no shuffling between batches
        # Since DataLoader shuffles only train, test_loader preserves dataset order.
        img_path = test_paths[idx]
        img = Image.open(img_path).convert("RGB")

        plt.subplot(1, k, t+1)
        plt.imshow(img)
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        plt.title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=8)
        plt.axis("off")
    plt.suptitle("Sample misclassifications (T=true, P=pred)", fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("[info] No misclassifications in sampled set (or very small test set).")

# -------------------------------------------------------------------------------------------
#  Code Task 11.2.6.2: Per-class accuracy + Top-3 worst classes
# Reuse predictions from earlier cells or recompute quickly:
CT_true, CT_pred = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        preds = logits.argmax(1).cpu().numpy()
        CT_pred.extend(preds.tolist())
        CT_true.extend(yb.numpy().tolist())
CT_true = np.array(CT_true)
CT_pred = np.array(CT_pred)

CT_cm = confusion_matrix(CT_true, CT_pred, labels=list(range(len(class_names))))
CT_row_sums = CT_cm.sum(axis=1).clip(min=1)
CT_per_class_acc = np.diag(CT_cm) / CT_row_sums  # np.diag(CT_cm) / CT_row_sums

# Build worst-3 list of (name, acc)
CT_order = np.argsort(CT_per_class_acc)[:3]
CT_worst3 = [(class_names[i], float(CT_per_class_acc[i])) for i in CT_order]
print("Worst-3 classes:", CT_worst3)

# -------------------------------------------------------------------------------------------
# Grid of predictions with confidence
# Helper for unnormalizing tensors for display
def unnormalize(img_tensor, mean, std):
    img = img_tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1)

# Get one batch from the test loader
model.eval()
xb, yb = next(iter(test_loader))
xb_dev = xb.to(device)

with torch.no_grad():
    logits = model(xb_dev)
    probs = F.softmax(logits, dim=1).cpu()

pred_idx = probs.argmax(dim=1).numpy()
true_idx = yb.numpy()
conf = probs.max(dim=1).values.numpy()

# Pick up to 12 examples for a compact grid
k = min(12, xb.size(0))
cols = 6
rows = int(np.ceil(k / cols))

plt.figure(figsize=(cols * 2.2, rows * 2.8))

for i in range(k):
    img_disp = unnormalize(xb[i], IMAGENET_MEAN, IMAGENET_STD).permute(1, 2, 0).numpy()
    t_lbl = class_names[true_idx[i]]
    p_lbl = class_names[pred_idx[i]]
    cval = conf[i]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img_disp)
    title = f"T: {t_lbl} P: {p_lbl} ({cval:.2f})"
    plt.title(title, fontsize=9)
    plt.axis("off")

plt.suptitle("Test predictions with confidence (softmax)", fontsize=12)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# Save model and logs
save_dir = Path("./checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

model_path = save_dir / "resnet18_feature_extraction.pth"
torch.save(model.state_dict(), model_path)

logs = {
    "train_loss": history["train_loss"],
    "val_loss": history["val_loss"],
    "val_acc": history["val_acc"]
}

with open(save_dir / "training_logs_feature_extraction.json", "w") as f:
    json.dump(logs, f, indent=4)

print(f"[done] Saved model to: {model_path}")
print(f"[done] Saved training logs to: {save_dir/'training_logs_feature_extraction.json'}")
# -------------------------------------------------------------------------------------------
