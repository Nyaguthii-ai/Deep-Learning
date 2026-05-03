import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from torchvision.utils import make_grid
from matplotlib import cm
from torch.utils.data import DataLoader
# -----------------------------------------------------------------------------------------------
# Reload Models and Inspect Parameters
# === Utility Function to Count Parameters ===
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# === Load base ResNet and get feature size ===
base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = base_model.fc.in_features

# --- 1) Pretrained baseline: "pure reuse" (we don't train it in our project) ---
model_pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Keep original 1000-class head or (if we prefer) swap to 10-class; either way:
for p in model_pretrained.parameters():
    p.requires_grad = False    # conceptually: 0 trainable params in our project

# --- 2) Feature-extracted model (NB02): backbone frozen, head trainable ---
model_feature = models.resnet18(weights=None)
model_feature.fc = nn.Linear(num_features, 10)
model_feature.load_state_dict(
    torch.load("./checkpoints/resnet18_feature_extraction.pth", weights_only=False, map_location="cpu")
)

# Freeze everything first
for p in model_feature.parameters():
    p.requires_grad = False
# Unfreeze only classifier head (this is what we *trained* in NB02)
for p in model_feature.fc.parameters():
    p.requires_grad = True

# --- 3) Fine-tuned model (NB03): layer4 + head trainable ---
model_finetuned = models.resnet18(weights=None)
model_finetuned.fc = nn.Linear(num_features, 10)
model_finetuned.load_state_dict(
    torch.load("./checkpoints/resnet18_finetuned.pth", weights_only=False, map_location="cpu")
)

# Freeze all layers first
for p in model_finetuned.parameters():
    p.requires_grad = False
# Unfreeze only layer4 and head (this is what we *trained* in NB03)
for p in model_finetuned.layer4.parameters():
    p.requires_grad = True
for p in model_finetuned.fc.parameters():
    p.requires_grad = True

# === Inspect which parameters are trainable ===
def print_trainable_layers(model, name):
    trainable_layers = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nModel: {name}")
    print(f"Trainable layers: {len(trainable_layers)} out of {len(list(model.parameters()))}")
    print("Example trainable params:",
          trainable_layers[-5:] if trainable_layers else "None")

print_trainable_layers(model_pretrained, "Pretrained (ImageNet)")
print_trainable_layers(model_feature, "Feature-Extracted (NB02)")
print_trainable_layers(model_finetuned, "Fine-Tuned (NB03)")

# === Count parameters ===
for name, mdl in zip(
    ["Pretrained", "Feature-Extracted", "Fine-Tuned"],
    [model_pretrained, model_feature, model_finetuned]
):
    total, trainable = count_params(mdl)
    print(f"{name:18s} | Total: {total/1e6:.2f}M | "
          f"Trainable: {trainable/1e6:.4f}M | "
          f"Frozen: {(total-trainable)/1e6:.4f}M")
# -----------------------------------------------------------------------------------------------
# CT_Task 1 – Reload fine-tuned CT model and inspect parameters

# 1. Recreate ResNet18 backbone (same as in NB03)
CT_model_finetuned_CT_NB03 = models.resnet18(weights=None)  # architecture only

# 2. Replace final fully-connected layer to match Caltech-101 classes
CT_num_classes = 10
CT_model_finetuned_CT_NB03.fc = nn.Linear(
num_features, CT_num_classes
)

# 3. Load saved weights from NB03
CT_checkpoint_path = "./checkpoints/resnet18_finetuned.pth"
CT_state_dict = torch.load(CT_checkpoint_path, weights_only=False, map_location="cpu")
CT_model_finetuned_CT_NB03.load_state_dict(CT_state_dict)

# 4. Reapply fine-tuning pattern: freeze all, unfreeze layer3+layer4+fc
for p in CT_model_finetuned_CT_NB03.parameters():
    p.requires_grad = False

for p in CT_model_finetuned_CT_NB03.layer3.parameters():
    p.requires_grad = True

for p in CT_model_finetuned_CT_NB03.layer4.parameters():
    p.requires_grad = True

for p in CT_model_finetuned_CT_NB03.fc.parameters():
    p.requires_grad = True

# 5. Compute parameter counts
CT_total_params = sum(p.numel() for p in CT_model_finetuned_CT_NB03.parameters())
CT_trainable_params = sum(p.numel() for p in CT_model_finetuned_CT_NB03.parameters() if p.requires_grad)
CT_frozen_params = CT_total_params - CT_trainable_params
# -----------------------------------------------------------------------------------------------
# Load test split with the saved mapping (consistency check)
# Paths (must match earlier notebooks)
DATA_ROOT = Path("./data/caltech101_10")
MAP_PATH  = Path("./checkpoints/class_to_idx.pth")

assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"
assert (DATA_ROOT / "test").exists(), f"Test split missing at {DATA_ROOT/'test'}"
assert MAP_PATH.exists(), f"class_to_idx mapping not found at {MAP_PATH}"

# Load saved mapping from NB01
class_to_idx_saved = torch.load(MAP_PATH, weights_only=False)
idx_to_class_saved = {v: k for k, v in class_to_idx_saved.items()}
class_names = [idx_to_class_saved[i] for i in range(len(idx_to_class_saved))]
num_classes = len(class_names)
print(f"[info] Saved classes ({num_classes}): {class_names}")

# Preprocessing (must match training notebooks)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
eval_tfms = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Build test dataset/loader
test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=eval_tfms)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"[info] Test set size: {len(test_ds)} images")
print(f"[info] Test class_to_idx (from disk): {test_ds.class_to_idx}")

# Verify mapping consistency (order and names)
assert set(test_ds.class_to_idx.keys()) == set(class_to_idx_saved.keys()), \
    "Mismatch between saved classes and test folder classes."

# Build ordered list according to saved mapping (0..K-1)
ordered_from_saved = [name for name, _ in sorted(class_to_idx_saved.items(), key=lambda kv: kv[1])]
ordered_from_disk  = [name for name, _ in sorted(test_ds.class_to_idx.items(), key=lambda kv: kv[1])]
print(f"[check] Order (saved): {ordered_from_saved}")
print(f"[check] Order (disk) : {ordered_from_disk}")

assert ordered_from_saved == ordered_from_disk, \
    "Index order differs between saved mapping and current dataset. Please align folder names or mapping."

print("[ok] Class index order is consistent with saved mapping.")
# ----------------------------------------------------
# CT_Task 2 – Show 4 sample "airplanes" images from the test set
# ----------------------------------------------------
class_names = test_ds.classes
CT_airplane_class_name = "airplanes"
CT_airplane_class_idx = class_names.index(CT_airplane_class_name)

CT_airplane_indices = [i for i, (_, label) in enumerate(test_ds) if label == CT_airplane_class_idx]
CT_airplane_indices = CT_airplane_indices[:4]

CT_airplane_fig, CT_airplane_axes = plt.subplots(1, 4, figsize=(12, 3))

for ax, idx in zip(CT_airplane_axes, CT_airplane_indices):
    img, label = test_ds[idx]
    img_np = img.numpy().transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    ax.imshow(img_np)
    ax.set_title(class_names[label])
    ax.axis("off")

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# Load the three models robustly (with fallbacks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
CKPT = Path("./checkpoints")
CKPT_FE = CKPT / "resnet18_feature_extraction.pth"
CKPT_FT = CKPT / "resnet18_finetuned.pth"
CKPT_PRE = CKPT / "resnet18_pretrained.pth"          # optional (if you created this)
CKPT_INIT = CKPT / "resnet18_frozen_init.pth"        # saved in NB01

num_classes = len(class_names)

def make_resnet18(num_classes: int, weights="IMAGENET1K_V1"):
    m = models.resnet18(weights=getattr(models.ResNet18_Weights, weights) if isinstance(weights, str) else weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m

# --- Pretrained baseline: prefer user-saved ckpt; else NB01 init; else random head on ImageNet backbone
if CKPT_PRE.exists():
    model_pretrained = make_resnet18(num_classes, "IMAGENET1K_V1")
    model_pretrained.load_state_dict(torch.load(CKPT_PRE, weights_only=False, map_location="cpu"), strict=True)
    baseline_name = "pretrained.pth"
elif CKPT_INIT.exists():
    model_pretrained = make_resnet18(num_classes, "IMAGENET1K_V1")
    model_pretrained.load_state_dict(torch.load(CKPT_INIT, weights_only=False, map_location="cpu"), strict=True)
    baseline_name = "frozen_init.pth"
else:
    model_pretrainemodel_pretrained = make_resnet18(num_classes, "IMAGENET1K_V1")  # random 10-class head
    baseline_name = "ImageNet backbone + random 10-cls head"

model_pretrained = model_pretrained.to(device).eval()

# --- Feature extraction model (NB02)
assert CKPT_FE.exists(), f"Missing NB02 checkpoint: {CKPT_FE}"
model_feature = make_resnet18(num_classes, "IMAGENET1K_V1")
model_feature.load_state_dict(torch.load(CKPT_FE, weights_only=False, map_location="cpu"), strict=True)
# emulate FE inference setup (freezing not required for eval, but explicit)
for p in model_feature.parameters(): p.requires_grad = False
model_feature = model_feature.to(device).eval()

# --- Fine-tuned model (NB03)
assert CKPT_FT.exists(), f"Missing NB03 checkpoint: {CKPT_FT}"
model_finetuned = make_resnet18(num_classes, "IMAGENET1K_V1")
model_finetuned.load_state_dict(torch.load(CKPT_FT, weights_only=False, map_location="cpu"), strict=True)
model_finetuned = model_finetuned.to(device).eval()

print("[ok] Loaded models:")
print(f"  - Baseline: {baseline_name}")
print(f"  - Feature-extracted: {CKPT_FE.name}")
print(f"  - Fine-tuned:        {CKPT_FT.name}")
# ---------------------------------------------
# Inference helpers and batch predictions
# ---------------------------------------------
@torch.no_grad()
def predict_batch(model, xb):
    logits = model(xb.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return pred_idx, conf

# get one batch
xb, yb = next(iter(test_loader))
xb_dev = xb.to(device)

pred_pre, conf_pre = predict_batch(model_pretrained, xb)
pred_fe,  conf_fe  = predict_batch(model_feature,   xb)
pred_ft,  conf_ft  = predict_batch(model_finetuned, xb)

y_true = yb.numpy()
# ---------------------------------------------
# Side-by-side visualization grid
# ---------------------------------------------
def unnormalize(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    t = img_tensor.clone()
    for c,(m,s) in enumerate(zip(mean,std)):
        t[c] = t[c]*s + m
    return t.clamp(0,1)

k = min(12, xb.size(0))
cols = 3  # show 3 columns per row (we'll stack rows)
rows = k

plt.figure(figsize=(10, rows * 2.0))
for i in range(k):
    img = unnormalize(xb[i]).permute(1,2,0).numpy()
    t_lbl  = class_names[y_true[i]]
    p_pre  = class_names[pred_pre[i]]
    p_fe   = class_names[pred_fe[i]]
    p_ft   = class_names[pred_ft[i]]

    # one row per image with 1 big panel
    ax = plt.subplot(rows, 1, i+1)
    ax.imshow(img)
    title = (
        f"T: {t_lbl} | "
        f"PRE: {p_pre} ({conf_pre[i]:.2f}) | "
        f"FE: {p_fe} ({conf_fe[i]:.2f}) | "
        f"FT: {p_ft} ({conf_ft[i]:.2f})"
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.suptitle("Batch predictions — True vs Pretrained (PRE), Feature-Extracted (FE), Fine-Tuned (FT)", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.98])
plt.show()
# -----------------------------------------------
# Compact tabular comparison for the same batch
# -----------------------------------------------
rows = []
for i in range(k):
    rows.append({
        "true": class_names[y_true[i]],
        "PRE_pred": class_names[pred_pre[i]],
        "PRE_conf": f"{conf_pre[i]:.2f}",
        "FE_pred":  class_names[pred_fe[i]],
        "FE_conf":  f"{conf_fe[i]:.2f}",
        "FT_pred":  class_names[pred_ft[i]],
        "FT_conf":  f"{conf_ft[i]:.2f}",
    })

pd.DataFrame(rows)
# -----------------------------------------------------------------------------------------------
# Compute metrics for all three models
def evaluate_model(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "name": name,
        "acc": acc,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weight,
        "cm": cm
    }

results = []
for name, model in [
    ("Pretrained", model_pretrained),
    ("Feature-Extracted", model_feature),
    ("Fine-Tuned", model_finetuned)
]:
    print(f"Evaluating {name}...")
    res = evaluate_model(model, test_loader, name)
    results.append(res)
    print(f"{name} done.")
# -----------------------------------------    
# Display comparative metrics table
# -----------------------------------------
df_metrics = pd.DataFrame([{
    "Model": r["name"],
    "Accuracy": f"{r['acc']*100:.2f}%",
    "F1 (Macro)": f"{r['f1_macro']*100:.2f}%",
    "F1 (Weighted)": f"{r['f1_weighted']*100:.2f}%"
} for r in results])

display(df_metrics)
# -----------------------------------------    
# Visualize confusion matrices side-by-side
# -----------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(7, 16))
for ax, r in zip(axes, results):
    cm = r["cm"]
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
    ax.set_title(r["name"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticklabels(class_names, rotation=45)
plt.suptitle("Confusion Matrices — Pretrained vs Feature-Extracted vs Fine-Tuned", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# -------------------------------------------------------------
#  Code Task 11.4.5.1 – Test Accuracy & F1 (Macro / Weighted)
# -------------------------------------------------------------
# CT_Task 3 – Compute Accuracy, Macro F1, Weighted F1 on test set

CT_model_finetuned_CT_NB03.eval()
CT_all_preds = []
CT_all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to("cpu"), labels.to("cpu")
        outputs = CT_model_finetuned_CT_NB03(inputs)
        preds = outputs.argmax(dim=1)
        CT_all_preds.extend(preds.numpy())
        CT_all_labels.extend(labels.numpy())

CT_all_preds = np.array(CT_all_preds)
CT_all_labels = np.array(CT_all_labels)

CT_test_acc = accuracy_score(CT_all_labels, CT_all_preds)
CT_test_f1_macro = f1_score(CT_all_labels, CT_all_preds, average="macro")
CT_test_f1_weighted = f1_score(CT_all_labels, CT_all_preds, average="weighted")
# -----------------------------------------------------------------------------------------------
# Capture layer4 activations for one image
# 1) Get a single sample from test set
xb, yb = next(iter(test_loader))
img = xb[0:1].to(device)     # shape [1,3,H,W]
true_idx = yb[0].item()
true_cls = class_names[true_idx]

# 2) Hook to capture layer4 activations
def capture_layer4_activations(model, x):
    acts = {}
    handle = model.layer4.register_forward_hook(lambda m, i, o: acts.setdefault("a", o.detach().cpu()))
    with torch.no_grad():
        _ = model(x)
    handle.remove()
    # acts["a"] ⇒ [1, C, H, W] → mean over C ⇒ [H, W]
    a = acts["a"][0].mean(dim=0)
    # normalize to 0..1 for visualization
    a = a - a.min()
    a = a / a.max().clamp(min=1e-8)
    return a.numpy()

act_pre = capture_layer4_activations(model_pretrained, img)
act_fe  = capture_layer4_activations(model_feature,   img)
act_ft  = capture_layer4_activations(model_finetuned, img)

# 3) Show original image and activation maps
def unnormalize_for_vis(x, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    x = x[0].detach().cpu().clone()
    for c,(m,s) in enumerate(zip(mean,std)):
        x[c] = x[c]*s + m
    return x.clamp(0,1).permute(1,2,0).numpy()

orig = unnormalize_for_vis(img)

plt.figure(figsize=(10,3))
plt.subplot(1,4,1)
plt.imshow(orig)
plt.title(f"Original\nTrue: {true_cls}")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(act_pre, cmap="viridis")
plt.title("Pretrained\nlayer4 (mean)")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(act_fe, cmap="viridis")
plt.title("Feature-Extracted\nlayer4 (mean)")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(act_ft, cmap="viridis")
plt.title("Fine-Tuned\nlayer4 (mean)")
plt.axis("off")

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# Optional: Grad-CAM overlays for the same image (all three models)
# --- Grad-CAM helper ---

def gradcam(model, x, target_layer):
    """
    Compute Grad-CAM heatmap [H, W] in 0..1 for the model's predicted class.
    Works for pretrained, feature-extraction, and fine-tuned models.
    """
    model.eval()
    acts, grads = {}, {}

    # Forward hook: save activations of target_layer
    def fwd_hook(_m, _i, o):
        acts["a"] = o

    # Backward hook: save gradient w.r.t. those activations
    def bwd_hook(_m, grad_in, grad_out):
        grads["g"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    # Use a fresh input that tracks gradients
    x_local = x.clone().detach().requires_grad_(True)

    # Forward pass
    logits = model(x_local)                   # [1, num_classes]
    pred_idx = logits.argmax(dim=1)[0].item() # scalar int

    # Backprop wrt predicted class logit
    score = logits[0, pred_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    # Remove hooks
    h1.remove()
    h2.remove()

    # Activations & grads: [C, H, W]
    A = acts["a"][0]
    G = grads["g"][0]

    # Channel weights via global average pooling over spatial dims
    weights = G.mean(dim=(1, 2), keepdim=True)   # [C,1,1]
    cam = (weights * A).sum(dim=0)              # [H,W]

    # Standard Grad-CAM: ReLU + normalize to [0,1]
    cam = torch.relu(cam).detach().cpu()
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)

    return cam.numpy(), pred_idx

# --- Overlay utility with CAM upsampling ---

def overlay_cam(rgb, cam, alpha=0.35, cmap="jet"):
    """
    rgb: (H, W, 3) in [0,1]
    cam: (h, w) in [0,1] from Grad-CAM
    """
    H, W = rgb.shape[:2]
    # Resize CAM to image size if needed
    if cam.shape != (H, W):
        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        cam_t = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam_t.squeeze().cpu().numpy()  # [H,W]

    heat = cm.get_cmap(cmap)(cam)[..., :3]   # RGBA -> RGB
    return (1 - alpha) * rgb + alpha * heat

# --- Compute Grad-CAM for each model at layer4 ---

cam_pre, p_pre = gradcam(model_pretrained, img, model_pretrained.layer4)
cam_fe,  p_fe  = gradcam(model_feature,   img, model_feature.layer4)
cam_ft,  p_ft  = gradcam(model_finetuned, img, model_finetuned.layer4)

# rgb is the original unnormalized image: [H, W, 3] in [0,1]
rgb   = orig
ov_pre = overlay_cam(rgb, cam_pre)
ov_fe  = overlay_cam(rgb, cam_fe)
ov_ft  = overlay_cam(rgb, cam_ft)

# --- Plot ---

plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(ov_pre)
plt.title(f"PRE Grad-CAM\npred={class_names[p_pre]}")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(ov_fe)
plt.title(f"FE Grad-CAM\npred={class_names[p_fe]}")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(ov_ft)
plt.title(f"FT Grad-CAM\npred={class_names[p_ft]}")
plt.axis("off")

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# CT_Task 4 – Layer4 mean activation map for CT_model_finetuned_CT_NB03

CT_model_finetuned_CT_NB03.eval()

CT_activations_layer4_CT = []

def CT_hook_layer4_CT(module, input, output):
    CT_activations_layer4_CT.append(output.detach())

# 1. Register hook
CT_handle_CT = CT_model_finetuned_CT_NB03.layer4.register_forward_hook(CT_hook_layer4_CT)

# 2. Select a sample image (same as used earlier if possible)
CT_idx_sample = 0
CT_img_sample, CT_label_sample = test_ds[CT_idx_sample]
CT_input_sample = CT_img_sample.unsqueeze(0)  # add batch dim

with torch.no_grad():
    _ = CT_model_finetuned_CT_NB03(CT_input_sample)

# 3. Extract feature map and compute mean over channels
CT_feat_layer4_CT = CT_activations_layer4_CT[0].squeeze(0)  # C x H x W
CT_layer4_mean_CT = CT_feat_layer4_CT.mean(dim=0)           # H x W

# 4. Remove hook
CT_handle_CT.remove()

# 5. Plot original vs mean activation
CT_fig_layer4_CT, CT_axes_layer4_CT = plt.subplots(1, 2, figsize=(8, 4))

# Original image (unnormalize)
CT_img_np = CT_img_sample.numpy().transpose(1, 2, 0)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
CT_img_np = std * CT_img_np + mean
CT_img_np = np.clip(CT_img_np, 0, 1)

CT_axes_layer4_CT[0].imshow(CT_img_np)
CT_axes_layer4_CT[0].set_title("Original")
CT_axes_layer4_CT[0].axis("off")
# ---------------------------------------------------
# ✅ Code Task 11.4.6.2 – Grad-CAM Overlay with CT Model
# ---------------------------------------------------
CT_cam_ft,  CT_p_ft  = gradcam(CT_model_finetuned_CT_NB03, img, CT_model_finetuned_CT_NB03.layer4)

# rgb is the original unnormalized image: [H, W, 3] in [0,1]
rgb   = orig
CT_ov_ft = overlay_cam(rgb, CT_cam_ft)

# --- Plot ---

plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(CT_ov_ft)
plt.title(f"FT Grad-CAM\npred={class_names[CT_p_ft]}")
plt.axis("off")

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------
# Load zero-shot mini-set 
ZS_ROOT = Path("./data/caltech101_10_zeroshot")

zs_loader = None
if ZS_ROOT.exists():
    zs_ds = datasets.ImageFolder(ZS_ROOT, transform=eval_tfms)
    zs_loader = DataLoader(zs_ds, batch_size=12, shuffle=False, num_workers=0)
    zs_classnames = sorted(list({c for c, _ in zs_ds.class_to_idx.items()}))
    print(f"[info] Zero-shot mini-set found: {ZS_ROOT}")
    print(f"[info] Zero-shot folders (treated as unseen categories): {zs_classnames}")
    print(f"[info] Zero-shot size: {len(zs_ds)}")
else:
    print("[note] No zero-shot mini-set found at ./data/caltech101_10_zeroshot")
    print("      Create it as ./data/caltech101_10_zeroshot/<new_class>/*.jpg and re-run this section.")
# ------------------------------------------------
# Helper: top-k predictions and a compact table
# ------------------------------------------------
def topk_predictions(model, xb, k=3):
    with torch.no_grad():
        logits = model(xb.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        topk_idx = np.argsort(-probs, axis=1)[:, :k]
        topk_prob = np.take_along_axis(probs, topk_idx, axis=1)
    return topk_idx, topk_prob

def summarize_topk_rows(img_indices, true_folder_names, topk_idx, topk_prob, label_names, prefix):
    rows = []
    for i, gidx in enumerate(img_indices):
        preds = [label_names[j] for j in topk_idx[i]]
        probs = [f"{p:.2f}" for p in topk_prob[i]]
        rows.append({
            "img_id": int(gidx),
            "unseen_folder": true_folder_names[i],
            f"{prefix}_top1": preds[0],
            f"{prefix}_p1": probs[0],
            f"{prefix}_top2": preds[1] if len(preds) > 1 else "",
            f"{prefix}_p2": probs[1] if len(probs) > 1 else "",
            f"{prefix}_top3": preds[2] if len(preds) > 2 else "",
            f"{prefix}_p3": probs[2] if len(probs) > 2 else "",
        })
    return rows
# ------------------------------------------------
# Run top-3 predictions across all three models (if zero-shot exists)
# ------------------------------------------------
import itertools

if zs_loader is not None:
    xb_zs, yb_zs = next(iter(zs_loader))  # use first small batch
    # Map folder indices to folder names for display (these are NOT our trained labels)
    zs_idx_to_folder = {v: k for k, v in zs_ds.class_to_idx.items()}
    zs_folder_names = [zs_idx_to_folder[i] for i in yb_zs.numpy().tolist()]

    topk_pre_idx, topk_pre_prob = topk_predictions(model_pretrained, xb_zs, k=3)
    topk_fe_idx,  topk_fe_prob  = topk_predictions(model_feature,   xb_zs, k=3)
    topk_ft_idx,  topk_ft_prob  = topk_predictions(model_finetuned, xb_zs, k=3)

    rows = []
    rows += summarize_topk_rows(range(len(xb_zs)), zs_folder_names, topk_pre_idx, topk_pre_prob, class_names, "PRE")
    rows_fe = summarize_topk_rows(range(len(xb_zs)), zs_folder_names, topk_fe_idx, topk_fe_prob, class_names, "FE")
    rows_ft = summarize_topk_rows(range(len(xb_zs)), zs_folder_names, topk_ft_idx, topk_ft_prob, class_names, "FT")

    # Merge per-image dicts side-by-side
    merged = []
    for r_pre, r_fe, r_ft in zip(rows, rows_fe, rows_ft):
        merged.append({**r_pre, **{k:v for k,v in r_fe.items() if k not in ("img_id","unseen_folder")},
                               **{k:v for k,v in r_ft.items() if k not in ("img_id","unseen_folder")}})

    df_topk = pd.DataFrame(merged)
    display(df_topk)
else:
    print("[skip] Zero-shot mini-set not present; skipping this table.")
# ------------------------------------------------
# Visualize a few zero-shot images with top-1 per model
# ------------------------------------------------
def unnormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img_tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1)

if zs_loader is not None:
    k = min(8, xb_zs.size(0))
    plt.figure(figsize=(12, 8))
    for i in range(k):
        img_disp = unnormalize(xb_zs[i]).permute(1, 2, 0).numpy()
        pre_top1 = class_names[topk_pre_idx[i,0]]
        fe_top1  = class_names[topk_fe_idx[i,0]]
        ft_top1  = class_names[topk_ft_idx[i,0]]

        # plot in subplot of 2 rows x 4 cols
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_disp)
        title = (f"{zs_folder_names[i]}\n"
                 f"PRE:{pre_top1} FE:{fe_top1} FT:{ft_top1}")
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.suptitle("Zero-shot images: folder (unseen) vs model Top-1 predictions", fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("[skip] Zero-shot mini-set not present; skipping visualization.")
# -----------------------------------------------------------------------------------------------