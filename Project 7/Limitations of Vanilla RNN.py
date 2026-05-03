from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Code Task: New train DataLoader
# -----------------------------------------------------------------------------------------------------
# CT_Task 1 — New train DataLoader with a different batch size

CT_BATCH_SIZE_TRAIN = 48   # e.g., 48
CT_train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),   # X_train_t, y_train_t
    batch_size= 48,
    shuffle=True,
    drop_last=False
)

# sanity check one batch
CT_xb, CT_yb = next(iter(CT_train_loader))
print("CT_train batch shapes:", CT_xb.shape, CT_yb.shape)
# -----------------------------------------------------------------------------------------------------
# CT_Task 2 — Create RNN and load saved weights (hidden_size must match)

CT_model_loaded = RNNClassifier(
    input_size= 9,
    hidden_size= 32,     # must be 32 to load rnn_har.pth safely
    num_classes=6,
    num_layers=1,
    nonlinearity="tanh"
)
CT_state = torch.load("rnn_har.pth", map_location="cpu", weights_only=False)
CT_model_loaded.load_state_dict(CT_state)
CT_model_loaded.eval()

# quick forward to confirm shapes
CT_xb, CT_yb = next(iter(CT_train_loader))
with torch.no_grad():
    CT_logits, CT_out_seq, CT_hn = CT_model_loaded(CT_xb)
print("CT_out_seq shape:", CT_out_seq.shape)
# -----------------------------------------------------------------------------------------------------
# Recreate model (same as Section 2)
model = RNNClassifier(input_size=9, hidden_size=32, num_classes=6, num_layers=1, nonlinearity="tanh")
model.load_state_dict(torch.load("rnn_har.pth", map_location="cpu", weights_only=False))
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Take one mini-batch
xb, yb = next(iter(train_loader))

# Forward pass
logits, out_seq, h_n = model(xb)
loss = criterion(logits, yb)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Collect gradient norms
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

grad_norms
# -----------------------------------------------------------------------------------------------------
# Visualize gradient magnitudes

plt.figure(figsize=(8,4))
plt.bar(range(len(grad_norms)), list(grad_norms.values()), tick_label=list(grad_norms.keys()))
plt.ylabel("Gradient L2 norm")
plt.title("Gradient norms after one backward pass")
plt.xticks(rotation=45, ha="right")
plt.show()
# -----------------------------------------------------------------------------------------------------
# Measure how much the final loss "pulls" on each input timestep
model.train()

xb, yb = next(iter(train_loader))
xb = xb.clone().detach().requires_grad_(True)  # (B, 128, 9) — enable grad on inputs

logits, out_seq, h_n = model(xb)               # loss depends ONLY on last hidden via logits
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, yb)

model.zero_grad()
loss.backward()                                 # single backward pass from t=128

# Gradients wrt inputs: shape (B, T, F). Reduce over batch & channels → (T,)
import torch
grad_x = xb.grad.detach()                       # (B, 128, 9)
grad_x_norm_per_t = torch.linalg.vector_norm(grad_x, dim=(0, 2))  # (T,)

print("First 10 timestep input-grad norms:", grad_x_norm_per_t[:10])

import matplotlib.pyplot as plt
T = grad_x_norm_per_t.numel()
plt.figure(figsize=(8,4))
plt.plot(range(1, T+1), grad_x_norm_per_t.numpy(), marker="o")
plt.xlabel("Timestep t (1 … 128)")
plt.ylabel(r"Mean $\|\partial \mathcal{L} / \partial x_t\|$ (L2)")
plt.title("Vanishing gradient: sensitivity of loss to inputs across time")
plt.grid(True, alpha=0.3)
plt.show()
# -----------------------------------------------------------------------------------------------------
# Hidden-state convergence diagnostic: mean ||h_t - h_{t-1}|| over batch
model.eval()
with torch.no_grad():
    xb_eval, _ = next(iter(test_loader))         # (B, T, F)
    logits_eval, out_seq_eval, _ = model(xb_eval)  # out_seq_eval: (B, T, H)

# Step-to-step hidden change: (B, T-1, H) -> mean L2 over batch
delta = out_seq_eval[:, 1:, :] - out_seq_eval[:, :-1, :]
delta_norm_t = torch.linalg.vector_norm(delta, dim=2).mean(dim=0)  # (T-1,)

plt.figure(figsize=(8,4))
plt.plot(range(2, T+1), delta_norm_t.numpy(), marker="o")
plt.axvline(x=(T//4), color="k", ls="--", alpha=0.5, label="early window")
plt.xlabel("Timestep t")
plt.ylabel(r"Mean $\|h_t - h_{t-1}\|$ (L2)")
plt.title("Hidden-state convergence diagnostic")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Quantify early stagnation relative to the distribution of changes
th = float(0.05 * torch.median(delta_norm_t).item()) if delta_norm_t.numel() else 1e-3
early_plateau_frac = float((delta_norm_t[:(T//4 - 1)] < th).float().mean().item()) if T > 4 else 0.0
print(f"Fraction of early timesteps with near-zero hidden change: {early_plateau_frac:.2f} (threshold ~ {th:.2e})")
# -----------------------------------------------------------------------------------------------------
# CT_Task 4 — Vanishing gradient plot with new config

# one mini-batch from CT_train_loader
CT_xb_grad, CT_yb_grad = next(iter(CT_train_loader))
CT_xb_grad = CT_xb_grad.clone().detach().requires_grad_(True)

CT_model_loaded.train()
CT_logits_g, CT_out_seq_g, CT_hn_g = CT_model_loaded(CT_xb_grad)
CT_loss_g = nn.CrossEntropyLoss()(CT_logits_g, CT_yb_grad)

CT_model_loaded.zero_grad()
CT_loss_g.backward()

# mean L2 norm over features, then mean over batch -> (T,)
CT_grad_x = CT_xb_grad.grad.detach()          # (B, T, F)
CT_grad_x_norm_per_t = torch.linalg.vector_norm(CT_grad_x, dim=2).mean(dim=0)  # fill axis

# dynamic threshold vs last 8 steps
T = CT_grad_x_norm_per_t.numel()
CT_late_ref = CT_grad_x_norm_per_t[-8:].mean().item()
CT_vanish_thresh = float(0.01 * CT_late_ref) if CT_late_ref > 0 else 1e-4

plt.figure(figsize=(8,4))
plt.plot(range(1, T+1), CT_grad_x_norm_per_t.numpy(), marker="o", label="||dL/dx_t||")
plt.axhline(CT_vanish_thresh, ls="--", label=f"threshold≈{CT_vanish_thresh:.2e}")
plt.xlabel("Timestep t")
plt.ylabel("Mean L2 grad norm")
plt.title("Vanishing gradient (new batch sizes)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------------
# CT_Task 5 — Hidden-state convergence with new config

CT_model_loaded.eval()
with torch.no_grad():
    CT_xb_eval, _ = next(iter(CT_train_loader))
    CT_logits_e, CT_out_seq_e, _ = CT_model_loaded(CT_xb_eval)   # (B, T, H)

# (B, T-1, H)
CT_delta = CT_out_seq_e[:, 1:, :] - CT_out_seq_e[:, :-1, :]
CT_delta_norm_t = torch.linalg.vector_norm(CT_delta, dim=2).mean(dim=0)  # fill axis

plt.figure(figsize=(8,4))
plt.plot(range(2, 128+1), CT_delta_norm_t.numpy(), marker="o")
plt.axvline(x=(128//4), color="k", ls="--", alpha=0.5, label="early window")
plt.xlabel("Timestep t")
plt.ylabel("Mean ||h_t - h_{t-1}||")
plt.title("Hidden-state convergence (new batch sizes)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

# early plateau fraction
CT_th = float(0.05 * torch.median(CT_delta_norm_t).item()) if CT_delta_norm_t.numel() else 1e-3
CT_early_plateau_frac = float((CT_delta_norm_t[:(128//4 - 1)] < CT_th).float().mean().item())
print(f"CT_early_plateau_frac: {CT_early_plateau_frac:.2f}")
# -----------------------------------------------------------------------------------------------------
#Quick overall test accuracy
correct, total = 0, 0
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        logits, _, _ = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)

test_acc = correct / total
print(f"Overall Test Accuracy: {test_acc:.3f}")
# -----------------------------------------------------------------------------------------------------
# Assumes: model, test_loader, device, ACTIVITY_MAP already defined in §2
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _, _ = model(xb)             # (B, 6)
        preds = logits.argmax(dim=1)         # (B,)
        all_true.append(yb.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

y_true = np.concatenate(all_true)            # shape (N_test,)
y_pred = np.concatenate(all_pred)            # shape (N_test,)

print("Collected predictions:", y_true.shape, y_pred.shape)
# -----------------------------------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIVITY_MAP))))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

def plot_confusion_matrix(matrix, labels, title, normalize=False):
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if matrix[i,j] > matrix.max()*0.6 else "black")
    plt.tight_layout()
    plt.show()

labels = [ACTIVITY_MAP[i] for i in range(len(ACTIVITY_MAP))]

plot_confusion_matrix(cm, labels, title="Confusion Matrix — Counts", normalize=False)
plot_confusion_matrix(cm_norm, labels, title="Confusion Matrix — Row-Normalized", normalize=True)