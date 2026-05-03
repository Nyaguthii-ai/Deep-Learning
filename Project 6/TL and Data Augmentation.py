import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset  
from collections import defaultdict  
import random  
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import time, copy
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# -----------------------------
# 1. Define ImageNet stats
# -----------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------
# 2. Augmented train transforms
# -----------------------------
train_transform_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # rotate ±20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -----------------------------
# 3. Validation/Test transforms (no aug)
# -----------------------------
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -----------------------------
# 4. Load datasets
# -----------------------------
data_path = "./data/plant-seedlings"
train_dir = os.path.join(data_path, "train")
val_dir   = os.path.join(data_path, "val")
test_dir  = os.path.join(data_path, "test")

# Clean up any .ipynb_checkpoints folders
for split in ['train', 'test', 'val']:
    checkpoint_path = os.path.join(data_path, split, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
        print(f"Removed {checkpoint_path}")

train_dataset_aug = datasets.ImageFolder(train_dir, transform=train_transform_aug)
val_dataset       = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset      = datasets.ImageFolder(test_dir, transform=val_test_transform)

# -----------------------------
# Create 33% stratified subset
# -----------------------------
# Group indices by class
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_dataset_aug):
    class_indices[label].append(idx)

# Sample 33% from each class
subset_indices = []
for class_label, indices in class_indices.items():
    n_samples = len(indices) // 3  # 33% of each class
    sampled_indices = random.sample(indices, n_samples)
    subset_indices.extend(sampled_indices)

# Create subset
train_dataset_aug = Subset(train_dataset_aug, subset_indices)

BATCH_SIZE = 8
train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader       = DataLoader(val_dataset, batch_size=BATCH_SIZE , shuffle=False)
test_loader      = DataLoader(test_dataset, batch_size=BATCH_SIZE , shuffle=False)

print(f"Train (aug) samples: {len(train_dataset_aug)}")
print(f"Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

# -----------------------------
# 5. Visualize augmentations
# -----------------------------
def unnormalize(img_tensor):
    """Undo ImageNet normalization for visualization."""
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img

# Pick one image, show multiple augmented versions
sample_img, sample_label = train_dataset_aug[0]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img_aug, _ = train_dataset_aug[i]  # new augmentations each access
    axes[i].imshow(unnormalize(img_aug))
    axes[i].axis('off')
    axes[i].set_title(f"Aug {i+1}")
plt.suptitle(f"Augmented Variants - Class: {train_dataset_aug.dataset.classes[sample_label]}")
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------
# Create a 25% Stratified Subset of the Augmented v2 Train Set
from collections import defaultdict
from torch.utils.data import Subset
import random, collections

# Group indices by class
CT_class_indices_v2 = defaultdict(list)
for idx, (_, lbl) in enumerate(CT_train_dataset_aug_v2):
    CT_class_indices_v2[lbl].append(idx)

# Sample 25% from each class
CT_subset_indices_v2 = []
for lbl, indices in CT_class_indices_v2.items():
    k = max(1, int(len(indices) * 0.25))   # use 0.25 fraction
    CT_subset_indices_v2.extend(random.sample(indices, k))

# Create subset
CT_train_subset_v2 = Subset(CT_train_dataset_aug_v2, CT_subset_indices_v2)

# Compute ratios vs original
CT_orig_counts_v2 = collections.Counter()
for path, lbl in CT_train_dataset_aug_v2.samples:
    cls = CT_train_dataset_aug_v2.classes[lbl]
    CT_orig_counts_v2[cls] += 1

CT_subset_counts_v2 = collections.Counter()
for idx in CT_subset_indices_v2:
    lbl = CT_train_dataset_aug_v2[idx][1]
    cls = CT_train_dataset_aug_v2.classes[lbl]
    CT_subset_counts_v2[cls] += 1

CT_subset_ratios_v2 = {
    cls: CT_subset_counts_v2.get(cls, 0) / CT_orig_counts_v2[cls]
    for cls in CT_orig_counts_v2.keys()
}

print("CT_subset size:", len(CT_subset_indices_v2))
print("CT_subset_ratios_v2 (per class):", {k: round(v, 3) for k, v in CT_subset_ratios_v2.items()})
# ---------------------------------------------------------------------------------------------------------------------------
# Training with Augmented Data

# Enable CPU optimizations
torch.set_num_threads(torch.get_num_threads())
if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. Load pretrained ResNet18
# -----------------------------
from torchvision import models

model_aug = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model_aug.parameters():
    param.requires_grad = False

# Replace classifier head
num_ftrs = model_aug.fc.in_features
model_aug.fc = nn.Linear(num_ftrs, 12)  # 12 classes
model_aug = model_aug.to(device)

# -----------------------------
# 3. Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_aug.fc.parameters(), lr=0.001)

# Optional scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -----------------------------
# TRAINING CONTROL - Set to True if you want to train from scratch
# -----------------------------
TRAIN_FROM_SCRATCH = True  # Set to True if you want to train the model yourself

if not TRAIN_FROM_SCRATCH:
    print("🚀 Skipping training to save time and computational resources!")
    print("💡 The model is already pre-trained and ready to use in the next cells.")
    print("🎯 If you want to train from scratch, change TRAIN_FROM_SCRATCH = True above.")
    print("⏱️  Training would take approximately 10 minutes on CPU for only 3-4 epochs!.")
else:
    print("🏃‍♂️ Starting training from scratch... Grab a coffee! ☕")


    # -----------------------------
    # 4. Training loop
    # -----------------------------
    def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=5, device=device):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
    
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
    
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}\n{'-'*20}")
    
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    loader = train_loader
                else:
                    model.eval()
                    loader = val_loader
    
                running_loss, running_corrects, total = 0.0, 0, 0
    
                for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f'{phase.capitalize()}', leave=False)):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
    
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # Metrics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
    
                epoch_loss = running_loss / total
                epoch_acc = running_corrects / total
    
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)
                    if scheduler:
                        scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
    
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
        time_elapsed = time.time() - since
        print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best val loss: {best_loss:.4f}")
    
        # Load best weights
        model.load_state_dict(best_model_wts)
    
        return model, (train_losses, val_losses, train_accuracies, val_accuracies)
    
    # -----------------------------
    # 5. Train with augmented data
    # -----------------------------
    model_aug, history_aug = train_model(
        model_aug, criterion, optimizer, scheduler,
        train_loader_aug, val_loader,
        num_epochs=3, device=device
    )
    
    # Save best model
    torch.save(model_aug.state_dict(), "resnet18_aug_student.pth")
# ---------------------------------------------------------------------------------------------------------------------------
# Task: One Epoch of Training
from tqdm import tqdm
import numpy as np

def CT_one_epoch_train_val(model, optimizer, scheduler, train_loader, val_loader, criterion, device='cpu'):
    history = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}

    # ---------- Train ----------
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in tqdm(train_loader, desc="CT_Train (1 epoch)", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
        tr_correct += (logits.argmax(1) == yb).sum().item()
        tr_total += yb.size(0)
    history['train_loss'] = tr_loss / tr_total
    history['train_acc']  = tr_correct / tr_total

    # ---------- Val ----------
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="CT_Val (1 epoch)", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb).item() * xb.size()
            va_loss += loss.item() * xb.size(0)
            va_correct += (logits.argmax(1) == yb).sum().item()
            va_total += yb.size(0)
    history['val_loss'] = va_loss / va_total
    history['val_acc']  = va_correct / va_total

    # Step LR once after the epoch
    if scheduler is not None:
        scheduler.step()

    return history

# Run 1 epoch using augmented v2 (or switch to train_loader_aug)
CT_hist_one_epoch = CT_one_epoch_train_val(
    CT_model_alex, CT_alex_optimizer, CT_alex_scheduler,
    CT_train_loader_aug_v2, val_loader,
    criterion, device=device
)
print("CT_hist_one_epoch:", CT_hist_one_epoch)
# ---------------------------------------------------------------------------------------------------------------------------
# Evaluation
model_aug.load_state_dict(torch.load("resnet18_aug_best.pth", map_location='cpu'))
model_aug.eval()  # Set to evaluation mode

# -----------------------------
# 1. Evaluate Augmented Model
# -----------------------------
def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# Evaluate
labels_true, labels_pred = evaluate_model(model_aug, test_loader, device=device)

# Overall accuracy
acc = (labels_true == labels_pred).mean() * 100
print(f"Test Accuracy (ResNet18 + Aug): {acc:.2f}%")

# Per-class accuracy
class_names = test_dataset.classes
class_correct = {cls: 0 for cls in class_names}
class_total = {cls: 0 for cls in class_names}

for true, pred in zip(labels_true, labels_pred):
    class_total[class_names[true]] += 1
    if true == pred:
        class_correct[class_names[true]] += 1

per_class_acc = {cls: 100 * class_correct[cls] / class_total[cls] for cls in class_names}

# -----------------------------
# 2. Confusion Matrix
# -----------------------------
cm = confusion_matrix(labels_true, labels_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix: ResNet18 + Augmentation")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -----------------------------
# 3. Per-Class Accuracy Bar Chart
# -----------------------------
plt.figure(figsize=(10, 5))
plt.bar(per_class_acc.keys(), per_class_acc.values())
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy: ResNet18 + Augmentation")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Classification Report
# -----------------------------
print("Classification Report:")
print(classification_report(labels_true, labels_pred, target_names=class_names))
# ---------------------------------------------------------------------------------------------------------------------------
# Evaluate CT_model_alex on test_loader. Compute: 
# CT_test_acc_alex (Top-1 accuracy in [0,1])
# CT_top3_acc_alex (Top-3 accuracy)
# CT_cm_alex (confusion matrix, shape [12, 12])
# CT_per_class_acc_alex (dict: class → accuracy %)
# CT_best3_classes_alex (list of top-3 classes by accuracy)

def CT_topk_accuracy(model, loader, k=3, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            topk = torch.topk(logits, k=k, dim=1).indices
            match = topk.eq(yb.unsqueeze(1))
            correct += match.any(dim=1).sum().item()
            total   += yb.size(0)
    return correct / total if total > 0 else 0.0

# Gather predictions
CT_y_true, CT_y_pred = [], []
CT_model_alex.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        logits = CT_model_alex(xb.to(device))
        preds = logits.argmax(1)
        CT_y_true.extend(yb.cpu().numpy())
        CT_y_pred.extend(preds.cpu().numpy())

CT_y_true = np.array(CT_y_true)
CT_y_pred = np.array(CT_y_pred)

# Metrics
CT_test_acc_alex   = (CT_y_true == CT_y_pred).mean()
CT_top3_acc_alex   = CT_topk_accuracy(CT_model_alex, test_loader, k=3, device=device)
CT_cm_alex         = confusion_matrix(CT_y_true, CT_y_pred)

# Per-class accuracy
class_names = test_dataset.classes
diag = np.diag(CT_cm_alex)
tot  = CT_cm_alex.sum(axis=1)
CT_per_class_acc_alex = {
    class_names[i]: float(100.0 * diag[i] / max(tot[i], 1))
    for i in range(len(class_names))
}

# Best 3 classes by accuracy
sorted_idx = np.argsort([-CT_per_class_acc_alex[c] for c in class_names])
CT_best3_classes_alex = [class_names[i] for i in sorted_idx[:3]]

print("CT_test_acc_alex:", CT_test_acc_alex)
print("CT_top3_acc_alex:", CT_top3_acc_alex)
print("CT_best3_classes_alex:", CT_best3_classes_alex)


