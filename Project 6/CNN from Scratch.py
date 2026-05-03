import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset  
from collections import defaultdict  
import random  
import shutil

# Load the dataset and create DataLoaders for training, validation, and testing.
# -----------------------------
# 1. Paths
# -----------------------------
data_path = "./data/plant-seedlings"
train_dir = os.path.join(data_path, "train")
val_dir   = os.path.join(data_path, "val")
test_dir  = os.path.join(data_path, "test")

# -----------------------------
# Clean up any .ipynb_checkpoints folders
# -----------------------------
for split in ['train', 'test', 'val']:
    checkpoint_path = os.path.join(data_path, split, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
        print(f"Removed {checkpoint_path}")

# -----------------------------
# 2. Transformations (224x224 + ImageNet stats)
# -----------------------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


# -----------------------------
# 3. Load datasets
# -----------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)


# -----------------------------
# Create 33% stratified subset 
# -----------------------------
# Group indices by class
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_dataset): # idx is the index of the sample, label is the class index
    class_indices[label].append(idx)

# Sample 33% from each class
subset_indices = []
for class_label, indices in class_indices.items():
    n_samples = len(indices) // 3  # 33% of each class # Round down to nearest integer
    sampled_indices = random.sample(indices, n_samples) # Randomly sample without replacement
    subset_indices.extend(sampled_indices) # Add sampled indices to the subset list

# Create subset
train_dataset = Subset(train_dataset, subset_indices)

# -----------------------------
# 4. DataLoaders
# -----------------------------
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 5. Verify
# -----------------------------
print(f"Classes: {train_dataset.dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ---------------------------------------------------------
# Display sample images (unnormalized for visualization)
# ---------------------------------------------------------
def imshow(inp, title=None):
    #Display a tensor image after unnormalizing (ImageNet stats).
    inp = inp.numpy().transpose((1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
    mean = np.array([0.485, 0.456, 0.406]) # ImageNet mean
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # unnormalize # Scale back to [0, 1] range for display
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')

# Show first 8 images from train set
images, labels = next(iter(train_loader))
fig = plt.figure(figsize=(12, 6))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    imshow(images[i], title=train_dataset.dataset.classes[labels[i]])
plt.show()

# ---------------------------------------------------------
# Task — Verify the 33% Stratified Subset
import collections

#Count per-class in the original full train (ImageFolder behind the Subset)
CT_orig_class_counts = collections.Counter()
for path, lbl in train_dataset.dataset.samples:
    cls_name = train_dataset.dataset.classes[lbl]
    CT_orig_class_counts[cls_name] += 1

# Count per-class in the subset using train_dataset.indices
CT_subset_class_counts = collections.Counter()
for idx in train_dataset.indices:
    lbl = train_dataset.dataset.samples[idx][1]
    cls_name = train_dataset.dataset.classes[lbl]
    CT_subset_class_counts[cls_name] += 1

# Compute per-class ratios
CT_subset_ratios = {cls: CT_subset_class_counts[cls] / CT_orig_class_counts[cls]
                    for cls in train_dataset.dataset.classes}

CT_ratio_min = min(CT_subset_ratios.values())
CT_ratio_max = max(CT_subset_ratios.values())

print("CT_subset_class_counts:", dict(CT_subset_class_counts))
print("CT_orig_class_counts:",   dict(CT_orig_class_counts))
print("CT_ratio_min/max:", CT_ratio_min, CT_ratio_max)

# ---------------------------------------------------------------------------=
# Task: From the stratified subset, sample a small, balanced batch: k=2 examples per class.
# Build CT_balanced_indices (list of indices into the original ImageFolder).
# Wrap with Subset(..., CT_balanced_indices) and create CT_balanced_loader (batch_size = 2 * num_classes).
import random
from torch.utils.data import Subset, DataLoader
from collections import Counter

CT_k = 2
CT_balanced_indices = []

# iterate classes by index
for lbl, cls_name in enumerate(train_dataset.dataset.classes):
    # find indices inside the current subset that belong to this class
    available = [i for i in train_dataset.indices
                 if train_dataset.dataset.samples[i][1] == lbl]
    # sample up to k from available
    chosen = random.sample(available, k=min(CT_k, len(available)))
    CT_balanced_indices.extend(chosen)

# subset referencing the original ImageFolder using the selected indices
CT_balanced_subset = Subset(train_dataset.dataset, CT_balanced_indices)
CT_balanced_loader = DataLoader(CT_balanced_subset, batch_size=CT_k * len(train_dataset.dataset.classes), shuffle=False)

CT_balanced_imgs, CT_balanced_labels = next(iter(CT_balanced_loader))
CT_balanced_label_counts = Counter(CT_balanced_labels.tolist())

print("CT_balanced_label_counts:", dict(CT_balanced_label_counts))
# -------------------------------------------------------------

# Implement custom CNN from scratch (LeNet-style) with 3 conv layers and 2 fully connected layers.

# -----------------------------
# Custom CNN (LeNet-style)
# -----------------------------
class SeedlingsCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(SeedlingsCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling (H,W halved each time)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Flatten after 3 pools
        self.fc2 = nn.Linear(256, num_classes)    # Output logits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3 -> 32, 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 64, 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 64 -> 128, 56 -> 28
        x = x.view(-1, 128 * 28 * 28)         # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # Raw logits
        return x

# Instantiate and print summary-like info
model = SeedlingsCNN(num_classes=12)
print(model)

# Parameter count
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_params(model):,}")
# -------------------------------------------------------------
# Task: Inspect the intermediate shapes in the CNN
CT_xb, CT_yb = next(iter(train_loader))
CT_xb = CT_xb[:4]  # small batch for speed

# forward with taps
CT_conv1 = model.conv1(CT_xb)
CT_act1  = F.relu(CT_conv1)  # apply ReLU to CT_conv1)
CT_pool1 = model.pool(CT_act1)

CT_conv2 = model.conv2(CT_pool1)
CT_act2  = F.relu(CT_conv2)
CT_pool2 = model.pool(CT_act2)

CT_conv3 = model.conv3(CT_pool2)
CT_act3  = F.relu(CT_conv3)
CT_pool3 = model.pool(CT_act3)

CT_shapes = {
    "conv1": CT_conv1.shape,
    "pool1": CT_pool1.shape,
    "conv3": CT_conv3.shape,
    "pool3": CT_pool3.shape
}
print("CT_shapes:", CT_shapes)
# -------------------------------------------------------------
# Training Setup (Loss, Optimizer, Loop)
import time
import copy
import torch.optim as optim
from tqdm import tqdm

# -----------------------------
# 1. Loss function & optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 2. Training loop (CPU-only)
# -----------------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5, device='cpu'):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f'{phase.capitalize()}', leave=False)):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

                # Save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val loss: {best_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    return model, (train_losses, val_losses, train_accuracies, val_accuracies)
# -------------------------------------------------------------
# Quick Validation Pass
def CT_eval_epoch(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb).item() * xb.size(0)  # accumulate total loss
            total_loss += loss
            total_correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    avg_loss = total_loss / total
    accuracy = total_correct / total
    return avg_loss, accuracy