import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

# Load Baseline CNN model

# -----------------------------
# 1. CPU-Only Device Setup
# -----------------------------
device = 'cpu'  # Enforce CPU across all code

# -----------------------------
# 2. Data Transformations
# -----------------------------
IMG_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1,1]
])

# -----------------------------
# 3. Load Train/Val/Test Datasets
# -----------------------------
data_path = "./data/"

# Trainval dataset (for split)
trainval_dataset = datasets.OxfordIIITPet(
    root=data_path,
    split="trainval",
    target_types="category",
    transform=transform,
    download=True
)

# Test dataset (provided separately)
test_dataset = datasets.OxfordIIITPet(
    root=data_path,
    split="test",
    target_types="category",
    transform=transform,
    download=True
)

# -----------------------------
# 4. Train/Val Split (80/20)
# -----------------------------
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size

train_dataset, val_dataset = random_split(
    trainval_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # ensure reproducibility
)

# -----------------------------
# 5. DataLoaders
# -----------------------------
BATCH_SIZE = 8

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Confirm splits
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")
print(f"Number of classes: {len(trainval_dataset.classes)}")

# -----------------------------
# 6. Baseline CNN Definition (same as NB02)
# -----------------------------
class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3x128x128 -> 16x128x128
        self.pool = nn.MaxPool2d(2, 2)                            # -> 16x64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 32x64x64
        self.fc1 = nn.Linear(32 * 32 * 32, 128)                   # -> 128
        self.fc2 = nn.Linear(128, 37)                             # -> 37 (classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))   # -> 32x32x32
        x = x.view(-1, 32 * 32 * 32)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and load baseline weights (map to CPU)
baseline_model = PetCNN().to(device)
baseline_model.load_state_dict(torch.load("petcnn_best.pth", map_location=device, weights_only=False))

print("Baseline CNN loaded successfully (CPU-only). Ready for comparison.")

# -----------------------------------------------------------------------------------------------------
# Create a Small Train/Val Split & DataLoaders
# small custom split for quick CPU experiments
CT_total = len(trainval_dataset)
CT_train_size = int(0.1 * CT_total)  # 800 for training
CT_val_size = CT_total - CT_train_size   # 200 for validation
CT_train_dataset, CT_val_dataset = random_split(
    trainval_dataset,
    [CT_train_size, CT_val_size],
    generator=torch.Generator().manual_seed(123)
)

CT_BATCH_SIZE = 16
CT_train_loader = DataLoader(CT_train_dataset, batch_size=CT_BATCH_SIZE, shuffle=...)
CT_val_loader   = DataLoader(CT_val_dataset,   batch_size=CT_BATCH_SIZE, shuffle=False)

# Peek one batch and record its shape (N, C, H, W)
CT_images, CT_labels = next(iter(CT_train_loader))
CT_batch_shape = CT_images.shape
print("CT_train/val sizes:", len(CT_train_dataset), len(CT_val_dataset))
print("CT_batch_shape:", CT_batch_shape)
# -- -----------------------------------------------------------------------------------------------------

# -------------------------------
# 1. Baseline CNN (for reference)
# -------------------------------
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3x128x128 -> 16x128x128
        self.pool = nn.MaxPool2d(2, 2)                            # Halve H,W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x64x64 -> 32x64x64
        self.fc1 = nn.Linear(32 * 32 * 32, 128)                   # 32x32x32 -> 128
        self.fc2 = nn.Linear(128, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))  # -> 32x32x32
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 2. LeNet-like CNN (3 conv layers)
# -------------------------------
class LeNetLikeCNN(nn.Module):
    def __init__(self):
        super(LeNetLikeCNN, self).__init__()
        # Conv Layer 1: 3 → 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # 3x128x128 -> 16x128x128
        # Conv Layer 2: 16 → 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 16x64x64 -> 32x64x64
        # Conv Layer 3: 32 → 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32x32x32 -> 64x32x32
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 poolings: 128 -> 64 -> 32 -> 16
        # Final feature map: 64 x 16 x 16 = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Fully connected layer
        self.fc2 = nn.Linear(256, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))   # -> 32x32x32
        x = self.pool(F.relu(self.conv3(x)))   # -> 64x16x16
        x = x.view(x.size(0), -1)              # Flatten: 64*16*16 = 16384
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 3. AlexNet-mini CNN (5 conv layers)
# -------------------------------
class AlexNetMini(nn.Module):
    def __init__(self):
        super(AlexNetMini, self).__init__()
        # Conv Layer 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # 3x128x128 -> 32x128x128
        # Conv Layer 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 32x64x64 -> 64x64x64
        # Conv Layer 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 64x32x32 -> 128x32x32
        # Conv Layer 4: 128 → 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128x16x16 -> 128x16x16
        # Conv Layer 5: 128 → 256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128x8x8 -> 256x8x8
        self.pool = nn.MaxPool2d(2, 2)

        # After 5 conv + 5 pool layers:
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 (but here last pooling is applied after conv5)
        # Final feature map: 256 x 4 x 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32x64x64
        x = self.pool(F.relu(self.conv2(x)))   # -> 64x32x32
        x = self.pool(F.relu(self.conv3(x)))   # -> 128x16x16
        x = self.pool(F.relu(self.conv4(x)))   # -> 128x8x8
        x = self.pool(F.relu(self.conv5(x)))   # -> 256x4x4
        x = x.view(x.size(0), -1)              # Flatten: 256*4*4 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# -----------------------------------------------------------------------------------------------   
#Task: Implement a LeNet Variant
class CT_LeNetVariant(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1: 3 -> 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)

        # Conv block 2: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Conv block 3: 32 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # After 3 pools: 128 -> 64 -> 32 -> 16, so flatten = 64*16*16
        self.fc1   = nn.Linear(64 * 16 * 16, 256)
        self.drop  = nn.Dropout(p=0.5)
        self.fc2   = nn.Linear(256, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 16x64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 32x32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> 64x16x16
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)  # logits (N, 37)
        return x

CT_model = CT_LeNetVariant().to(device)
print(CT_model)
# -----------------------------------------------------------------------------------------------------
# Task: Param Count & Forward-Shape Sanity Check
def CT_count_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

CT_model.eval()
with torch.no_grad():
    CT_x, CT_y = next(iter(CT_train_loader))
    CT_x = CT_x.to(device)
    CT_logits = CT_model(CT_x)
    CT_n_params = CT_count_params(CT_model)

print("CT_logits shape:", CT_logits.shape)  # should be (N, 37)
print("CT_n_params:", CT_n_params)
# -----------------------------------------------------------------------------------------------------
# Training loop 
import time
import copy

# Accuracy calculation helper
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / len(labels)

# Training function (CPU-only)
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

        # Train and validation phases
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

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
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
# -----------------------------------------------------------------------------------------------------
# parameter count
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Instantiate models
baseline_model = BaselineCNN()
lenet_model = LeNetLikeCNN()
alexmini_model = AlexNetMini()

# Print parameter counts
print(f"BaselineCNN parameters: {count_params(baseline_model):,}")
print(f"LeNetLikeCNN parameters: {count_params(lenet_model):,}")
print(f"AlexNetMini parameters: {count_params(alexmini_model):,}")

# Define common loss and optimizer settings
criterion = nn.CrossEntropyLoss()

# Train Baseline
print("\n= Training Baseline CNN (we are using pre-trainned models) =")
# Train LeNet-like
print("\n= Training LeNet-like CNN (we are using pre-trainned models) =")
# Train AlexNet-mini
print("\n= Training AlexNet-mini CNN (we are using pre-trainned models) =")

#Output:
#BaselineCNN parameters: 4,204,293
#LeNetLikeCNN parameters: 4,227,653
#AlexNetMini parameters: 2,652,645
# ------------------------------------------------------------------------------------------------------
CT_criterion = nn.CrossEntropyLoss()
CT_optimizer = torch.optim.Adam(CT_model.parameters(), lr=1e-3)

def CT_train_one_epoch(model, loader, criterion, optimizer, device='cpu'):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for CT_inputs, CT_labels in loader:
        CT_inputs, CT_labels = CT_inputs.to(device), CT_labels.to(device)
        optimizer.zero_grad()
        CT_outputs = model(CT_inputs)
        CT_loss = criterion(CT_outputs, CT_labels)
        CT_loss.backward()
        optimizer.step()

        running_loss += CT_loss.item() * CT_inputs.size(0)
        running_corrects += (CT_outputs.argmax(1) == CT_labels).sum().item()
        total += CT_labels.size(0)

    avg_loss = running_loss / total
    avg_acc  = running_corrects / total
    return avg_loss, avg_acc

CT_train_loss1, CT_train_acc1 = CT_train_one_epoch(CT_model, CT_train_loader, CT_criterion, CT_optimizer, device=device)
print("CT_train_loss1:", CT_train_loss1)
print("CT_train_acc1:", CT_train_acc1)
# -------------------------------------------------------------------------------------------------------
# Evaluation Function and Comparison
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reuse helper to compute accuracy
def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

# Evaluate all three models
baseline_acc = evaluate_model(baseline_model, test_loader)
lenet_acc = evaluate_model(lenet_model, test_loader)
alexmini_acc = evaluate_model(alexmini_model, test_loader)

print(f"Baseline CNN Test Accuracy: {baseline_acc:.4f}")
print(f"LeNet-like CNN Test Accuracy: {lenet_acc:.4f}")
print(f"AlexNet-mini CNN Test Accuracy: {alexmini_acc:.4f}")

# Parameter counts
baseline_params = count_params(baseline_model)
lenet_params = count_params(lenet_model)
alexmini_params = count_params(alexmini_model)

# Compare parameter count vs accuracy
print("\nParameter vs Accuracy Comparison:")
print(f"Baseline: {baseline_params:,} params → {baseline_acc:.2%}")
print(f"LeNet-like: {lenet_params:,} params → {lenet_acc:.2%}")
print(f"AlexNet-mini: {alexmini_params:,} params → {alexmini_acc:.2%}")
# -------------------------------------------------------------------------------------------------------
# Quick Validation + Baseline vs Variant Comparison
def CT_evaluate(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for CT_imgs, CT_lbls in loader:
            CT_imgs, CT_lbls = CT_imgs.to(device), CT_lbls.to(device)
            CT_out = model(CT_imgs)
            CT_pred = CT_out.argmax(1)
            correct += (CT_pred == CT_lbls).sum().item()
            total += CT_lbls.size(0)
    return correct / total

CT_baseline_for_eval = BaselineCNN().to(device)   # fresh baseline
CT_params_baseline = CT_count_params(CT_baseline_for_eval)
CT_params_variant  = CT_count_params(CT_model)

CT_val_acc_baseline = CT_evaluate(CT_baseline_for_eval, CT_val_loader, device=device)
CT_val_acc_variant  = CT_evaluate(CT_model, CT_val_loader, device=device)

CT_compare = {
    "baseline_acc": CT_val_acc_baseline,
    "variant_acc": CT_val_acc_variant,
    "params_baseline": CT_params_baseline,
    "params_variant":  CT_params_variant
}
print("CT_compare:", CT_compare)