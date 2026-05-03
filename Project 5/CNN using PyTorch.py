import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. Define data path and transforms
# -----------------------------
data_path = "data"

# Image size for this notebook
IMG_SIZE = 128

# Transform: Resize → ToTensor → Normalize
# Note: Mean/std values are approximations; for production, compute from dataset
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1,1]
])

# -----------------------------
# 2. Load datasets
# -----------------------------
# Trainval split (to be further split into train/val)
trainval_dataset = datasets.OxfordIIITPet(
    root="./data/",
    split="trainval",
    target_types="category",
    transform=transform,
    download=True
)

# Test set
test_dataset = datasets.OxfordIIITPet(
    root="./data/",
    split="test",
    target_types="category",
    transform=transform,
    download=True
)

# -----------------------------
# 3. Train/Validation Split
# -----------------------------
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

# -----------------------------
# 4. DataLoaders
# -----------------------------
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 5. Inspect dataset properties
# -----------------------------
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")
print(f"Number of classes: {len(trainval_dataset.classes)}")
print(f"Example classes: {trainval_dataset.classes[:5]}")

# -----------------------------
# 6. Visualize a batch
# -----------------------------
def show_batch(loader, classes, n=4):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, n, figsize=(18, 3))
    for i in range(n):
        img = images[i] * 0.5 + 0.5  # unnormalize to [0,1] for display
        axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axes[i].set_title(classes[labels[i]], fontsize=9)
        axes[i].axis("off")
    plt.show()

show_batch(train_loader, trainval_dataset.classes)

# ------------------------------------------------------------------------------------
# Task: Create Subset Datasets and Show a Batch

from torch.utils.data import Subset

# Define index ranges for each subset
CT_train_idx = list(range(40))  # e.g., 40 samples
CT_val_idx = list(range(40, 60))  # e.g., next 20
CT_test_idx = list(range(60, 80))    # e.g., next 20

# Create subsets from original datasets
CT_train_subset = Subset(train_dataset, CT_train_idx)
CT_val_subset = Subset(train_dataset, CT_val_idx)
CT_test_subset = Subset(train_dataset, CT_test_idx)

# Create DataLoaders for each subset
CT_train_loader = DataLoader(CT_train_subset, batch_size=32, shuffle=True)
CT_val_loader = DataLoader(CT_val_subset, batch_size=32, shuffle=False)
CT_test_loader = DataLoader(CT_test_subset, batch_size=32, shuffle=False)

# Show one batch of images from training subset
show_batch(CT_train_loader, trainval_dataset.classes)
# ------------------------------------------------------------------------------------
# Define CNN Model 
# Ensure reproducibility
torch.manual_seed(42)

class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # Input: 3x128x128 → 16x128x128
        self.pool = nn.MaxPool2d(2, 2)                                                   # Downsample → 16x64x64
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # 16x64x64 → 32x64x64
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)   # Flatten: 32x32x32 → 128
        self.fc2 = nn.Linear(128, 37)             # 37 classes (pet breeds)

    def forward(self, x):
        # Input: 3 x 128 x 128
        x = self.pool(F.relu(self.conv1(x)))   # After Conv1+ReLU+Pool → 16 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))   # After Conv2+ReLU+Pool → 32 x 32 x 32
        x = x.view(-1, 32 * 32 * 32)           # Flatten → 32768
        x = F.relu(self.fc1(x))                # FC1 + ReLU → 128
        x = self.fc2(x)                        # FC2 (logits) → 37
        return x

# Instantiate and print model
model = PetCNN()
print(model)

# ----------------------------------------
#Parameter Count Function
def count_parameters_by_type(model):
    # Total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convolutional layers
    conv_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Conv2d) 
                      for p in m.parameters())
    
    # Fully connected (Linear) layers
    fc_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) 
                    for p in m.parameters())
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f" - Convolutional layers: {conv_params:,}")
    print(f" - Fully connected layers: {fc_params:,}")

count_parameters_by_type(model)

# -----------------------------------------------------------------------------
# Alternate sequential model
model_seq = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 37)
)

print(model_seq)
count_parameters_by_type(model_seq)

# -----------------------------------------------------------------------------
# Define Loss and Optimizer
import torch.optim as optim

# Define loss function (CrossEntropyLoss handles Softmax internally)
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Loss Function:", criterion)
print("Optimizer:", optimizer)

# --------------------------------------------------------------------------------
# Training Loop 
import time
import copy

# Accuracy calculation helper
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / len(labels)

# Training function
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

        # Each epoch has train and validation phases
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

            # Iterate over data
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward + optimize in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()()

                # Stats
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

                # Deep copy the model if validation improves
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Time tracking
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val loss: {best_loss:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)


    # Save checkpoint
    torch.save(model.state_dict(), "petcnn_best.pth")

    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

# Example: Train for 5 epochs (CPU-only)
device = 'cpu'  # Force CPU for consistency across all project notebooks
model, history = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5, device=device)

# --------------------------------------------------------------------------------
# Train Your Sequential Model Using train_model()
CT_model_trained, CT_history = train_model(
    model_seq,                    # your model
    criterion,                    # loss function
    optimizer=optim.Adam(model_seq.parameters(), lr=1e-3),  # optimizer
    train_loader=train_loader,       # subset train loader
    val_loader=val_loader,         # subset val loader
    num_epochs=3,         # 3 epochs
    device='cpu'
)
# Plot Accuracy Curve for Your Model
# CT_Task – Plot accuracy curve
CT_train_acc = CT_history[2]  # Assuming history is (train_losses, val_losses, train_accuracies, val_accuracies)
CT_val_acc = CT_history[3]  # Assuming history is (train_losses, val_losses, train_accuracies, val_accuracies)

plt.plot(CT_train_acc, label="Train Accuracy")
plt.plot(CT_val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# --------------------------------------------------------------------------------