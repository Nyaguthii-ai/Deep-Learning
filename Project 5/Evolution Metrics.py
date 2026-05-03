import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Reload Model Data

# -----------------------------
# 1. Device setup
# -----------------------------
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Using device: {device}")

# -----------------------------
# 2. Transforms (same as NB02)
# -----------------------------
#IMG_SIZE = 128
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
])

# -----------------------------
# 3. Load test dataset
# -----------------------------
#data_path = "./data/oxford-iiit-pet"
data_path = "./data"
test_dataset = datasets.OxfordIIITPet(
    root=data_path,
    split="test",
    target_types="category",
    transform=transform,
    download=True
)

# DataLoader
BATCH_SIZE = 32
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test dataset size: {len(test_dataset)} images")
print(f"Number of classes: {len(test_dataset.classes)}")
print(f"Example classes: {test_dataset.classes[:5]}")

# -----------------------------
# 4. Recreate CNN architecture
# -----------------------------
class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3→16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16→32
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 3×128×128 → 16×64×64
        x = self.pool(F.relu(self.conv2(x)))   # 16×64×64 → 32×32×32
        x = x.view(-1, 32 * 32 * 32)           # flatten
        x = F.relu(self.fc1(x))                # → 128
        x = self.fc2(x)                        # → 37 logits
        return x

# Instantiate and load weights
model = PetCNN().to(device)
model.load_state_dict(torch.load("petcnn_best.pth", weights_only=False))
model.eval()

print("Model and weights loaded successfully.")

# --------------------------------------------------------------------------------
# Task: Inspect one sample from the test dataset (should be 3×224×224 after transform)
CT_img, CT_label = test_dataset[0]
print("CT_Sample shape:", CT_img.shape) 
print("CT_Label index:", CT_label)
# --------------------------------------------------------------------------------

# Evaluation 

from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# -----------------------------
# 1. Collect predictions on test set
# -----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1) # Get predicted class indices
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# -----------------------------
# 2. Overall accuracy
# -----------------------------
overall_acc = accuracy_score(all_labels, all_preds) # Compute overall accuracy
print(f"Overall Test Accuracy: {overall_acc:.4f}")

# -----------------------------
# 3. Per-class accuracy
# -----------------------------
class_correct = np.zeros(len(test_dataset.classes))
class_total = np.zeros(len(test_dataset.classes))

for label, pred in zip(all_labels, all_preds):
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1

per_class_acc = class_correct / class_total

# Top 5 best and worst classes
sorted_indices = np.argsort(per_class_acc)
worst_indices = sorted_indices[:5]
best_indices = sorted_indices[-5:][::-1]

print("\nTop 5 Worst-Performing Classes:")
for idx in worst_indices:
    print(f"{test_dataset.classes[idx]}: {per_class_acc[idx]*100:.2f}%")

print("\nTop 5 Best-Performing Classes:")
for idx in best_indices:
    print(f"{test_dataset.classes[idx]}: {per_class_acc[idx]*100:.2f}%")

# -----------------------------
# 4. Classification Report
# -----------------------------
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0)
print("\nClassification Report:\n")
print(report)

# -----------------------------------------------------------------------------------------
# Task: Predict labels from one-mini batch
# Predict labels for the first batch
CT_model = model
CT_model.eval()
with torch.no_grad():
    for CT_X_batch, _ in test_loader:
        CT_X_batch = CT_X_batch.to(device)
        CT_logits = CT_model(CT_X_batch)
        CT_preds = torch.argmax(CT_logits, dim=1)
        break  # Only process the first batch

print("Predicted class indices for the first batch:", CT_preds.cpu().numpy())
# -----------------------------------------------------------------------------------------
# Task: Manual accuracy calculation for the first batch

# Task 4 – Compute accuracy from predictions
CT_preds_np = all_preds
CT_labels_np = all_labels
CT_correct = np.sum(CT_preds_np == CT_labels_np)
CT_accuracy = CT_correct / len(CT_labels_np)
print("CT_Manual Accuracy:", CT_accuracy)
# -----------------------------------------------------------------------------------------
# Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Use class names from test_dataset (same ordering as trainval split)
class_names = test_dataset.classes

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

disp.plot(ax=ax,
          xticks_rotation='vertical',
          cmap='Blues',
          colorbar=True)

plt.title("Confusion Matrix – Oxford-IIIT Pets (37 Classes)")
plt.tight_layout()
plt.show()

# print cm as a tables with labels
import pandas as pd
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion Matrix as DataFrame:\n")
print(cm_df)    

# save as CSV
cm_df.to_csv("confusion_matrix.csv", index=True)
# -----------------------------------------------------------------------------------------
# Task 5 – Identify 5 worst-performing classes
import pandas as pd

# build a row per class rather than wrapping lists inside lists
CT_class_df = pd.DataFrame({
    "Class": class_names,        # list of class names
    "Accuracy": per_class_acc    # array of per-class accuracies
})

CT_sorted_df = CT_class_df.sort_values(by="Accuracy", ascending=True)
CT_worst_5 = CT_sorted_df.head(5)
print(CT_worst_5)

# ------------------------------------------------------------------------------------------
# Visualize misclassified samples from the test set
# Note: Class mappings (breed names) are consistent across train/val/test splits
# so we can safely use test_dataset.classes here.

def show_misclassified_samples(model, loader, classes, n=8):
    """
    Displays n misclassified test images with predicted vs true labels.
    """
    model.eval()
    misclassified = []

    # Gather misclassified samples
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(1).cpu()

            for img, pred, true in zip(images, preds, labels):
                if pred != true:
                    misclassified.append((img, pred.item(), true.item()))
                if len(misclassified) >= n:
                    break
            if len(misclassified) >= n:
                break

    # Display misclassified samples
    fig, axes = plt.subplots(1, n, figsize=(20, 3))
    for i, (img, pred, true) in enumerate(misclassified):
        img = img * 0.5 + 0.5  # unnormalize back to [0,1]
        axes
