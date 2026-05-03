# Load datasets for CIFAR-10 in a CNN model

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

# Load CIFAR-10 to compute mean and std
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=50000, shuffle=False)

# Compute mean and std
images, _ = next(iter(loader))
mean = images.mean(dim=[0, 2, 3])
std = images.std(dim=[0, 2, 3])
print("Mean:", mean)
print("Std:", std)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10
train_val_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

# Class names
class_names = train_val_dataset.classes
print(f"✅ Class names: {class_names}")

# Split training into train/val
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Sanity check
print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Validation batches: {len(val_loader)}")
print(f"✅ Test batches: {len(test_loader)}")

# Peek at shape
for images, labels in train_loader:
    print(f"✅ Image batch shape: {images.shape}")  # Expected: [64, 3, 32, 32]
    print(f"✅ Label batch shape: {labels.shape}")  # Expected: [64]
    break

# ----------------------------------------------------------------------------------
# PyTorch Implementation of LeNet for CIFAR-10
import torch.nn as nn
import torch.nn.functional as F

class LeNet_CIFAR10(nn.Module):
    def __init__(self):
        super(LeNet_CIFAR10, self).__init__()
        
        # Convolutional Block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),     # (3,32,32) → (6,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                        # → (6,14,14)
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),    # → (16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                         # → (16,5,5)
        )
        
        # Fully Connected Classifier
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                    # → 16*5*5 = 400
            nn.Linear(400, 120),             
            nn.ReLU(),
            nn.Linear(120, 84),              
            nn.ReLU(),
            nn.Linear(84, 10)                # 10 output logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
# ----------------------------------------------------------------------------------  
# Task – View conv output shape
CT_sample_images, _ = next(iter(train_loader))
CT_sample_images = CT_sample_images.to(device)

CT_lenet = LeNet_CIFAR10().to(device)
CT_conv_out = CT_lenet.conv_layers(CT_sample_images)
print(f"✅ Conv output shape: {CT_conv_out.shape}")  # Expected: [64, 16, 5, 5]
# -----------------------------------------------------------------------------------

#Train & Evaluate LeNet on CIFAR-10
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Instantiate model, loss, optimizer
lenet_model = LeNet_CIFAR10().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters(), lr=0.001)

# For tracking metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

epochs = 20

for epoch in range(epochs):
    lenet_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = lenet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    lenet_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = lenet_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {epoch_acc:.4f}, Val Acc: {val_correct / val_total:.4f}")

# Save the model
torch.save(lenet_model.state_dict(), "lenet_cifar10.pth")

# Test Set Evaluation and compare with the best optimzer model 

# Load the saved Lenet model
lenet_model_loaded = LeNet_CIFAR10().to(device)
lenet_model_loaded.load_state_dict(torch.load("lenet_cifar10.pth"))

# Create the class for MLP model from NB03 

class MLP_CIFAR10(nn.Module):
    def __init__(self, input_dim=3072, output_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim)  # output logits
        )

    def forward(self, x):
        return self.model(x)

# Load model
mlp_model = MLP_CIFAR10().to(device)
mlp_model.load_state_dict(torch.load("model_bestopt_NB03.pth"))
mlp_model.eval()

#Define Evaluation Function
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, name="Model"):
 
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, MLP_CIFAR10):  # Only apply to MLP model
                images = images.view(images.size(0), -1)  # Flatten to [batch_size, 3072]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"🔍 {name} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels

# Evaluate both models
print("Evaluating MLP Model:")
mlp_preds, mlp_labels = evaluate_model(mlp_model, test_loader, name="MLP")

print("\nEvaluating LeNet Model:")
lenet_preds, lenet_labels = evaluate_model(lenet_model, test_loader, name="LeNet")

#Confusion Matrix Comparison
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# MLP Confusion Matrix
cm_mlp = confusion_matrix(mlp_labels, mlp_preds)
disp_mlp = ConfusionMatrixDisplay(cm_mlp, display_labels=class_names)
disp_mlp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix — MLP (NB03)")
plt.grid(False)
plt.show()

# LeNet Confusion Matrix
cm_lenet = confusion_matrix(lenet_labels, lenet_preds)
disp_lenet = ConfusionMatrixDisplay(cm_lenet, display_labels=class_names)
disp_lenet.plot(cmap='Greens', xticks_rotation=45)
plt.title("Confusion Matrix — LeNet (NB04)")
plt.grid(False)
plt.show()

# --------------------------------------------------------------------------------------
# CT_Task 6.1 – Compare MLP and LeNet test accuracy
from sklearn.metrics import accuracy_score

CT_mlp_preds, CT_mlp_labels = evaluate_model(mlp_model, test_loader, name="CT_MLP")
CT_lenet_preds, CT_lenet_labels = evaluate_model(lenet_model, test_loader, name="CT_LeNet")

CT_mlp_acc = accuracy_score(CT_mlp_labels, CT_mlp_preds)
CT_lenet_acc = accuracy_score(CT_lenet_labels, CT_lenet_preds)
# ----------------------------------------------------------------------------------------

#PyTorch Implementation: AlexNet Mini
import torch.nn as nn

class AlexNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # [B, 3, 32, 32] → [B, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [B, 64, 32, 32] → [B, 64, 16, 16]
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 64, 16, 16] → [B, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → [B, 128, 8, 8]
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # → [B, 256, 8, 8]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # → [B, 256, 8, 8]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # → [B, 256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → [B, 256, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: [B, 256, 4, 4] → [B, 4096]
        x = self.classifier(x)
        return x

# Instantiate model
alexnet_model = AlexNetMini().to(device)
print(alexnet_model)

#Training & Validation of AlexNetMini
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model, loss, optimizer
alexnet_model = AlexNetMini().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet_model.parameters(), lr=0.001)

# Track metrics
epochs = 1 # Do not change this because it is just to illustrate how long it takes to train just one epoch without a GPU.
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    # Training Phase
    alexnet_model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = alexnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation Phase
    alexnet_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = alexnet_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_correct / val_total:.4f}")

# AlexNetMini-Lite model
import torch.nn as nn

class AlexNetMiniLite(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [B, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
#Training and validation of AlexNetMiniLite
# Instantiate model
alexnet_lite = AlexNetMiniLite().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet_lite.parameters(), lr=0.001)

# Tracking metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

epochs = 20

for epoch in range(epochs):
    alexnet_lite.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet_lite(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    alexnet_lite.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = alexnet_lite(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

#Evaluate AlexNetMiniLite on test data
print("Evaluating AlexNetMiniLite Model:")
alexnet_lite_preds, alexnet_lite_labels = evaluate_model(alexnet_lite, test_loader, name="AlexNetMiniLite")

#Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_alexnet_lite = confusion_matrix(alexnet_lite_labels, alexnet_lite_preds)
disp_alexnet_lite = ConfusionMatrixDisplay(confusion_matrix=cm_alexnet_lite, display_labels=class_names)
disp_alexnet_lite.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix — AlexNetMiniLite")
plt.grid(False)
plt.show()

#---------------------------------------------------------------------------------------
# CT_Task 7.1 – Inspect AlexNet conv feature map shape
CT_batch_images, _ = next(iter(train_loader))
CT_batch_images = CT_batch_images.to(device)

CT_alexnet = AlexNetMini().to(device)
CT_feature_output = CT_alexnet.features(CT_batch_images)
print(f"✅ AlexNet conv feature map shape: {CT_feature_output.shape}")  # Expected: [64, 256, 4, 4]