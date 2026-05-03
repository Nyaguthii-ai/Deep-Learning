# Load Pretrained ResNet18 & Adapt Classifier
import torch
import torch.nn as nn
from torchvision import models

# -----------------------------
# 1. Load Pretrained ResNet18
# -----------------------------
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# -----------------------------
# 2. Freeze Backbone (Feature Extraction)
# -----------------------------
for param in resnet18.parameters():
    param.requires_grad = False  # freeze all layers

# -----------------------------
# 3. Replace Final Layer (Classifier)
# -----------------------------
num_features = resnet18.fc.in_features  # 512 for ResNet18
resnet18.fc = nn.Linear(num_features, 12)  # 12 plant species
model = resnet18
# -----------------------------
# 4. Verify Model
# -----------------------------
print(resnet18)

# Count trainable parameters (should be small)
trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
print(f"Trainable parameters (classifier head only): {trainable_params:,}")
# -----------------------------------------------------------------------------------

# Task: Build & Freeze AlexNet, Replace Classifier
# 1) Load pretrained AlexNet
CT_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# 2) Freeze ALL params first
for p in CT_model.parameters():
    p.requires_grad = False

# 3) Replace final classifier layer (index 6) with 12 outputs
# AlexNet classifier: [Dropout, Linear(9216,4096), ReLU, Dropout, Linear(4096,4096), ReLU, Linear(4096,1000)]
in_features = CT_model.classifier[6].in_features
CT_model.classifier[6] = nn.Linear(in_features, 12)

# 4) Checks to store
CT_backbone_frozen = all((not p.requires_grad) for n, p in CT_model.named_parameters() if not n.startswith("classifier.6"))
CT_trainable_params = sum(p.numel() for p in CT_model.parameters() if p.requires_grad)

print("CT_backbone_frozen:", CT_backbone_frozen)
print("CT_trainable_params:", CT_trainable_params)
print(CT_model)
# -----------------------------------------------------------------------------------
# Training using ResNet18
import time
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle

# Enable CPU optimizations
torch.set_num_threads(torch.get_num_threads())
if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# -----------------------------
# 1. Loss Function
# -----------------------------
criterion = nn.CrossEntropyLoss()

# -----------------------------
# 2. Optimizer (only classifier head params)
# -----------------------------
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Decays LR by factor of 0.1 every 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -----------------------------
# TRAINING CONTROL - Set to True if you want to train from scratch
# -----------------------------
TRAIN_FROM_SCRATCH = False  # Set to True if you want to train the model yourself

if not TRAIN_FROM_SCRATCH:
    print("🚀 Skipping training to save time and computational resources!")
    print("💡 The model is already pre-trained and ready to use in the next cells.")
    print("🎯 If you want to train from scratch, change TRAIN_FROM_SCRATCH = True above.")
    print("⏱️  Training would take approximately 10 minutes on CPU for only 3-4 epochs!.")
else:
    print("🏃‍♂️ Starting training from scratch... Grab a coffee! ☕")

    # -----------------------------
    # 3. Training Loop
    # -----------------------------
    def train_model_transfer(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=4, device='cpu'):
        """
        Training loop for transfer learning:
        - Only classifier head updates (backbone frozen).
        - Tracks train/validation loss and accuracy.
        """
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
    
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
        model = model.to(device)

    
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
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
    
                    # Stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (outputs.argmax(1) == labels).sum().item()
                    total_samples += labels.size(0)
    
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects / total_samples
    
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)
    
                # Track best model (lowest val loss)
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
    
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
            # Step scheduler after each epoch (if provided)
            scheduler.step()
    
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val loss: {best_loss:.4f}')
    
        # Load best weights
        model.load_state_dict(best_model_wts)
    
        return model, history
    
    # -----------------------------
    # 4. Train the Model
    # -----------------------------
    model, history_transfer = train_model_transfer(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader,
        num_epochs=10, device='cpu'
    )
    
    torch.save(model.state_dict(), "resnet18_best_student.pth")
    
    # Save the training history
    with open('history_transfer_student.pkl', 'wb') as f:
        pickle.dump(history_transfer, f)
    
    print("✅ Model and training history saved!")
# -----------------------------------------------------------------------------------
#Create an Adam optimizer that updates only the new classifier layer (classifier[6]) with lr=1e-3. 
# Add a StepLR with step_size=7, gamma=0.1. 
# 1) Optimizer on the new head ONLY
CT_optimizer = optim.Adam(CT_model.classifier[6].parameters(), lr=0.001 )

# 2) LR scheduler
CT_scheduler = optim.lr_scheduler.StepLR( CT_optimizer, step_size=7, gamma=0.1 )

# 3) How many parameters will be updated?
CT_num_opt_params = sum(p.numel() for p in CT_model.classifier[6].parameters() if p.requires_grad)

print("CT_num_opt_params:", CT_num_opt_params)
# -----------------------------------------------------------------------------------
# Evaluation and comparison 
model.load_state_dict(torch.load("resnet18_best.pth", map_location='cpu'))
model.eval()  # Set to evaluation mode

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the training history
with open('history_transfer.pkl', 'rb') as f:
    history_transfer = pickle.load(f)

print("✅ Training history loaded successfully!")

# -----------------------------
# 1. Helper: Evaluate model
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

    acc = accuracy_score(all_labels, all_preds)
    return acc, np.array(all_labels), np.array(all_preds)

# -----------------------------
# 2. Evaluate on test set
# -----------------------------
test_acc, y_true, y_pred = evaluate_model(model, test_loader, device='cpu')
print(f"ResNet18 Transfer Learning Test Accuracy: {test_acc:.2%}")

# -----------------------------
# 3. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt="d",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes,
            cmap="Blues")
plt.title("Confusion Matrix - ResNet18 Transfer Learning")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -----------------------------
# 4. Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# -----------------------------
# 5. Per-Class Accuracy
# -----------------------------
class_correct = np.diag(cm)
class_totals = cm.sum(axis=1)
class_accuracy = class_correct / class_totals * 100

# Sort by accuracy
sorted_idx = np.argsort(-class_accuracy)
sorted_classes = [test_dataset.classes[i] for i in sorted_idx]
sorted_acc = class_accuracy[sorted_idx]

plt.figure(figsize=(12, 5))
plt.bar(sorted_classes, sorted_acc)
plt.xticks(rotation=45, ha='right')
plt.title("Per-Class Accuracy (Sorted)")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Compare Training Curves
# -----------------------------
#epochs = range(1, len(history_resnet[0]) + 1)
epochs = range(1, len(history_transfer['train_loss']) + 1)

plt.figure(figsize=(12, 4))
# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history_transfer['train_loss'], label="Train Loss")
plt.plot(epochs, history_transfer['val_loss'], label="Val Loss", linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history_transfer['train_acc'], label="Train Acc")
plt.plot(epochs, history_transfer['val_acc'], label="Val Acc", linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()
