import time
import copy
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import torch

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
    # 1. Accuracy helper
    # -----------------------------
    def calculate_accuracy(outputs, labels):
        _, preds = torch.max(outputs, 1)
        return (preds == labels).sum().item() / len(labels)
    
    # -----------------------------
    # 2. Training loop (CPU)
    # -----------------------------
    def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=3, device='cpu'):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
    
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
    
        model.to(device)
    
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('-' * 20)
    
            # Train & validation phases
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

        # Time summary
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best val loss: {best_loss:.4f}")
    
        # Load best weights
        model.load_state_dict(best_model_wts)
        return model, (train_losses, val_losses, train_accuracies, val_accuracies)
    
    # -----------------------------
    # 3. Instantiate & train
    # -----------------------------
    device = 'cpu'
    model = SeedlingsCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model, history = train_model(
        model, criterion, optimizer,
        train_loader, val_loader,
        num_epochs=3, device=device
    )
    
    # Save model
    torch.save(model.state_dict(), "seedlings_cnn_student.pth")
    
    # -----------------------------
    # 4. Plot learning curves
    # -----------------------------
    train_losses, val_losses, train_acc, val_acc = history
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # Save the plot before showing it
    plt.savefig('training_curves_student.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.show()


# Load and display the training curves (pre-trained model)
img = mpimg.imread('training_curves.png')
plt.figure(figsize=(12, 4))
plt.imshow(img)
plt.axis('off') 
plt.title('Training Curves (pre-trained model)')
plt.show()
# -------------------------------------------------------------------------------------
# Task — Compute Top-3 Test Accuracy
def CT_topk_accuracy(model, loader, k=3, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            topk_idx = torch.topk(logits, k, dim=1).indices
            match = topk_idx.eq(yb.view(-1, 1)).any(dim=1) # Check if true label is in top-k predictions
            correct += match.sum().item()
            total += yb.size(0)
    return correct / total

CT_top3_acc = CT_topk_accuracy(model, test_loader, k=3, device='cpu')
print("CT_top3_acc:", CT_top3_acc)
# -------------------------------------------------------------------------------------
# Evaluate Overall and Per-Class Accuracy

model = SeedlingsCNN()  
model.load_state_dict(torch.load("seedlings_cnn_best.pth", map_location='cpu'))
model.eval()  # Set to evaluation mode
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# Helper: Evaluate model
# -----------------------------
def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# Evaluate on test set
true_labels, pred_labels = evaluate_model(model, test_loader, device='cpu')

# -----------------------------
# 1. Overall accuracy
# -----------------------------
overall_acc = (true_labels == pred_labels).mean()
print(f"Overall Test Accuracy: {overall_acc:.2%}")

# -----------------------------
# 2. Per-class accuracy
# -----------------------------
class_correct = np.zeros(len(test_dataset.classes))
class_total = np.zeros(len(test_dataset.classes))

for t, p in zip(true_labels, pred_labels):
    if t == p:
        class_correct[t] += 1
    class_total[t] += 1

per_class_acc = class_correct / class_total

# Print top/bottom performing classes
sorted_indices = np.argsort(per_class_acc)
print("\nTop 3 Best Classes:")
for idx in sorted_indices[-3:][::-1]:
    print(f"{test_dataset.classes[idx]}: {per_class_acc[idx]*100:.2f}%")

print("\nTop 3 Worst Classes:")
for idx in sorted_indices[:3]:
    print(f"{test_dataset.classes[idx]}: {per_class_acc[idx]*100:.2f}%")

# -----------------------------
# 3. Confusion Matrix
# -----------------------------
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Plant Seedlings (Custom CNN)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=test_dataset.classes))
# -------------------------------------------------------------------------------------
# -----------------------------
# Per-Class Accuracy Bar Chart
# -----------------------------
plt.figure(figsize=(10, 4))
plt.bar(range(len(per_class_acc)), per_class_acc * 100)
plt.xticks(range(len(per_class_acc)), test_dataset.classes, rotation=45, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy - Custom CNN on Plant Seedlings")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------
# Classification Report (Precision, Recall, F1-Score)
from sklearn.metrics import classification_report

# ---------------------------
# 1. Classification Report
# ---------------------------
# Reload best weights (if saved separately)
model.load_state_dict(torch.load("seedlings_cnn_best.pth", map_location='cpu', weights_only=False))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate report
print("=== Classification Report (Precision/Recall/F1) ===")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
# -------------------------------------------------------------------------------------
def CT_topk_accuracy(model, loader, k=3, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            topk_idx = torch.topk(logits, k, dim=1).indices
            match = topk_idx.eq(yb.view(-1, 1)).any(dim=1)
            correct += match.sum().item()
            total   += yb.size(0)
    return correct / total if total > 0 else 0.0

# Compute on AlexNet (CT_model)
CT_top1_acc = CT_topk_accuracy(CT_model, test_loader, k=1, device='cpu')
CT_top3_acc = CT_topk_accuracy(CT_model, test_loader, k=3, device='cpu')

# Per-class accuracy via confusion matrix
# If evaluate_model(...) exists in the notebook, you may reuse it; otherwise compute labels here.
if 'evaluate_model' in globals():
    _acc_tmp, _y_true_tmp, _y_pred_tmp = evaluate_model(CT_model, test_loader, device='cpu')
    cm = confusion_matrix(_y_true_tmp, _y_pred_tmp)
else:
    y_true_list, y_pred_list = [], []
    CT_model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = CT_model(xb)
            preds = logits.argmax(1)
            y_true_list.extend(yb.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
    cm = confusion_matrix(np.array(y_true_list), np.array(y_pred_list))

class_correct = np.diag(cm)
class_totals  = cm.sum(axis=1)
class_accuracy = (class_correct / np.maximum(class_totals, 1)) * 100.0

CT_per_class_acc = {
    test_dataset.classes[i]: class_accuracy[i]
    for i in range(len(test_dataset.classes))
}

sorted_idx = np.argsort(-class_accuracy)
CT_best3_classes = [ test_dataset.classes[i] for i in sorted_idx[:3] ]

print("CT_top1_acc:", CT_top1_acc)
print("CT_top3_acc:", CT_top3_acc)
print("CT_best3_classes:", CT_best3_classes)


