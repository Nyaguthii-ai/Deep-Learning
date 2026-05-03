import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset

# Set seed for reproducibility
torch.manual_seed(42)

# 1. Define transforms: ToTensor + Normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts from [0, 255] → [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standardize to [-1, 1]
])

# 2. Load CIFAR-10 dataset
trainval_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 3. Flatten each image from [3, 32, 32] → [3072]
def flatten_dataset(dataset):
    images = torch.stack([data[0] for data in dataset])  # shape: [N, 3, 32, 32]
    labels = torch.tensor([data[1] for data in dataset])  # shape: [N]
    images_flat = images.view(images.shape[0], -1)       # shape: [N, 3072]
    return TensorDataset(images_flat, labels)

trainval_dataset = flatten_dataset(trainval_set)
test_dataset = flatten_dataset(test_set)

# 4. Split into 80% train and 20% validation
train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

# 5. Create DataLoaders for training, validation, and testing
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 6. Get class names (for evaluation/visualization later)
class_names = trainval_set.classes
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

print(f"Train loader batches: {len(train_loader)}")
print(f"Validation loader batches: {len(val_loader)}")
print(f"Test loader batches: {len(test_loader)}")

print("Class labels:", class_names)

# ----------------------------------------------------------------------------------
# Introducing Batch Normalization.

#Define MLP with Batch Normalization
class MLP_BatchNorm(nn.Module):
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

# Instantiate model and optimizer
model_bn = MLP_BatchNorm().to(device)

#Re-train the model with batch normalization
torch.manual_seed(42)  # For reproducibility
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_bn.parameters(), lr=0.01)

# Tracking
train_losses_bn, val_losses_bn = [], []
train_accs_bn, val_accs_bn = [], []

epochs = 20

for epoch in range(epochs):
    model_bn.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_bn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_losses_bn.append(running_loss / total)
    train_accs_bn.append(correct / total)

    # Validation
    model_bn.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_bn(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses_bn.append(val_loss / val_total)
    val_accs_bn.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Acc: {train_accs_bn[-1]:.4f}, Val Acc: {val_accs_bn[-1]:.4f}")
# ----------------------------------------------------------------------------------
# Task – Count BatchNorm layers
CT_bn_layers = [m for m in model_bn.modules() if isinstance(m, nn.BatchNorm1d)]
print("CT_Number of BatchNorm layers:", len(CT_bn_layers))

# Print details of the first BatchNorm layer
if len(CT_bn_layers) > 0:
    print("CT_First BatchNorm layer:", CT_bn_layers[0])
# ----------------------------------------------------------------------------------

#Training with all Optimizers
import torch
import torch.nn as nn
import torch.optim as optim

def train_model_with_optimizer(optimizer_name="SGD", epochs=10, seed=42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize model
    model = MLP_BatchNorm().to(device)
    criterion = nn.CrossEntropyLoss()

    # Choose optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "SGD_momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer")

    # Tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss_epoch, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(train_loss_epoch / total)
        train_accuracies.append(correct / total)

        # Validation phase
        model.eval()
        val_loss_epoch, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_epoch += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss_epoch / val_total)
        val_accuracies.append(val_correct / val_total)

        # Print progress
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs} - "
              f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies, model

# Train with all optimizers
eval_opt_results = {}
for opt in ["SGD", "SGD_momentum", "Adam", "RMSProp"]:
    print(f"\n🚀 Training with {opt}")
    eval_opt_results[opt] = train_model_with_optimizer(optimizer_name=opt, epochs=20)
# -----------------------------------------------------------------------------------
# Task 5.1 – Compare optimizer validation accuracies
for opt_name, results in eval_opt_results.items():
    CT_val_acc_final = results[3][-1]  # Last epoch's validation accuracy
    print(f"CT_Optimizer: {opt_name} | Final Val Accuracy: {CT_val_acc_final:.4f}")
# Output
#CT_Optimizer: SGD | Final Val Accuracy: 0.5188
#CT_Optimizer: SGD_momentum | Final Val Accuracy: 0.5284
#CT_Optimizer: Adam | Final Val Accuracy: 0.5443
#CT_Optimizer: RMSProp | Final Val Accuracy: 0.5340

# ------------------------------------------------------------------------------------
# Including Dropout and L2 Regularization in the MLP architecture
import torch.nn as nn

class MLP_With_Dropout(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Train with Adam and L2 regularization(weight decay)
def train_dropout_l2_model(epochs=10, weight_decay=1e-4):
    torch.manual_seed(42)  # For reproducibility

    model = MLP_With_Dropout().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / total)
        train_acc.append(correct / total)

        # Validation phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss.append(val_running_loss / val_total)
        val_acc.append(val_correct / val_total)

        print(f"[Dropout+L2] Epoch {epoch+1}/{epochs} - "
              f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

    return train_loss, val_loss, train_acc, val_acc, model

# Train the model with dropout and L2 regularization
results_dropout_l2 = train_dropout_l2_model(epochs=20)

#Final Model Evaluation on Test Set
_, _, _, _, model_bestopt = train_model_with_optimizer("Adam", epochs=20) #Best optimizer based on validation accuracy
_, _, _, _, model_reg = train_dropout_l2_model(epochs=20) #Model with dropout and L2 regularization

from sklearn.metrics import classification_report, accuracy_score, f1_score
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation function
def evaluate_model(model, dataloader, name="Model"):
    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    duration = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"🔍 {name} | Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Inference Time: {duration:.2f}s")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return acc, f1, duration

# Use previously trained models
model_baseline = eval_opt_results["SGD"][4]                # Baseline: SGD without momentum
model_bn = eval_opt_results["SGD_momentum"][4]             # With BatchNorm + SGD+momentum
model_bestopt = eval_opt_results["Adam"][4]                # With BatchNorm + Adam
model_reg = results_dropout_l2[4]                 # With BN + Adam + Dropout + L2

# Evaluate each model
eval_results = []
models_to_evaluate = [
    ("Baseline (SGD)", model_baseline),
    ("BN (SGD+Momentum)", model_bn),
    ("BN + Adam", model_bestopt),
    ("BN + Adam + Dropout + L2", model_reg)
]

for name, model in models_to_evaluate:
    acc, f1, duration = evaluate_model(model, test_loader, name)
    eval_results.append((name, acc, f1, duration))

# Convert to DataFrame
df_results = pd.DataFrame(eval_results, columns=["Model", "Accuracy", "Macro F1", "Time (s)"])
print(df_results)

# ------------------------------------------------------------------------------------
# Task – Evaluate final regularized model on test data
CT_acc_final, CT_f1_final, CT_time_final = evaluate_model(model_reg, test_loader, name="Final Reg Model")

print(f"CT_Final Test Accuracy: {CT_acc_final:.4f}")
print(f"CT_Final Macro F1 Score: {CT_f1_final:.4f}")

# ------------------------------------------------------------------------------------
