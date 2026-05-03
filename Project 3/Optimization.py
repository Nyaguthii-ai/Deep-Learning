#Load and prepare dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
df = pd.read_csv("yeast.csv")  # Adjust path if needed

# Separate features and target
X = df.drop("localization_site", axis=1)
y = df["localization_site"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check shapes and class info
print("✅ Train set:", X_train_tensor.shape)
print("✅ Test set:", X_test_tensor.shape)
print("✅ Number of Classes:", len(class_names))
print("✅ Class Names:", class_names)

#----------------------------------------------------------------------------------
# Task 1 – Simple class count summary
CT_class_counts = torch.bincount(y_train_tensor)
print("CT_Class Distribution:")
for i, count in enumerate(CT_class_counts):
    print(f"{class_names[i]}: {count.item()}")

#using zip instead of enumerate to print class names and counts together
CT_class_counts = torch.bincount(y_train_tensor)
print("CT_Class Distribution:")
for i, c in zip(class_names,CT_class_counts):
    print(f"{class_names}: {c.item()}")
#----------------------------------------------------------------------------------
# Model Definition (MLP Architecture)

import torch.nn as nn

# Define the MLP class
class YeastMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(YeastMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model
input_dim = X_train_tensor.shape[1]      # 8 features
hidden_dim = 16
output_dim = len(class_names)            # 10 classes

model = YeastMLP(input_dim, hidden_dim, output_dim)

# Display model structure
print(model)

#----------------------------------------------------------------------------------
# Task 2 - Implement full batch Gradient Descent (GD) training loop
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Set seed for reproducibility
torch.manual_seed(42)

# Model from earlier section (assumed defined)
model_gd = YeastMLP(input_dim, hidden_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_gd.parameters(), lr=0.1)

# Store loss and accuracy per epoch
train_loss_gd = []
train_acc_gd = []

# Training loop - full batch GD
n_epochs = 100

start_time = time.perf_counter()

for epoch in range(n_epochs):
    # Set model to training mode
    model_gd.train()

    # Forward pass on the whole training set
    logits = model_gd(X_train_tensor)
    loss = loss_fn(logits, y_train_tensor)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward() # Manually compute gradients
    optimizer.step() # Update parameters

    # Store loss
    train_loss_gd.append(loss.item())

    # Compute accuracy
    preds = torch.argmax(logits, dim=1)
    acc = (preds == y_train_tensor).float().mean().item()
    train_acc_gd.append(acc)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>2} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")

end_time = time.perf_counter()
print(f"Training completed in {end_time - start_time:.2f} seconds")

#--------------------------------------------------------------------------------------------
# Task 3 - Implement Stochastic Gradient Descent (SGD) training loop
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seed for reproducibility
torch.manual_seed(42)

# Create a new model instance (same as before)
model_sgd = YeastMLP(input_dim=8, hidden_dim=16, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_sgd.parameters(), lr=0.01)

# Track metrics
sgd_loss_values = []
sgd_accuracy_values = []

# Convert train set into list of samples
train_samples = list(zip(X_train_tensor, y_train_tensor))

start_time = time.perf_counter()

# Training loop: SGD
for epoch in range(100):
    epoch_loss = 0
    correct = 0

    # Shuffle training samples
    torch.random.manual_seed(epoch)  # Ensures reproducibility across epochs
    shuffled_samples = train_samples.copy()
    random.shuffle(shuffled_samples)  

    for x_sample, y_sample in shuffled_samples:
        x_sample = x_sample.unsqueeze(0)  # add batch dim
        y_sample = y_sample.unsqueeze(0)

        # Forward pass
        outputs = model_sgd(x_sample)
        loss = criterion(outputs, y_sample)

        # Backprop + Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_sample).sum().item()

    avg_loss = epoch_loss / len(train_samples)
    accuracy = correct / len(train_samples)

    sgd_loss_values.append(avg_loss)
    sgd_accuracy_values.append(accuracy)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
        
end_time = time.perf_counter()
print(f"Training completed in {end_time - start_time:.2f} seconds")

#--------------------------------------------------------------------------------------------
# Task 4 - Implement Mini-batch Gradient Descent (MBGD) training loop
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define hyperparameters
batch_size = 32
epochs = 100
learning_rate = 0.01

# Re-initialize model and optimizer
model_mb = YeastMLP(input_dim, hidden_dim, output_dim)
optimizer_mb = optim.SGD(model_mb.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create DataLoader for mini-batch SGD
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Store metrics
losses_mb = []
accs_mb = []

start_time = time.perf_counter()

# Training loop
for epoch in range(epochs):
    model_mb.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model_mb(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer_mb.zero_grad()
        loss.backward()
        optimizer_mb.step()

        # Track loss
        epoch_loss += loss.item() * X_batch.size(0)

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = epoch_loss / total
    acc = correct / total

    losses_mb.append(avg_loss)
    accs_mb.append(acc)

    print(f"[Mini-Batch] Epoch {epoch+1:2d}: Loss = {avg_loss:.4f} | Accuracy = {acc:.4f}")

end_time = time.perf_counter()
print(f"Training completed in {end_time - start_time:.2f} seconds")

#--------------------------------------------------------------------------------------------
# Task 5 - Evaluation on test set

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, class_names, title):
    # Put model in eval mode
    model.eval()
    
    # Disable gradient tracking
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)  # get predicted class indices

    # Convert to numpy for sklearn
    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy ({title}): {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix – {title}")
    plt.show()

# Evaluate all models
evaluate_model(model_gd, X_test_tensor, y_test_tensor, class_names, "Gradient Descent (Batch)")
evaluate_model(model_sgd, X_test_tensor, y_test_tensor, class_names, "Stochastic Gradient Descent (Per Sample)")
evaluate_model(model_mb, X_test_tensor, y_test_tensor, class_names, "Mini-Batch SGD")

#------------------------------------------------------
# Looking at evaluation on a smaller scale

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

CT_model_gd.eval()
with torch.no_grad():
    CT_outputs = CT_model_gd(X_test_tensor)
    _, CT_preds = torch.max(CT_outputs, 1)

from sklearn.metrics import accuracy_score
CT_test_acc = accuracy_score(y_test_tensor.cpu().numpy(), CT_preds.cpu().numpy())

print(f"CT_Test Accuracy (GD): {CT_test_acc:.4f}")
