# Load dataset 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# 1. Load the yeast dataset (already cleaned and merged in NB01)
df = pd.read_csv("yeast.csv")  # Replace with your file path if needed

# 2. Separate features and target
X = df.drop(columns=["localization_site"])
y = df["localization_site"]

# 3. Encode target labels as integers (e.g., MIT → 0, NUC → 1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6. Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)

# 7. Check dimensions
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Number of classes:", len(label_encoder.classes_))

#-----------------------------------------------------------------------------------
#Building a simple MLP
import torch.nn as nn

# Define model class
class CT_SimpleMLP(nn.Module):
    def __init__(self):
        super(CT_SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model
CT_model = CT_SimpleMLP()
print(CT_model)

#------------------------------------------------------------------------------------
# Define loss function and optimizer
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 200
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    # Forward pass
    logits = model(X_train)
    loss = criterion(logits, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record loss
    train_losses.append(loss.item())
    
    # Compute training accuracy
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_train).sum().item()
    accuracy = correct / len(y_train)
    train_accuracies.append(accuracy)
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy*100:.2f}%")

#---------------------------------------------------------------------------------------------
# Evaluate on test set, track accuracy and losses
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----- Predict on test data -----
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_labels = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

# ----- Convert ground truth to numpy -----
y_test_np = y_test.cpu().numpy()

# ----- Accuracy -----
acc = accuracy_score(y_test_np, y_pred_labels)
print(f"Test Accuracy: {acc:.4f}")

# ----- Confusion Matrix -----
cm = confusion_matrix(y_test_np, y_pred_labels)

# Assuming you used LabelEncoder before
from sklearn.preprocessing import LabelEncoder

# Re-create encoder and fit to original labels if needed
encoder = LabelEncoder()
encoder.fit(df['localization_site'])  # Ensure same order as before
class_names = encoder.classes_

# Display with class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.show()
