# Exercise 1: Dataset Preparation
import pandas as pd

# Load data
df = pd.read_csv("heart.csv")
print(df.shape)
print(df.head())

#check for missing values 
df.info()
df.isnull().sum()

#drop rows with missing values
df = df.dropna()
print("Remaining rows:", df.shape[0])

#simplify target variable to binary classification
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df['target'].value_counts()

# use : df.columns[:-1].tolist() # List of column names (excluding target)

#use all 13 input columns (excluding the original num),and predict the new target.
X = df.drop(columns=['num', 'target'])
y = df['target']

#Normalize the input features - standardizing input features (mean = 0, std = 1).
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Let’s split the dataset into: 80% training 20% testing
from sklearn.model_selection import train_test_split
import torch

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

#-----------------------------------------------------------------------------------
#Exercise 2: Define FNN with nn.sequential
import torch.nn as nn

# Define the model
model = nn.Sequential(

    nn.Linear (13, 32), # Input layer → second Hidden layer (13 → 32)
    nn.LeakyReLU(),     # Activation
    nn.Linear (32, 16), # Second Hidden layer → Hidden layer (32 → 16)
    nn.Sigmoid(),
    nn.Linear (16, 1),  # Hidden → Output layer (16 → 1)
    nn.Sigmoid(),       # Output activation for binary classification
)

# Show architecture
print(model)
sum(p.numel() for p in model.parameters()) #total trainable parameters

#loss function and optimizer
loss_fn = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Loss function:", loss_fn)
print("Optimizer:", optimizer)

#-----------------------------------------------------------------------------------
#Exercise 3: Training loop
# Number of epochs
num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save loss
    train_losses.append(loss.item())

    # TODO: Evaluate on test set
    # Store test loss in test_losses
    model.eval() 
    
    with torch.no_grad():
    # Forward pass (testing)
        outputs_test = model(X_test)
        loss_test = loss_fn(outputs_test, y_test)

    # Save training loss
    test_losses.append(loss_test.item())
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss.item():.4f}, "
              f"Test Loss: {loss_test.item():.4f}")  # 👈 Replace with actual test loss
        
#Plot training loss
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, linestyle='--', marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.grid(True)
plt.show()

# predicting based on train set
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train)
    y_train_pred = (y_train_pred > 0.5).float()

# confusion matrix
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score
import seaborn as sns
import numpy as np

confusion_matrix(y_train, y_train_pred)

# Plot confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on train Set")
plt.grid(False)
plt.show()

# Put model in evaluation mode
model.eval()

# Turn off gradient tracking
with torch.no_grad():
    y_train_logits = model(X_train)
    y_train_probs = torch.sigmoid(y_train_logits)
    y_train_preds = (y_train_probs >= 0.5).float()  # threshold at 0.5

    # Compute test loss
    train_loss = loss_fn(y_train_logits, y_train)

# Accuracy
train_acc = accuracy_score(y_train.numpy(), y_train_preds.numpy())

print(f"Train Loss: {train_loss.item():.4f}")
print(f"Train Accuracy: {train_acc * 100:.2f}%")

#-----------------------------------------------------------------------------------
#can do the same for the training model 

model.eval()

# Turn off gradient tracking
with torch.no_grad():
    # TODO: Compute predictions for X_test
    # Use sigmoid, threshold at 0.5 to get binary labels
    # Then compute test_loss and test_acc
    # Put model in evaluation mode
    
    y_test_logits = model(X_test)
    y_test_probs = torch.sigmoid(y_test_logits)
    y_test_preds = (y_test_probs >= 0.5).float()  # threshold at 0.5

    # Compute test loss
    test_loss = loss_fn(y_test_logits, y_test)

# Accuracy
test_acc = accuracy_score(y_test.numpy(), y_test_preds.numpy())
pass

# Final print
print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
