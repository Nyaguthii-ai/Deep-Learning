import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define model with dropout
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # 🔽 Dropout added here
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate model
model_dropout = MLPWithDropout(input_dim, hidden_dim, output_dim, dropout_prob=0.5)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_dropout.parameters(), lr=0.01)

# For tracking loss
train_losses_dropout = []
val_losses_dropout = []

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Training
    model_dropout.train()
    optimizer.zero_grad()
    outputs = model_dropout(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses_dropout.append(loss.item())

    # Validation
    model_dropout.eval()
    with torch.no_grad():
        val_outputs = model_dropout(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses_dropout.append(val_loss.item())

    # Optional: Print every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses_dropout, label='Training Loss (Dropout)')
plt.plot(val_losses_dropout, label='Validation Loss (Dropout)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss (With Dropout)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate on test set
model_dropout.eval() # Set model to evaluation mode to disable dropout
# Compute predictions on test set
with torch.no_grad():
    CT_logits = model_dropout(X_test_tensor) # Forward pass on test set
    CT_preds = torch.argmax(CT_logits, dim=1) # Get predicted class labels

# Convert tensors to numpy for sklearn metrics
CT_y_val_np = y_test_tensor.cpu().numpy() # True labels
CT_y_pred_np = CT_preds.cpu().numpy() # Predicted labels

from sklearn.metrics import accuracy_score, f1_score
CT_val_acc = accuracy_score(CT_y_val_np, CT_y_pred_np) # Compute accuracy
CT_val_f1 = f1_score(CT_y_val_np, CT_y_pred_np, average="macro") # Compute macro F1 score

print(f"CT_Val Accuracy (Dropout): {CT_val_acc:.4f}")
print(f"CT_Val Macro F1 (Dropout): {CT_val_f1:.4f}")
