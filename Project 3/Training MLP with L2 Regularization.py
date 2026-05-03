import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Model (same as before)
class MLP_L2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_L2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model
model_l2 = MLP_L2(input_dim=8, hidden_dim=64, output_dim=10)

# Loss and optimizer with weight decay (L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=1e-4)

# Train model
epochs = 1000
train_losses_l2 = []
val_losses_l2 = []


for epoch in range(epochs):
    # Training
    model_l2.train()
    outputs = model_l2(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    train_losses_l2.append(loss.item())

    # Validation
    model_l2.eval()
    with torch.no_grad():
        val_outputs = model_l2(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses_l2.append(val_loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional: Print every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# Plot training vs. validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses_l2, label="Training Loss (L2)")
plt.plot(val_losses_l2, label="Validation Loss (L2)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss (With L2 Regularization)")
plt.legend()
plt.grid(True)
plt.show()

# Compare final validation loss
CT_l2_val_loss = val_losses_l2[-1] # Final validation loss for L2 regularization
CT_dropout_val_loss = val_losses_dropout[-1] # Final validation loss for Dropout (from previous code)

print(f"CT_Final Val Loss (L2): {CT_l2_val_loss:.4f}")
print(f"CT_Final Val Loss (Dropout): {CT_dropout_val_loss:.4f}")
