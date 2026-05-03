import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the model
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # First linear layer
            nn.ReLU(), # Activation function
            nn.Linear(hidden_dim, output_dim) # Output layer
        )

    # Forward pass
    def forward(self, x):
        return self.network(x)

# Instantiate model
input_dim = X_train_tensor.shape[1]
hidden_dim = 32
output_dim = num_classes

# Create model instance
model_base = BaselineMLP(input_dim, hidden_dim, output_dim)
print(model_base)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_base.parameters(), lr=0.01)

# Store loss values
train_losses_base = []
val_losses_base = []

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Training phase
    model_base.train() # Set model to training mode
    optimizer.zero_grad() # Clear gradients
    outputs = model_base(X_train_tensor) # Forward pass
    loss = criterion(outputs, y_train_tensor) # Compute loss
    loss.backward() # Backpropagation
    optimizer.step() # Update parameters
    train_losses_base.append(loss.item()) # Store training loss


    # Validation phase
    model_base.eval()
    # Compute validation loss without gradient tracking
    with torch.no_grad():
        val_outputs = model_base(X_val_tensor) # Forward pass on validation set
        val_loss = criterion(val_outputs, y_val_tensor) # Compute validation loss
        val_losses_base.append(val_loss.item()) # Store validation loss

    # Optional: Print every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# Plot training and validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses_base, label="Training Loss")
plt.plot(val_losses_base, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss (No Regularization)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

