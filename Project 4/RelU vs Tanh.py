# TanH Network
import torch.nn as nn

# New network: same depth/width, but with Tanh activations
class DeepMLP_Tanh(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
    
# Instantiate the model
tanh_deep_mlp = DeepMLP_Tanh()
print(tanh_deep_mlp)

# Instantiate and train Tanh-based model
tanh_train_losses, tanh_val_losses, tanh_train_accuracies, tanh_val_accuracies = train_deep_model(
    tanh_deep_mlp, train_loader, val_loader, epochs=15
)

#Tanh vs RelU comparison
import matplotlib.pyplot as plt

epochs = range(1, len(tanh_train_losses) + 1)

plt.figure(figsize=(12, 10))

# Train Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, relu_train_losses, label="ReLU")  # from Section 3
plt.plot(epochs, tanh_train_losses, label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()

# Validation Loss
plt.subplot(2, 2, 2)
plt.plot(epochs, relu_val_losses, label="ReLU")  # from Section 3
plt.plot(epochs, tanh_val_losses, label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()

# Train Accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, relu_train_accuracies, label="ReLU")  # from Section 3
plt.plot(epochs, tanh_train_accuracies, label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Comparison")
plt.legend()

# Validation Accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, relu_val_accuracies, label="ReLU")  # from Section 3
plt.plot(epochs, tanh_val_accuracies, label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()

plt.tight_layout()
plt.show()

