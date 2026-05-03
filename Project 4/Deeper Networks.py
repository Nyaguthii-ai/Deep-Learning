import torch
import torch.nn as nn

# Define deeper MLP
class DeepMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, output_dim=10):
        super(DeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Final output layer
        )

    def forward(self, x):
        return self.network(x)

# Instantiate the model
relu_deep_mlp = DeepMLP()
print(relu_deep_mlp)

# Task 2 – Define and inspect Deep MLP model

# Instantiate model
CT_model_deep = DeepMLP()

# Print model summary
print(CT_model_deep)

# Count parameters
CT_total_params = sum(p.numel() for p in CT_model_deep.parameters() if p.requires_grad)
print("CT_Total trainable parameters:", CT_total_params)

# Get one batch from the train loader
CT_images_batch, CT_labels_batch = next(iter(train_loader))

CT_Number_of_classes = torch.unique(CT_labels_batch).numel()

# Print batch details
print("CT_Train batch size:", CT_images_batch.shape[0])     # Expect 64
print("CT_Input dimension:", CT_images_batch.shape[1])      # Expect 3072
print("CT_Number of classes:", CT_Number_of_classes)  # Expect 10

# ---------------------------------------------------------------------------------------------

#Tracking loss and accuracy
def train_deep_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, epochs = 10):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Optional: for reproducibility
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= total
        train_accuracy = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

# Instantiate and train Tanh-based model
relu_train_losses, relu_val_losses, relu_train_accuracies, relu_val_accuracies = train_deep_model(
    relu_deep_mlp, train_loader, val_loader, epochs=15
)

#Plot Training Curves
import matplotlib.pyplot as plt
# Plot loss and accuracy
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(relu_train_losses, label='Train Loss')
plt.plot(relu_val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(relu_train_accuracies, label='Train Acc')
plt.plot(relu_val_accuracies, label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

