import matplotlib.pyplot as plt

# Define Custom Initialization Function

# Apply Xavier or He initialization
def initialize_weights(model, method="xavier"):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "he_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif method == "he_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                raise ValueError("Unsupported init method.")

# Create Models and Apply Init
# ReLU MLP using He Normal
he_mlp = DeepMLP()  # same arch as before
initialize_weights(he_mlp, method="he_normal")

# Tanh MLP using Xavier
xavier_mlp = DeepMLP_Tanh()  # same arch as before
initialize_weights(xavier_mlp, method="xavier")

# Train and Evaluate Models
#He Init (Normal)
he_train_losses, he_val_losses, he_train_accuracies, he_val_accuracies = train_deep_model(
    he_mlp, train_loader, val_loader, num_epochs=15)

# Xavier Init
xavier_train_losses, xavier_val_losses, xavier_train_accuracies, xavier_val_accuracies = train_deep_model(
    xavier_mlp, train_loader, val_loader, num_epochs=15)

#Visualize Results - ReLU
# Plot train loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(relu_train_losses, label='Default Init - ReLU Uniform')
plt.plot(he_train_losses, label='He Normal Init - ReLU Normal')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(relu_train_accuracies, label='Default Init - ReLU Uniform')
plt.plot(he_train_accuracies, label='He Normal Init - ReLU Normal')
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()

#Visualize Results - Tanh
# Plot train loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(tanh_train_losses, label='Default Init - Tanh')
plt.plot(xavier_train_losses, label='Xavier Init')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(tanh_train_accuracies, label='Default Init - Tanh')
plt.plot(xavier_train_accuracies, label='Xavier Init')
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()