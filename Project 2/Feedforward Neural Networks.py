import numpy as np
import torch

#Exercise 1: manually implement a simple Feedforward Neural Network
# Dimensions
input_dim = 2
hidden_dim = 3
output_dim = 1

# Input sample (batch of 1)
x = np.array([[1.5, -0.5]])  # shape (1, 2)

# Seed and weight initialization
np.random.seed(0)
W1 = np.random.randn(input_dim, hidden_dim)  # (2 → 3)
b1 = np.random.randn(1, hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim)  # (3 → 1)
b2 = np.random.randn(1, output_dim)

# Activation function: ReLU
def relu(z):
    return np.maximum(0, z)

# Optional: You can try sigmoid too
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass
z1 = np.dot(x, W1) + b1      # Linear transform
h = relu(z1)                 # Apply activation
y_hat = np.dot(h, W2) + b2   # Output layer

print("Input:", x)
print("Hidden activations:", h)
print("Output prediction:", y_hat)

#---------------------------------------------------------------------------
#Exercise 2: Manual forward pass on pyTorch
# Input
x_task = torch.tensor([[0.8, -1.2]], dtype=torch.float32)

# Weights and biases
torch.manual_seed(42)
W1_task = torch.randn(2, 4)
b1_task = torch.randn(1, 4)

W2_task = torch.randn(4, 1)
b2_task = torch.randn(1, 1)

# ReLU activation
def relu(z):
    return torch.maximum(z, torch.tensor(0.0))

# TODO: Complete the forward pass
# Step 1: Linear transformation
#z1 = None
z1 = x_task @ W1_task + b1_task 

# Step 2: Apply activation
#h = None
h = relu(z1)               # Activation

# Step 3: Output layer
#y_hat_task = None
y_hat_task = h @ W2_task + b2_task    # Output layer
print("Output prediction:", y_hat_task.item())

#---------------------------------------------------------------------------------------
#Exercise 3: manual interference and accuracy calculation

# Steps:
for xi, yi in zip(X_real, y_real): 
    # 1. Compute z1 = X_real @ W1_real + b1_real
    z1 = X_real @ W1_real + b1_real
    
    # 2. Apply ReLU: h1 = relu(z1)
    h1 = relu(z1)
    
    # 3. Compute output logits = h1 @ W2_real + b2_real
    logits = h1 @ W2_real + b2_real
        
    y_pred = 1 if (logits >= 0).any() >= 0 else 0   # binary case

    # 4. Compare
    if y_pred == yi:
        correct_predictions += 1

# 5. Compare predictions to y_real and compute accuracy
model_accuracy = correct_predictions / total_predictions
print(f"✅ Model Accuracy: {model_accuracy:.2f}")