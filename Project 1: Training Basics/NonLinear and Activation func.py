import torch
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------------
#Example: Non-linear (no activation)
# Set seed
torch.manual_seed(0)

# Input tensor (4 samples, 1 feature)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# Define dimensions
input_dim = 1
hidden_dim = 3
output_dim = 1

# Initialize parameters

W1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1 = torch.randn(hidden_dim, requires_grad=True)
W2 = torch.randn(hidden_dim, input_dim, requires_grad=True)
b2 = torch.randn(output_dim, requires_grad=True)

# Define forward function
def predict(X):
    hidden = X @ W1 + b1
    output = hidden @ W2 + b2
    return output

# Compute prediction
y_pred = predict(x)
print("y_pred:", y_pred)

#------------------------------------------------------------------------------------
#Example 2: Manual Input
# Define ReLU manually
def my_relu(x):
    return torch.clamp(x, min=0)

# Test on a sample tensor
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", test_x)
print("ReLU Output:", my_relu(test_x))

#--------------------------------------------------
# Define Sigmoid manually
def my_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Test
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", test_x)
print("Sigmoid Output:", my_sigmoid(test_x))

#-------------------------------------------------- 

# Step 1: Define the tanh function manually
def my_tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# Step 2: Test input
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Step 3: Print results
print("Input:", test_x)
print("Tanh Output:", my_tanh(test_x))

#----------------------------------------------------------------------------------------
# Example 3: Non-linear with ReLU activation
input_dim_t = 1
hidden_dim_t = 4
output_dim_t = 1

# Sample input
x_t = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize parameters with requires_grad=True
W1t_relu_t = torch.randn(input_dim_t, hidden_dim_t, requires_grad=True)
b1t_relu_t = torch.randn(hidden_dim_t, requires_grad=True)
W2t_relu_t = torch.randn(hidden_dim_t, output_dim_t, requires_grad=True)
b2t_relu_t = torch.randn(output_dim_t, requires_grad=True)

# Define ReLU activation
def my_relu_t(x):
    return torch.clamp(x, max = 0)

# Define prediction function with ReLU
def predict_relu_t(X):
    hidden = my_relu_t(X @ W1t_relu_t + b1t_relu_t)
    output = hidden @ W2t_relu_t + b2t_relu_t
    return output

# Compute predictions
y_pred_relu_t = predict_relu_t(x)

# Print output
print("y_pred_relu_t:", y_pred_relu_t)

#-----------------------------------------------------------------------------------------
#Example 4: Training Loop with ReLU and MSE Loss
torch.manual_seed(42)
x = torch.linspace(0, 2, 100).unsqueeze(1)  # 100 points from 0 to 2
y = x**2 + 0.1 * torch.randn_like(x)        # Nonlinear targets

# Set learning rate and number of epochs
lr = 0.01
epochs = 3000

# Store training losses
losses_linear = []
losses_relu = []

for epoch in range(epochs):
    
    # ===== Linear → Linear =====
    preds_lin = predict_linear(x)
    loss_lin = mse_loss(preds_lin, y)
    loss_lin.backward()

    with torch.no_grad():
        W1_lin -= lr * W1_lin.grad
        b1_lin -= lr * b1_lin.grad
        W2_lin -= lr * W2_lin.grad
        b2_lin -= lr * b2_lin.grad
        
        # Zero gradients
        W1_lin.grad.zero_()
        b1_lin.grad.zero_()
        W2_lin.grad.zero_()
        b2_lin.grad.zero_()

    losses_linear.append(loss_lin.item())

    # ===== Linear → ReLU → Linear =====
    preds_relu = predict_relu(x)
    loss_relu = mse_loss(preds_relu, y)
    loss_relu.backward()

    with torch.no_grad():
        W1_relu -= lr * W1_relu.grad
        b1_relu -= lr * b1_relu.grad
        W2_relu -= lr * W2_relu.grad
        b2_relu -= lr * b2_relu.grad

        W1_relu.grad.zero_()
        b1_relu.grad.zero_()
        W2_relu.grad.zero_()
        b2_relu.grad.zero_()

    losses_relu.append(loss_relu.item())