# Exercise 1: Custom Feedforward Pass with Sigmond and RelU
import numpy as np
import matplotlib.pyplot as plt

# Custom input (batch size = 1)
x_custom = np.array([[2.0, -1.0]])

# Weights and biases
np.random.seed(123)
W1_custom = np.random.randn(2, 4)
b1_custom = np.random.randn(1, 4)
W2_custom = np.random.randn(4, 1)
b2_custom = np.random.randn(1, 1)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# TODO: Implement the forward pass and store the final output in `y_out`

#linearization 
z1 = x_custom @ W1_custom + b1_custom 

#Activation formula: sigmoid
h = sigmoid(z1)

#Second Linearization
z2 = h @ W2_custom + b2_custom 

#output relU activation
y_out = relu(z2)

print("Output = ", y_out)

#-----------------------------------------------------------------------------------
#exercise 2: Implementing forward and backward pass

# ===== Initialize a simple network =====
np.random.seed(42)

x = np.array([[2.0, -1.0]])   # single sample input (1, 2)
y = np.array([[1.0]])         # binary target

# Weights & biases
W1 = np.random.randn(2, 3)    # (input_dim, hidden_dim)
b1 = np.random.randn(1, 3)

W2 = np.random.randn(3, 1)    # (hidden_dim, output_dim)
b2 = np.random.randn(1, 1)

# ReLU and its derivative
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Forward pass
z1 = x @ W1 + b1             # Hidden linear
h = relu(z1)                 # Hidden activation
y_hat = h @ W2 + b2          # Output (no activation yet)

# Compute loss (MSE)
loss = 0.5 * (y_hat - y)**2

print("Prediction:", y_hat)
print("Loss:", loss)

# Backward pass
# Derivative of loss w.r.t. prediction
dL_dyhat = y_hat - y     # shape (1, 1)

# Gradients for output layer
dL_dW2 = h.T @ dL_dyhat  # (3, 1)
dL_db2 = dL_dyhat        # (1, 1)

# Gradient flowing into hidden layer
dL_dh = dL_dyhat @ W2.T              # shape (1, 3)
dL_dz1 = dL_dh * relu_derivative(z1) # shape (1, 3)

# Gradients for first layer
dL_dW1 = x.T @ dL_dz1     # (2, 3)
dL_db1 = dL_dz1           # (1, 3)

print("Gradients for W2:", dL_dW2)
print("Gradients for b2:", dL_db2)
print("Gradients for W1:", dL_dW1)
print("Gradients for b1:", dL_db1)

#Stochastic Gradient Descent (SGD) update
learning_rate = 0.01

# Gradient descent update
W2 -= learning_rate * dL_dW2
b2 -= learning_rate * dL_db2
W1 -= learning_rate * dL_dW1
b1 -= learning_rate * dL_db1

#-------------------------------------------------------------------------------
#Exercise 3: Implementing a training loop

# Initialize weights and biases again (reset)
np.random.seed(42)

x = np.array([[1.0, -1.0]])
y = np.array([[1.0]])

W1 = np.random.randn(2, 3)
b1 = np.random.randn(1, 3)
W2 = np.random.randn(3, 1)
b2 = np.random.randn(1, 1)

# Hyperparameters
lr = 0.1
epochs = 10
losses = []

# Training loop
for epoch in range(epochs):
    # ----- Forward pass -----
    z1 = x @ W1 + b1
    h = relu(z1)
    y_hat = h @ W2 + b2
    loss = 0.5 * (y_hat - y)**2
    losses.append(loss.item())

    # ----- Backward pass -----
    dL_dyhat = y_hat - y
    dL_dW2 = h.T @ dL_dyhat
    dL_db2 = dL_dyhat

    dL_dh = dL_dyhat @ W2.T
    dL_dz1 = dL_dh * relu_derivative(z1)
    dL_dW1 = x.T @ dL_dz1
    dL_db1 = dL_dz1
    

    # ----- Parameter update -----
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

#Plot the Loss Over Epochs
plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='--')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()