import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Exercise 1: Create a simple dataset X with a few input values 
# Manually define parameters W and b, Implement a predict(X) function
# Compute predictions using our linear model

# Step 1: Tiny toy dataset
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Inputs
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Ground truth

# Step 2: Manually initialized weight and bias
W = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# Step 3: Define predict function (ŷ = XW + b)
def predict(X):
    return torch.matmul(X, W) + b

# Step 4: Make predictions
y_pred = predict(X)

# Step 5: Show results
print("Predictions:")
print(y_pred)

#---------------------------------------------------------------------------
# Exercise 2: Compute the Mean Squared Error (MSE) loss between predictions and true values
# MSE Loss = (1/n) * Σ(y_true - y_pred)^2

# Step 1: Define MSE loss function
def mse_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

# Step 2: Define targets and predictions
Y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])
Y_pred = torch.tensor([[2.5], [5.5], [6.0], [9.0], [12.0]])

# Step 3: Compute loss
loss =  mse_loss(Y_pred, Y_true)

# Step 4: Print the result
print("MSE Loss:", loss.item())

#---------------------------------------------------------------------------
#Exercise 3: Load a dataset from a CSV file, preprocess it, and prepare it for training a linear regression model.
# 1. Load dataset
data = pd.read_csv("Concrete_Data.csv")  # Adjust path as needed

# 2. Separate features and target
inputs = data.iloc[:, :-1].values       # First 8 columns = features
targets = data.iloc[:, -1].values.reshape(-1, 1)  # Last column = target

# 3. Normalize input features
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)

# 4. Convert to PyTorch tensors
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# 5. Shuffle and split into training (80%) and test (20%) sets
torch.manual_seed(42)
n_samples = inputs_tensor.shape[0]
indices = torch.randperm(n_samples)
split_idx = int(n_samples * 0.8)

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = inputs_tensor[train_indices]
y_train = targets_tensor[train_indices]
X_test = inputs_tensor[test_indices]
y_test = targets_tensor[test_indices]

# 6. Initialize model parameters
num_features = X_train.shape[1]  # Should be 8
W = torch.randn((num_features, 1), requires_grad=True)
W.data *= 0.01  # Scale down initial values
b = torch.zeros((1,), requires_grad=True)

# 7. Confirm everything is set up
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)

#----------------------------------------------------------------------------
#Exercise 4: Implement a training loop to optimize the parameters W and b using gradient descent.

# Initialize list to store loss values
losses = []
# Set learning rate and number of training epochs
learning_rate = 0.001
epochs = 100

for epoch in range(epochs):
    # 1. Predict - Forward pass: compute predictions
    y_pred = predict(X_train)

    # Evaluate the model on test set

    # Step 1: Predict on test data
    y_test_pred = predict(X_test)

    # 2. Compute loss
    loss = mse_loss(y_pred, y_train)

    # Step 2: Compute MSE on test set
    test_loss = mse_loss(y_test_pred, y_test)

    losses.append(loss.item())  # Track the loss 

    # 3. Backpropagation - Backward pass: compute gradients
    loss.backward()

    # 4. Update weights
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # 5. Reset gradients
    W.grad.zero_()
    b.grad.zero_()

 # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Step 3: Print the result
    print(f"Test Loss: {test_loss.item():.4f}")
    
# Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.show()
