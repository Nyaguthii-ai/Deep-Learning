import numpy as np
#Exercise 1: performs one prediction step using a perceptron with given weights, bias, and a batch of 3 input vectors.
# Inputs (3 samples, 2 features each)
X_t = np.array([
    [1, 2],
    [-1, -2],
    [0, 1]
])

# Fixed weights and bias
w = np.array([1.0, -1.0])
b = 0.5

# TODO: Compute predictions using z = X_t @ w + b, then apply step function
# Store the result in a variable called `y_pred`

# Linear combination
z = X_t @ w + b

# Step function
y_pred = (z >= 0).astype(int)

print(y_pred)

#-----------------------------------------------------------------------------------------------------
#Exercise 2: Plot a perceptron decision boundary 

# Use the learned weights and bias from Section 4
w1, w2 = w[0], w[1]

# Create range of x1 values
x1_vals = np.linspace(-5, 5, 100)

# Compute corresponding x2 values from decision boundary equation
x2_vals = -(w1 * x1_vals + b) / w2

# Plot original data
plt.figure(figsize=(6,6))
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.8)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.8)

# Plot the decision boundary line
plt.plot(x1_vals, x2_vals, color='black', linestyle='--', label='Decision Boundary')

plt.title("Perceptron Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------------------------------------------------------------
#Perception training on real dataset

# Random initialize weights and bias
w_real = np.random.randn(X_real.shape[1])  # shape (2,)
b_real = np.random.randn()


# Training settings
lr = 0.1
epochs = 50
errors_real = []

# Step function
def step(z): return 1 if z >= 0 else 0

# Training loop
for epoch in range(epochs):
    total_errors = 0
    for xi, yi in zip(X_real, y_real):
        z = np.dot(w_real, xi) + b_real
        y_hat = step(z)

        error = yi - y_hat
        if error != 0:
            w_real += lr * error * xi
            b_real += lr * error
            total_errors += 1
    errors_real.append(total_errors)

print("Final weights:", w_real)
print("Final bias:", b_real)

#Exercise 3: Evaluate the perceptron on a test set and compute accuracy

# Step function vectorized
def step(z): return 1 if z >= 0 else 0

#initiate correct predictions and total predictions
correct_predictions = 0
total_predictions = len(X_real) 

# Compute predictions
for xi, yi in zip(X_real, y_real): 
    z = np.dot(w_real, xi) + b_real
    
    # Step function
    y_pred = 1 if z >= 0 else 0

    # Compare with true label
    if y_pred == yi:
        correct_predictions += 1

# Compute accuracy
expected_accuracy = correct_predictions / total_predictions
print(f"✅ Expected Accuracy: {expected_accuracy:.2f}")

#create the decision boundary for the real dataset

# Extract weights and bias from trained model
w1, w2 = w_real[0], w_real[1]
b = b_real

# Set up line for decision boundary
x1_vals = np.linspace(X_real[:, 0].min() - 5, X_real[:, 0].max() + 5, 100)
x2_vals = (-b + w1 * x1_vals) / w2
# Plotting
plt.figure(figsize=(6,6))
plt.scatter(X_real[y_real==0][:, 0], X_real[y_real==0][:, 1], color='cornflowerblue', label='No Disease')
plt.scatter(X_real[y_real==1][:, 0], X_real[y_real==1][:, 1], color='darkorange', label='Disease')

# Plot the decision boundary
plt.plot(x1_vals, x2_vals, color='black', linestyle='--', label='Decision Boundary')

plt.xlabel('Thalach (max heart rate)')
plt.ylabel('Oldpeak (ST depression)')
plt.title('Perceptron Decision Boundary (Real Data)')
plt.legend()
plt.grid(True)
plt.show()

