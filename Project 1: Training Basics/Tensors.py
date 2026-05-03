import torch
#Generate scalars, vectors, matrices
# 1. Scalar tensor
my_scalar = torch.tensor(10)

# 2. Vector tensor
my_vector = torch.tensor([5, 10, 15])

# 3. Matrix tensor (2×3)
my_matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Print tensors and shapes
print("Scalar:", my_scalar, "Shape:", my_scalar.shape)
print("Vector:", my_vector, "Shape:", my_vector.shape)
print("Matrix:\n", my_matrix, "Shape:", my_matrix.shape)

#--------------------------------------------------------------------------------
#Exercise 2: Create two vectors, add and multiply them element-wise, 
# and print the results.
# Create the vectors
a = torch.tensor([1, 3, 5])
b = torch.tensor([2, 4, 6])

# Operations
add_result = (a + b)
mult_result = (a * b)
dot_result = (torch.dot(a, b))

# Display results
print("Addition:", add_result)
print("Multiplication:", mult_result)
print("Dot product:", dot_result)

#--------------------------------------------------------------------------------
#Exercise 3: Use broadcasting to add a scalar to a vector and print the result.
# Define matrix and vector
my_matrix = torch.tensor([[7, 8, 9], [10, 11, 12]])
my_vector = torch.tensor([1, 1, 1])

# Perform addition with broadcasting
broadcast_sum = (my_matrix + my_vector)

# Display result
print("Result of broadcasting addition:")
print(broadcast_sum)

#--------------------------------------------------------------------------------
#Exercise 4: Given a dataset with features and targets, convert them into PyTorch tensors. 

# Inputs: all columns except the last one
inputs = data.iloc[:, :-1]

# Targets: only the last column
targets = data.iloc[:, -1]

# Convert inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

# Slice the first 100 rows
inputs_100 = torch.tensor(inputs_tensor[:100], dtype=torch.float32)
targets_100 = torch.tensor(targets_tensor[:100], dtype=torch.float32)

# Print shapes
print("Shape of inputs_100:", inputs_100.shape)
print("Shape of targets_100:", targets_100.shape)

#--------------------------------------------------------------------------------
#Exercise 5: Plot a histogram for the Water (component 4) feature.
import matplotlib.pyplot as plt
# Extract the column
water_feature = torch.tensor(inputs_tensor[:, 4], dtype=torch.float32 )

# Plot histogram for the target variable (Concrete Strength)
plt.hist(inputs, bins=10, edgecolor='k')
plt.title('Water (component 3)(kg in a m^3 mixture)')
plt.xlabel('Water (m^3)')
plt.ylabel('Count')
plt.show()


