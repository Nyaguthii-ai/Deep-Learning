# Creating Tensors in PyTorch
Once PyTorch is imported, we can create tensors using different methods:

torch.tensor(): Create tensor from a Python list or number.
torch.zeros(): Create tensor filled with zeros.
torch.ones(): Create tensor filled with ones.
torch.rand(): Create tensor filled with random numbers between 0 and 1.

📐 Understanding Dimensions (Shape)
A scalar has 0 dimensions: just a number.
A vector has 1 dimension: a list of numbers.
A matrix has 2 dimensions: rows and columns.
A 3D tensor is like a stack of matrices.
Knowing the shape of your tensors is very important in deep learning!
(Mismatched shapes cause errors.)

Tensor Attributes
Every tensor in PyTorch comes with important attributes that describe its properties.

Attribute	Description	Example
.dtype	Data type of tensor elements (e.g., float32, int64)	torch.float32
.shape	Dimensions of the tensor (rows, columns, etc.)	(2, 3)
.device	Where the tensor is stored (CPU or GPU)	cpu or cuda
Knowing these attributes helps you debug and optimize your deep learning models easily.

Broadcasting is a powerful concept in PyTorch that allows tensors with different shapes to be combined automatically during operations.
Broadcasting saves memory (no need to manually copy or reshape tensors),
Makes code shorter, cleaner, and more efficient,
Often used to add bias terms to each row in neural networks.

# We load datasets in the form of Tensors for our deep learning modules

Once a dataset is loaded, the first step is to explore its structure:

.shape → Tells us the number of rows and columns.
.head() → Shows the first 5 rows to get a quick sense of the data.
.columns/columns.tolist() → Lists all feature names.

# Inputs: all columns except the last one
inputs = data.iloc[:, :-1]

# Targets: only the last column
targets = data.iloc[:, -1]

The inputs should become a tensor of shape:(1030, 8)
(1030 examples, each with 8 features)

The targets should become a tensor of shape: (1030, )
(One target value for each example)

📌 Note:
When creating tensors from a NumPy array or DataFrame, we often need to specify:

dtype=torch.float32