#using nn.sequential to build a multi-layer perceptron
import torch
import torch.nn as nn

input_size = 8
hidden_size = 16
output_size = 1

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),  # Layer 1 - Linear layer mapping from a-dimensional input to b-dimensional output
    nn.ReLU(),                           # Activation
    nn.Linear(hidden_size, output_size)  # Output layer
)

#output: Sequential(
#  (0): Linear(in_features=8, out_features=16, bias=True)
#  (1): ReLU()
#  (2): Linear(in_features=16, out_features=1, bias=True)
#)

#------------------------------------------------------------------------------
#exercise 1: MLP with 1 hidden layer
import torch
import torch.nn as nn

# Custom MLP class for regression
class ConcreteMLP(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(ConcreteMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()                          # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
    
    def forward(self, x):
        x = self.fc1(x)   # Apply first linear layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)   # Apply output layer
        return x

# Create model instance
model = ConcreteMLP()
print(model)
#output: ConcreteMLP(
#  (fc1): Linear(in_features=8, out_features=16, bias=True)
#  (relu): ReLU()
#  (fc2): Linear(in_features=16, out_features=1, bias=True  )
#)

#------------------------------------------------------------------------------
#exercise 2: MLP with 2 hidden layers
class DeeperConcreteMLP(nn.Module):
    def __init__(self):
        super(DeeperConcreteMLP, self).__init__()
        
        self.fc1 = nn.Linear(8, 32)   # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)   # Output layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Create and print model
deep_model = DeeperConcreteMLP()
print(deep_model)

#output: DeeperConcreteMLP( 
#  (fc1): Linear(in_features=8, out_features=32, bias=True)
#  (relu1): ReLU()
#  (fc2): Linear(in_features=32, out_features=16, bias=True
#  (relu2): ReLU()
#  (fc3): Linear(in_features=16, out_features=1, bias=True
#)
