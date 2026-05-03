import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Set seed for reproducibility
torch.manual_seed(42)

# Define transform: scale to [0, 1] and flatten to [3072]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load training and test sets
train_val_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data      = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Split training into 90% train, 10% validation
train_size = int(0.9 * len(train_val_data))   # 45,000
val_size   = len(train_val_data) - train_size # 5,000
train_data, val_data = random_split(train_val_data, [train_size, val_size])

# Wrap in DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Class names
class_names = train_val_data.classes

# Quick sanity checks
print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Validation batches: {len(val_loader)}")
print(f"✅ Test batches: {len(test_loader)}")
print(f"✅ Input dimension: {next(iter(train_loader))[0].shape[1]}")
print(f"✅ Number of classes: {len(class_names)}")

#Output
# **Train batches: 704**: 45,000 training images divided into mini-batches of 64. => 45000/64 = approx 704 batches.

# **Validation batches: 79**: 5,000 validation images in batches of 64. => 5000/64 = approx 79 batches.

# **Test batches: 157**: Same logic as validation — CIFAR-10 test set has 10,000 images.

# **Image shape (flattened):** CIFAR-10 images are 32×32×3 = 3072 pixels. Each image is now represented as a **3072-dimensional vector** (flattened).

# **Number of classes: 10**: CIFAR-10 includes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

# Get one batch from the train loader
CT_images_batch, CT_labels_batch = next(iter(train_loader))

# Print batch details
print("CT_Train batch size:", CT_images_batch.shape[0])     # Expect 64
print("CT_Input dimension:", CT_images_batch.shape[1])      # Expect 3072
print("CT_Number of classes:", len(torch.unique(CT_labels_batch)))  # Expect 10