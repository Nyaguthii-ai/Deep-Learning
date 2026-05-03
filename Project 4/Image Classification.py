import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 training set (we’ll reuse this later too)
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Get class names
class_names = train_dataset.classes

# Plot some random images
fig, axes = plt.subplots(2, 5, figsize=(6, 4))
for i in range(10):
    image, label = train_dataset[i]
    ax = axes[i//5, i%5]
    ax.imshow(image.permute(1, 2, 0))  # convert CHW to HWC
    ax.set_title(class_names[label])
    ax.axis('off')
plt.suptitle("Sample Images from CIFAR-10 Dataset", fontsize=14)
plt.tight_layout()
plt.show()

#count the number of images in each class
from collections import Counter

# Extract labels from the dataset
labels = [label for _, label in train_dataset]

# Count occurrences of each class
class_counts = Counter(labels)

# Convert to class name keys
class_counts_named = {class_names[k]: v for k, v in class_counts.items()}
print(class_counts_named)

# Plot
plt.figure(figsize=(6, 3))
plt.bar(class_counts_named.keys(), class_counts_named.values())
plt.title("Training Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from collections import Counter

# --------------------------------------------------------------------------------
# Counting the number of images in each class for the CT dataset    

# Step 1: Extract labels from train_dataset
CT_labels = []
for _, label in train_dataset:
    CT_labels.append(label)

# Step 2: Count occurrences of each label
CT_class_counts = Counter(CT_labels)

# Step 3: Map indices to class names
CT_class_counts_named = {}
for idx, count in CT_class_counts.items():
    CT_class_counts_named[class_names[idx]] = count

# Step 4: Print counts
print("CT_Class Counts:")
for name, count in CT_class_counts_named.items():
    print(f"{name}: {count}")

# --------------------------------------------------------------------------------
#Task – Inspect single image tensor

# Step 1: Access first sample
CT_image, CT_label = train_dataset[0]

# Step 2: Print properties
print("CT_Image shape:", CT_image.shape)       # Expect [3, 32, 32]
print("CT_Min pixel value:", float(image.min())) # Expect 0.0 (after ToTensor normalization)
print("CT_Max pixel value:", float(image.max())) # Expect 1.0 (after ToTensor normalization)
print("CT_Label index:", CT_label) # Expect an integer between 0 and 9 = CT_Label index: 6
print("CT_Label name:", class_names[CT_label]) # Expect a class name corresponding to the label index = CT_Label name: frog

# --------------------------------------------------------------------------------
# Visualise flattening of small image
import matplotlib.pyplot as plt
import numpy as np

# Simulate a grayscale 4x4 image (1 channel)
image = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [130, 140, 150, 160]
])

# Flatten it
flattened = image.flatten()

# Plot original and flattened
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original 4×4 Image")
axs[0].axis('off')

axs[1].bar(np.arange(len(flattened)), flattened)
axs[1].set_title("Flattened Vector (Length 16)")
axs[1].set_xlabel("Index")
axs[1].set_ylabel("Pixel Value")

plt.suptitle("Visualizing the Flattening Process")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------
# Now apply flattening to a real image from the dataset

# Fetch one sample
image, label = train_dataset[0]
print(f"Original shape: {image.shape}")  # [3, 32, 32]

# Flatten the image
flattened = image.view(-1)  # or image.reshape(-1)
print(f"Flattened shape: {flattened.shape}")  # [3072]
 
# --------------------------------------------------------------------------------
#redefine our transform to convert image to tensor as well as to flatten it to a 1D vector.
import torchvision.transforms as transforms

# New transform: ToTensor + Flatten
transform = transforms.Compose([
    transforms.ToTensor(),                            # [3, 32, 32], scaled to [0,1]
    transforms.Lambda(lambda x: x.view(-1))           # Flatten to [3072]
])

# Reload the dataset with the new transform
from torchvision.datasets import CIFAR10

# Reload with new transform
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# Check the shape of the first sample
flattened_image, label = train_dataset[0]
print(f"Flattened image shape: {flattened_image.shape}")  # Expect [3072]

# ----------------------------------------------------------------------------------
# Create DataLoaders to handle batching and shuffling for training our model.
import torch
from torch.utils.data import DataLoader, random_split

# Set seed for reproducibility
torch.manual_seed(42)

# Define batch size
batch_size = 64

# Optionally split train dataset into train + validation (e.g., 45k train, 5k val)
train_size = int(0.9 * len(train_dataset))  # 45,000
val_size = len(train_dataset) - train_size  # 5,000

train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Wrap in DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check sizes
print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Validation batches: {len(val_loader)}")
print(f"✅ Test batches: {len(test_loader)}")

#Output
#✅ Train batches: 704
#✅ Validation batches: 79
#✅ Test batches: 157

# check the shape of of an image and label in the first batch
for images, labels in train_loader:
    print(f"✅ Batch shape: {images.shape}")  # Should be [batch_size, 3072]
    print(f"✅ Labels shape: {labels.shape}")  # Should be [batch_size]
    break  # Just check the first batch

#Output
#✅ Batch shape: torch.Size([64, 3072])
#✅ Labels shape: torch.Size([64])

# ---------------------------------------------------------------------------------
# Fetch one batch from train_loader and print shape of features and labels to verify batching logic (64 × 3072 for features, 64 for labels)

# Step 1: Fetch one batch
CT_images_batch, CT_labels_batch = next(iter(train_loader))
# Step 2: Print shapes
print("CT_Batch feature shape:", CT_images_batch.shape)  # Expect [64, 3072]
print("CT_Batch label shape:", CT_labels_batch.shape)    # Expect [64]

# ---------------------------------------------------------------------------------
# Task – Check DataLoader batch shape

# Step 1: Fetch one batch
CT_images_batch, CT_labels_batch = next(iter(train_loader))

for CT_images_batch, CT_labels_batch in train_loader:
    
    # Flatten images
    CT_images_batch = CT_images_batch.view(CT_images_batch.size(0), -1)
    
    print("CT_Batch feature shape:", CT_images_batch.shape)   # Should be [64, 3072]
    print("CT_Batch label shape:", CT_labels_batch.shape)
    
    break

# ---------------------------------------------------------------------------------
