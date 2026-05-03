#load the dataset using `torchvision.datasets`
import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Point directly to the folder containing images/ and annotations/
data_path = "data/oxford-iiit-pet/"

# Check if expected subfolders exist (images + annotations)
images_exist = os.path.exists(os.path.join(data_path, "images"))
annots_exist = os.path.exists(os.path.join(data_path, "annotations"))

# Only download if either folder is missing
download_flag = not (images_exist and annots_exist)

print(f"Images exist: {images_exist}, Annotations exist: {annots_exist}")

# Check original resolutions
from torchvision import datasets

# Load dataset without resizing to check original image sizes
raw_dataset = datasets.OxfordIIITPet(
    root="data",
    split="trainval",  #Loads training + validation images combined. These are listed in the file trainval.txt provided with the dataset.
    #split="test"       Loads the official test set images listed in test.txt.
    target_types="category",
    download=True,   # assuming you have to download it
    transform=None    # No transform → retains original size
)
#trainval.txt file contains one line per image with four pieces of information:
#<image_id> <class_id> <species> <breed_id> 

# Inspect first few images for their original resolution
for i in range(3):
    img, label = raw_dataset[i]
    print(f"Original size of image {i}: {img.size} (W x H)")

# Define transformations: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # Resize for easier visualization and manageable training
    transforms.ToTensor()          # Convert to tensor, scales pixels to [0, 1]
])

# Load the Oxford-IIIT Pets dataset
train_dataset = datasets.OxfordIIITPet(
    root="./data/",
    split="trainval",
    target_types="category",
    download=download_flag,
    transform=transform
)

# Inspect dataset properties
print(f"Number of images: {len(train_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Example classes: {train_dataset.classes[:5]}")

# Visualize random samples
def show_samples(dataset, n=6):
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        idx = np.random.randint(len(dataset))
        img, label = dataset[idx]
        axes[i].imshow(img.permute(1, 2, 0))  # Convert CHW → HWC
        axes[i].set_title(dataset.classes[label], fontsize=10)
        axes[i].axis("off")
    plt.show()

show_samples(train_dataset)

# ------------------------------------------------------------------------------------------------------
# Task resize to 32x32 and visualize
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# Define transform to resize images to 32x32
CT_transform_32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset with custom transform
CT_dataset_32 = datasets.OxfordIIITPet(
    root="./data/",
    split="trainval",
    target_types="category",
    download=False,
    transform=CT_transform_32
)

# Extract subset of first 100 samples
CT_subset_32 = Subset(CT_dataset_32, list(range(100)))

# Visualize a sample (e.g., index 0)
CT_orig_img, _ = raw_dataset[0] # Original image without resizing
CT_resized_img, _ = CT_subset_32[0] # Resized image (32x32)

# Convert resized tensor back to PIL
CT_resized_pil = transforms.ToPILImage()(CT_resized_img)

# Plot both images
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(CT_orig_img)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(CT_resized_pil)
axs[1].set_title("Resized to 32×32")
axs[1].axis("off")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------------------
# Manual Convolution Implementation
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a toy grayscale image (6x6)
image = np.array([
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [0, 0, 0, 20, 20, 20],
    [0, 0, 0, 20, 20, 20],
    [0, 0, 0, 20, 20, 20]
])

# Step 2: Define a 3x3 horizontal edge detection filter
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Step 3: Convolution operation
def convolve2d(image, kernel, stride=1, padding=0):
    # Add padding if specified
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = image.shape

    # Calculate output dimensions
    out_height = (img_height - kernel_height) // stride + 1
    out_width = (img_width - kernel_width) // stride + 1

    # Initialize output feature map
    output = np.zeros((out_height, out_width))

    # Perform convolution
    for y in range(0, img_height - kernel_height + 1, stride):
        for x in range(0, img_width - kernel_width + 1, stride):
            region = image[y:y+kernel_height, x:x+kernel_width]
            output[y//stride, x//stride] = np.sum(region * kernel)

    return output

# Apply convolution
feature_map = convolve2d(image, kernel)


# Print resulting matrix for clarity
print("Resulting Feature Map (Matrix):")
print(feature_map)

# Visualize
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original Image")
axs[1].imshow(feature_map, cmap='gray')
axs[1].set_title("Feature Map (Edges)")
plt.show()

# -------------------------------------------------------------------------------------------------------
# Task: Implement a 3x3 vertical edge detection filter

# Define diagonal edge detection kernel (custom 3x3)
CT_diag_kernel = np.array([ 
    [-1, 0, 1], 
    [-1, 0, 1], 
    [-1, 0, 1] 
])
# Apply convolution using existing function
CT_diag_feature_map = convolve2d(image, CT_diag_kernel)

# Print resulting matrix for clarity
print("Resulting Feature Map (Matrix):")
print(CT_diag_feature_map)

# -----------------------------------------------------------------------------------------------
