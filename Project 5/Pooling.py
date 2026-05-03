import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Week5.Edge_Detection as ed   # rename the file

# reuse the map computed in the edge‑detection module
horizontal_feature_map = ed.horizontal_feature_map

# Create a simple 4x4 feature map
feature_map = np.array([
    [1, 2, 5, 6],
    [3, 4, 7, 8],
    [9, 10, 13, 14],
    [11, 12, 15, 16]
])

def pool2d(matrix, size=2, mode="max"):
    H, W = matrix.shape
    pooled = []
    for i in range(0, H, size):
        row = []
        for j in range(0, W, size):
            window = matrix[i:i+size, j:j+size]
            if mode == "max":
                row.append(np.max(window))
            else:
                row.append(np.mean(window))
        pooled.append(row)
    return np.array(pooled)

max_pooled = pool2d(feature_map, size=2, mode="max")
avg_pooled = pool2d(feature_map, size=2, mode="avg")

print("Original Feature Map:\n", feature_map)
print("\nMax Pooled (2x2):\n", max_pooled)
print("\nAverage Pooled (2x2):\n", avg_pooled)
#Output
#Original Feature Map:
#[[ 1  2  5  6]
#[ 3  4  7  8]
#[ 9 10 13 14]
#[11 12 15 16]]

#Max Pooled (2x2):
#[[ 4  8]
#[12 16]]

#Average Pooled (2x2):
#[[ 2.5  6.5]
#[10.5 14.5]]

# ----------------------------------------------------------------------------------------
#Pooling on Oxford Pet Feature Map (PyTorch)

# Apply Max Pooling (2x2) on the feature map
max_pooled_map = F.max_pool2d(horizontal_feature_map, kernel_size=2, stride=2)

# Apply Average Pooling (2x2) on the feature map
avg_pooled_map = F.avg_pool2d(horizontal_feature_map, kernel_size=2, stride=2)

# Plot original vs pooled feature maps
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].imshow(horizontal_feature_map[0, 0].detach().numpy(), cmap="gray")
axs[0].set_title("Original Feature Map")
axs[0].axis("off")

axs[1].imshow(max_pooled_map[0, 0].detach().numpy(), cmap="gray")
axs[1].set_title("Max Pooled (2x2)")
axs[1].axis("off")

axs[2].imshow(avg_pooled_map[0, 0].detach().numpy(), cmap="gray")
axs[2].set_title("Average Pooled (2x2)")
axs[2].axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------
# Task: Average Pooling on a Sample Image
# Get a sample from resized dataset
CT_sample_img, _ = CT_subset_32[0]  # Shape: (3, 32, 32)
CT_sample_batch = CT_sample_img.unsqueeze(0)

# Apply average pooling (2x2)
CT_avg_pooled = F.avg_pool2d(CT_sample_batch, kernel_size=2, stride=2)

# Visualize original vs pooled
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(CT_sample_img.permute(1, 2, 0))
axs[0].set_title("Original (32×32)")
axs[0].axis("off")

axs[1].imshow(CT_avg_pooled[0].permute(1, 2, 0).detach().numpy())
axs[1].set_title("Average Pooled (2x2)")
axs[1].axis("off")
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------------------

