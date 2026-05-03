import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Week5.Convolution_Neural_Networks_II as cnn2   # rename the file
import numpy as np
from PIL import Image

print(cnn2.train_dataset)   # only works if the script defines it

# Get a single image from Oxford Pets
img, label = train_dataset[0]  # CHW format, [0,1] range
img_batch = img.unsqueeze(0)   # Add batch dim: (1,3,H,W)

# Horizontal edge detection kernel (same for R,G,B)
horizontal_kernel = torch.tensor(
    [[[-1, -1, -1],
      [ 0,  0,  0],
      [ 1,  1,  1]],  # Red

     [[-1, -1, -1],
      [ 0,  0,  0],
      [ 1,  1,  1]],  # Green

     [[-1, -1, -1],
      [ 0,  0,  0],
      [ 1,  1,  1]]]  # Blue
).unsqueeze(0).float()  # Shape: (1,3,3,3)

# Apply convolution
horizontal_feature_map = F.conv2d(img_batch, horizontal_kernel, stride=1, padding=1)

# Show original and feature map side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img.permute(1, 2, 0))
axs[0].set_title(f"Original Image\n{train_dataset.classes[label]}")
axs[0].axis("off")

axs[1].imshow(horizontal_feature_map[0, 0].detach().numpy(), cmap="gray")
axs[1].set_title("Horizontal Edge Detection")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------
# Vertical edge detection kernel (same for R,G,B)
# Vertical edge detection kernel (same for R,G,B)
vertical_kernel = torch.tensor(
    [[[ -1,  0,  1],
      [ -1,  0,  1],
      [ -1,  0,  1]],  # Red

     [[ -1,  0,  1],
      [ -1,  0,  1],
      [ -1,  0,  1]],  # Green

     [[ -1,  0,  1],
      [ -1,  0,  1],
      [ -1,  0,  1]]]  # Blue
).unsqueeze(0).float()  # Shape: (1,3,3,3)

# Apply convolution
vertical_feature_map = F.conv2d(img_batch, vertical_kernel, stride=1, padding=1)

# Show original and feature map side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img.permute(1, 2, 0))
axs[0].set_title(f"Original Image\n{train_dataset.classes[label]}")
axs[0].axis("off")

axs[1].imshow(vertical_feature_map[0, 0].detach().numpy(), cmap="gray")
axs[1].set_title("Vertical Edge Detection")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------
#Horizontal Edge Detection (manually using Numpy)

# 1. Get an image (convert tensor to NumPy for demonstration)
# Assume `img` is the CHW tensor from your dataset (values 0–1)
img_np = img.permute(1, 2, 0).numpy()  # HWC format for convenience
img_np = (img_np * 255).astype(np.float32)  # scale to 0-255

# Split channels
R = img_np[:, :, 0]
G = img_np[:, :, 1]
B = img_np[:, :, 2]

# 2. Define horizontal edge kernel for each channel (same kernel)
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

def convolve2d_single_channel(channel, kernel, stride=1, padding=1):
    # Pad channel
    if padding > 0:
        channel = np.pad(channel, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    kernel_h, kernel_w = kernel.shape
    h, w = channel.shape
    out_h = (h - kernel_h) // stride + 1
    out_w = (w - kernel_w) // stride + 1
    output = np.zeros((out_h, out_w))

    # Convolution
    for y in range(0, h - kernel_h + 1, stride):
        for x in range(0, w - kernel_w + 1, stride):
            region = channel[y:y+kernel_h, x:x+kernel_w]
            output[y//stride, x//stride] = np.sum(region * kernel)

    return output

# 3. Apply kernel to each channel
feature_R = convolve2d_single_channel(R, kernel)
feature_G = convolve2d_single_channel(G, kernel)
feature_B = convolve2d_single_channel(B, kernel)

# 4. Sum across channels to get final feature map
feature_map = feature_R + feature_G + feature_B

# 5. Visualize
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img_np.astype(np.uint8))
axs[0].set_title(f"Original Image\n{train_dataset.classes[label]}")
axs[0].axis("off")

axs[1].imshow(feature_map, cmap="gray")
axs[1].set_title("Horizontal Edge Detection (Manual NumPy)")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------
# Task: Define a sharpening kernel (3×3) for RGB and apply it using F.conv2d to an image tensor from the dataset. 
# Visualize the original image and the sharpened feature map side-by-side.
# Kernel shape is (1, 3, 3, 3) and of type float. unsqueeze(0) to add batch dimension. Use .detach().numpy()to plot.

# Define sharpening kernel (same across RGB)
CT_sharpen_kernel = torch.tensor(
    [[[ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]],

     [[ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]],  # Green

      [[ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]]]  # Blue
).unsqueeze(0).float()  # Shape: (1,3,3,3)

# Prepare input image (batch, channel, H, W)
CT_img_batch = img.unsqueeze(0)

# Apply convolution
CT_sharpen_map = F.conv2d(CT_img_batch, CT_sharpen_kernel, stride=1, padding=1)

# Visualize
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img.permute(1, 2, 0))
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(CT_sharpen_map[0, 0].detach().numpy(), cmap="gray")
axs[1].set_title("Sharpened Feature Map")
axs[1].axis("off")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------------------

# Manual Convolution on RGB Channels (NumPy)
# Define sharpening kernel (NumPy)
CT_sharpen_np_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# Apply convolution to each channel
CT_feat_R = convolve2d_single_channel(R, CT_sharpen_np_kernel)
CT_feat_G = convolve2d_single_channel(G, CT_sharpen_np_kernel)
CT_feat_B = convolve2d_single_channel(B, CT_sharpen_np_kernel)

# Combine results
CT_sharpened_feature_map = CT_feat_R + CT_feat_G + CT_feat_B

# Combine results
CT_sharpened_feature_map = CT_feat_R + CT_feat_G + CT_feat_B

# Plot result
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img_np.astype(np.uint8))
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(CT_sharpened_feature_map, cmap="gray")
axs[1].set_title("Manual Sharpened Map")
axs[1].axis("off")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------------------
