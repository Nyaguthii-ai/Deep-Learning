import os
import shutil
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import collections

# -----------------------------
# 0. Control Flag
# -----------------------------
# Set it to False to skip this step as the data is already preprocessed.
# Set this flag to True if you want to run the data splitting and copying process. Not necessary.
RUN_PREPROCESSING = False


raw_data_dir = "data"
# Target folder where split data will be stored
split_data_dir = "data/plant-seedlings"

if RUN_PREPROCESSING:
    # -----------------------------
    # 1. Paths
    # -----------------------------
    
    train_dir = os.path.join(split_data_dir, "train")
    test_dir  = os.path.join(split_data_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # -----------------------------
    # 2. Split Logic
    # -----------------------------
    for class_name in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name == "plant-seedlings":  # Skip the target folder
            continue
        if class_name.startswith('.'):  # Skip hidden folders like .ipynb_checkpoints
            continue
        # All images for this class
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    
        # 80-20 train/test split
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
        # Create class subfolders in train/test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
        # Copy images to respective folders
        for img in train_imgs:
            shutil.copy(img, os.path.join(train_dir, class_name))
        for img in test_imgs:
            shutil.copy(img, os.path.join(test_dir, class_name))
    
    print(f"Train/Test split created at: {split_data_dir}")

else:
    print("You've chosen not to run the preprocessing script. No worries, the data is already preprocessed and ready to use! ✅")

# Check the downloaded dataset
for class_name in os.listdir(raw_data_dir):
    class_path = os.path.join(raw_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    print(f"{class_name}: {len(images)} images")

# Load the Train and Test directories to verify the split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1. Path to split dataset
# -----------------------------
data_path = "./data/plant-seedlings"

train_dir = os.path.join(data_path, "train")
test_dir  = os.path.join(data_path, "test")

# -----------------------------
# 2. Transformations
# -----------------------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# -----------------------------
# 3. Load datasets
# -----------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 4. Verify
# -----------------------------
print(f"Classes: {train_dataset.classes}")
print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ---------------------------------------------------------
# Display images after reversing normalization
# ---------------------------------------------------------
# Why do we "unnormalize" here?
# - During preprocessing, images were normalized to ImageNet mean/std
#   so model training works correctly (centered data, stable gradients).
# - For visualization, we must reverse this step to recover the original
#   color range (0–1) so the images look natural.
# ---------------------------------------------------------

def imshow(inp, title=None):
    """Display a tensor image after unnormalizing (ImageNet stats)."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # unnormalize
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Show first 8 training images
images, labels = next(iter(train_loader))
fig = plt.figure(figsize=(12, 6))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    imshow(images[i], title=train_dataset.classes[labels[i]])
plt.show()
# Class Balance Visualization

# Count samples per class
class_counts = collections.Counter([label for _, label in train_dataset.samples])

# Convert to list aligned with class_names
counts = [class_counts[i] for i in range(len(train_dataset.classes))]

# Plot
plt.figure(figsize=(10, 4))
plt.bar(train_dataset.classes, counts)
plt.xticks(rotation=45, ha='right')
plt.title("Class Distribution (Train Set)")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# Task:  Implement an Unnormalize Helper & Visualize

def CT_unnormalize(img_tensor):
    # img_tensor: (C,H,W), normalized by ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img_np = img_tensor.detach().cpu().numpy()
    img_np = img_np * std + mean           # de-normalize (C,H,W)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_hwc = np.transpose(img_np, (1, 2, 0))  # to HWC
    return img_hwc

# Take one sample
CT_one_img, CT_one_label = next(iter(train_loader))
CT_img_vis = CT_unnormalize(CT_one_img[0])
CT_img_unnorm_shape = CT_img_vis.shape

plt.imshow(CT_img_vis)
plt.title(f"Class: {train_dataset.classes[CT_one_label[0]]}")
plt.axis("off")
plt.show()

print("CT_img_unnorm_shape:", CT_img_unnorm_shape)
# -------------------------------------------------------------
# -----------------------------
# 1. Paths
# -----------------------------
data_path = "./data/plant-seedlings"
train_dir = os.path.join(data_path, "train")
val_dir   = os.path.join(data_path, "val")

os.makedirs(val_dir, exist_ok=True)

# -----------------------------
# 2. Create validation split (15% of train)
# -----------------------------
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # All images in this class
    images = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Skip if validation already exists for this class
    val_class_path = os.path.join(val_dir, class_name)
    if not os.path.exists(val_class_path):
        os.makedirs(val_class_path, exist_ok=True)

    # 15% split for validation
    train_imgs, val_imgs = train_test_split(images, test_size=0.15, random_state=42)

    # Move validation images
    for img in val_imgs:
        dest_path = os.path.join(val_class_path, os.path.basename(img))
        shutil.move(img, dest_path)

print(f"Validation split created at: {val_dir}")

# -----------------------------
# 3. Load datasets (train/val/test)
# -----------------------------
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Paths for train, val, test
test_dir = os.path.join(data_path, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 4. Verify splits
# -----------------------------
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
# -------------------------------------------------------------
# Alternative: Create validation split using indices (without moving files)

num_train = len(train_dataset)
CT_perm   = np.random.RandomState(42).permutation(num_train)
CT_val_sz = int(0.15 * num_train)     # 15%
CT_val_idx = CT_perm[:CT_val_sz]
CT_trn_idx = CT_perm[CT_val_sz:]

CT_train_dataset_split = Subset(train_dataset, CT_trn_idx)
CT_val_dataset_split   = Subset(train_dataset, CT_val_idx)

CT_train_loader_split = DataLoader(CT_train_dataset_split, batch_size=32, shuffle=True)
CT_val_loader_split   = DataLoader(CT_val_dataset_split,   batch_size=32, shuffle=False)

# Count class distribution in each split
def CT_count_classes(subset_ds, class_names):
    counts = collections.Counter()
    # subset_ds.indices maps into original dataset.samples (path, label)
    for idx in subset_ds.indices:
        _, lbl = train_dataset.samples[idx]
        counts[class_names[lbl]] += 1
    return dict(counts)

CT_train_class_counts = CT_count_classes(CT_train_dataset_split, train_dataset.classes)
CT_val_class_counts   = CT_count_classes(CT_val_dataset_split,   train_dataset.classes)

print("CT_train_class_counts:", CT_train_class_counts)
print("CT_val_class_counts:",   CT_val_class_counts)
# -------------------------------------------------------------