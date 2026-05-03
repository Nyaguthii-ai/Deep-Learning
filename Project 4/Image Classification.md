### **Image Classification and CIFAR-10**

> **🧠 What Is Image Classification?**

**Image classification** is the process of assigning a **label** to an image based on its content. For example:

- An image of a 🐸 → class: `"frog"`
- An image of a 🚗 → class: `"automobile"`

Unlike previous datasets where each row represented structured features (e.g., age, income), images are **grids of pixel values**. That makes things both interesting and challenging.

> **🗂️ The CIFAR-10 Dataset** : Source: https://www.cs.toronto.edu/~kriz/cifar.html

We’ll use the popular **CIFAR-10** dataset:
- 📦 60,000 color images of shape **32×32 pixels**
- 🎯 10 classes — each image belongs to one of these: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

- 🔀 Split into 50,000 training images and 10,000 test images
- 🎨 Each image has 3 color channels: Red, Green, Blue (RGB)

The small size (32×32) makes CIFAR-10 ideal for quick experiments — but it’s still visually rich and challenging.

**Check Class Distribution**

Before training any model, it’s important to check if the dataset is **balanced** — i.e., roughly an equal number of  samples per class.

Unbalanced datasets can bias the model toward frequent classes and lead to poor performance on rare classes.

Let’s inspect a sample image from the CIFAR-10 training set and walk through the **shape and structure** of image tensors, how images are **converted to PyTorch tensors** and what happens to **pixel values** during this transformation

>**Image Structure: From Pixels to Tensors**

Each CIFAR-10 image is stored as a **3-dimensional array** representing:
- **3 channels**: Red, Green, Blue (RGB)
- Each channel is a **32×32 matrix** of pixel values

So the shape of an image tensor is:  **[Channels, Height, Width] = [3, 32, 32]**

When we use `transforms.ToTensor()`, the raw image pixels — originally in the range **[0, 255]** — are automatically scaled to **[0.0, 1.0]**.  
This is a form of **min-max normalization**, and it's crucial for training neural networks.

> Why Do We Normalize?

- Neural networks learn **faster** when inputs are on a **consistent scale**  
- **Unnormalized data** can cause unstable gradients, especially with large pixel values  
- **Normalized inputs** lead to **smoother convergence** and **better training stability**

> 🧮 **Min-Max Scaling: General Formula**

To scale input features into the range \([0, 1]\), we use the general min-max normalization formula:

$$
x_{\text{scaled}} = \frac{x_{\text{raw}} - x_{\min}}{x_{\max} - x_{\min}}
$$

Where:

- $x_{\text{raw}}$ is the original pixel value  
- $x_{\min}$, $x_{\max}$ are the minimum and maximum possible pixel values

For images with pixel values in the range [0, 255] (which is the case in our current context): $$x_{\text{scaled}} = \frac{x_{\text{raw}} - 0}{255 - 0} = \frac{x_{\text{raw}}}{255}$$

So, a pixel value of 0 becomes 0.0, and 127 becomes approximately 0.498, whereas 255 becomes 1.0.

> Note for CNNs (Later Projects)

In future projects with **CNNs** and **pretrained models**, we’ll often use **standardization** instead of just min-max scaling.  That involves **subtracting the channel-wise mean** and **dividing by the standard deviation**: $$ x_{\text{standardized}} = \frac{x - \mu}{\sigma}$$

Where, $\mu$ is the **mean pixel value** (per channel) and $\sigma$ is the **standard deviation** (per channel). This further **centers the data** and improves **training dynamics** for deep CNNs.

Our CIFAR-10 images are in shape **[3, 32, 32]**, meaning 3 color channels (RGB), each with a 32×32 grid of pixel values. However, basic neural networks (like the MLP we’ll build soon) accept **1D input vectors** — not 2D/3D structures.

Out Goal here is to convert each image from shape: [C, H, W] = [3, 32, 32] to a flat vector of shape: [3072] = 3 × 32 × 32

This process is known as **flattening** or **vectorization**.

**Lets keep in mind that**, flattening **destroys spatial structure** (e.g., neighborhood relationships between pixels). CNNs handle this better — we’ll explore that in later projects. For now, we’re using MLPs, so flattening is necessary.

The `.view(-1)` method in PyTorch reshapes the image tensor to a **1D vector**. This is what we’ll feed into a fully connected layer of an MLP.

Just to share, we can also **reshape** the 1D vector back into an image format: `reshaped = flattened.view(3, 32, 32)`. Useful for debugging or visual checks!

### Prepare DataLoaders — Train, Validation, and Test

Now that our **CIFAR-10** dataset is loaded and normalized, we’ll use **PyTorch DataLoaders** to prepare it for training.

>`torch.utils.data.DataLoader` helps with efficient and flexible data loading by:

- **Batching**: Feeds small groups of images (e.g., 64) at each training step
- **Shuffling**: Randomizes the order of samples each epoch to improve generalization
- **Parallel Loading**: Uses multiple workers to load data faster in the background

CIFAR-10 provides us with: **50,000 training images** and **10,000 test images**

We'll further split the **training set** into **45,000 for training** and **5,000 for validation**

>✅ Why This Split?

- **Training set (45K)**: Used to train the model (weights updated here)
- **Validation set (5K)**: Used during training to monitor performance and detect overfitting
- **Test set (10K)**: Held out until the end for final unbiased evaluation

> Using separate validation and test sets allows us to monitor model behavior during training without contaminating our final performance estimate.

### Define a Simple MLP for CIFAR-10

We’ll now define our **first neural network** — a basic MLP (Multilayer Perceptron).

>MLP Structure

We flattened each 32×32 RGB image into a **3072-dimensional vector**.

Here’s our architecture:

- **Input layer**: 3072 units (32×32×3)
- **Hidden layer**: 128 units + ReLU
- **Output layer**: 10 units (logits, one per class)

This is a **fully connected network** — each neuron connects to every unit in the next layer.

We start with MLP to understand baseline performance — **before introducing CNNs** (which are better suited for images).

>What Does This Model Do?

- First `Linear` layer transforms 3072 → 128
- `ReLU` adds non-linearity to learn complex patterns
- Final `Linear` outputs 10 logits, which are used by CrossEntropyLoss

>Important Notes

- We don’t use `Softmax` in the model — because `nn.CrossEntropyLoss` internally applies it.
- We'll **track both training loss and accuracy** as we proceed.