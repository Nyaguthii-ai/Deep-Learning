# **Convulation Neural Networks**

>🚫 Why MLPs Struggle with Image Data

🔹 1. **MLPs Ignore Spatial Structure**: Images are not just collections of pixels — they have **spatial patterns**. Neighboring pixels form edges, textures, and shapes. But flattening an image from `[3 × 32 × 32]` to `[3072]` destroys that structure. A pixel from the top-left is treated the same as one from the bottom-right. Thus, the model **loses all sense of locality** and **spatial hierarchy**.

🔹 2. **Too Many Parameters**: An MLP connects **every input to every neuron**: For CIFAR-10: `3072 inputs × 512 hidden units = 1.5 million weights` — in just one layer! This makes the network slower to train, heavier to store and prone to **overfitting**

🔹 3. **No Translation Invariance**: If an object (say, a cat) shifts slightly within an image, an MLP treats it as a completely different pattern. It has **no ability to generalize** across positions.

>**✅ What CNNs Do Differently**

Convolutional Neural Networks (CNNs) are designed for **structured grid-like data** like images.

They bring several key benefits:

| 🚫 **MLP Limitation**              | ✅ **How CNN Solves It**                                                                 |
|-----------------------------------|------------------------------------------------------------------------------------------|
| **Spatial structure is lost**     | MLP flattens image to 1D, ignoring pixel neighborhoods. <br> 🔍 CNN uses **2D filters** that slide over the image to detect local patterns like edges and corners. |
| **Too many parameters**           | Fully connected layers in MLP require one weight per pixel per neuron. <br> 🔁 CNNs use **weight sharing** — the same filter is applied across the image — drastically reducing the number of parameters. |
| **No translation invariance**     | MLPs treat a "cat on the left" and "cat on the right" as totally different inputs. <br> 🌀 CNNs use **pooling layers** (e.g., max pooling) to retain key features regardless of exact position. |

**🧱 Building Blocks of CNNs (Preview)**

| Component     | What It Does                                      |
|---------------|---------------------------------------------------|
| `Conv2d`      | Applies learned filters over spatial regions      |
| `ReLU`        | Adds non-linearity (like in MLPs)                 |
| `MaxPool2d`   | Downsamples feature maps, adds invariance         |
| `Linear`      | Final dense layers for classification             |

In earlier notebooks, we **flattened the image tensors** to feed them into MLPs.

This time, we’ll preserve the **2D structure** of images — exactly what CNNs are designed for.

>🔄 Preprocessing Steps

We’ll define a standard transform pipeline using `torchvision.transforms`:

| Step             | Why It Matters                                      |
|------------------|-----------------------------------------------------|
| `ToTensor()`     | Converts PIL image to tensor & scales pixel values to `[0, 1]` |
| `Normalize()`    | Standardizes each channel using dataset-specific mean and std |

</br>

>Normalization Stats for CIFAR-10

We use the **empirical mean and standard deviation** computed over the training set:

```python
mean = [0.4914, 0.4822, 0.4465]  # RGB means
std  = [0.2470, 0.2435, 0.2616]  # RGB std devs
```

Using dataset-specific stats helps with:

- Smoother gradient flow during training
- Faster convergence
- Avoiding internal covariate shift (especially before BatchNorm)

>📐 Convolution: A Numerical Example

Let’s say we have a **3×3 image patch** and a **3×3 filter**:

```text
[ [1, 2, 0],
  [3, 1, 1],
  [0, 2, 4] ]
```
Filter (**learned by the CNN**):
```text
[ [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1] ]
```
👉 To compute the convolution output at this location, we do an element-wise product and sum all the results:
```text
(1×1)+(2×0)+(0×−1)+(3×1)+(1×0)+(1×−1)+(0×1)+(2×0)+(4×−1) = 1+0+0+3+0−1+0+0−4 = −1
```
✅ So the output of the convolution at this location is −1.

This filter will now slide over the image to compute more outputs, forming a feature map.

>🧮 Convolution Illustrated with a Larger Image

Let’s say we have a **5×5 grayscale image** and a **3×3 filter**. We want to apply **valid padding** (no zero-padding) and a **stride of 1**.

**🖼️ Input Image (5×5)**

```text
[ [ 1, 2, 3, 0, 1 ],
  [ 0, 1, 2, 3, 1 ],
  [ 3, 1, 0, 2, 2 ],
  [ 1, 2, 1, 0, 0 ],
  [ 0, 1, 3, 2, 1 ] ]
```
**Filter (3×3)**
```text
[ [ 1, 0, -1 ],
  [ 1, 0, -1 ],
  [ 1, 0, -1 ] ]
```
**▶️ Step-by-Step Convolution**

We slide the filter across the image. At each position, we compute a dot product between the filter and the 3×3 patch of the image.

🔹 Example: Top-left corner

Apply the filter on the top-left 3×3 patch:

```text
[ [1, 2, 3],
  [0, 1, 2],
  [3, 1, 0] ]
```
Dot product:
```text
(1×1) + (2×0) + (3×−1) + (0×1) + (1×0) + (2×−1) + (3×1) + (1×0) + (0×−1) = 1 + 0 −3 + 0 + 0 −2 + 3 + 0 + 0 = −1
```

✅ Output at this position = −1

We repeat this across the whole image using a stride of 1.

>🧮 Output Feature Map (3×3)

With a 5×5 input and 3×3 filter, no padding, and stride 1, output size is:

$$
(5 - 3 + 1) \times (5 - 3 + 1) = 3 \times 3
$$

Each element in the output is a result of the dot product at a different position.

>📉 Why the Output is Smaller

With **valid padding**, output size is calculated as:

$$
\text{Output Size} = \frac{(N - F)}{\text{stride}} + 1
$$

Where:

- $N = $ input size (e.g., 5)
- $F = $ filter size (e.g., 3)
- Stride = 1

So:
$$
\text{Output Size} = \frac{(5 - 3)}{1} + 1 = 3
$$

🧠 This reduction allows the network to **compress** and **abstract** spatial information layer by layer.

>🎯 What the Filter Learns

- In the above example, the filter “activates” when it sees for example **vertical edges** — strong contrast between left and right.
- In deep learning, filters are **learned during training** — not hand-coded.

This is how CNNs extract **low-level features** like edges, which are then combined into **higher-level features** in deeper layers.

>**🌀 2. Activation: ReLU**

After convolution, we apply a **non-linear activation**.

Why?
- To break linearity and let the network learn **complex patterns**.
- ReLU replaces negative values with zero:
  
  $$
  \text{ReLU}(x) = \max(0, x)
  $$

>**🔽 3. Max Pooling**

Max Pooling reduces the size of feature maps.

It keeps the **strongest activation** in each small window (like 2×2), helping with:
- Dimensionality reduction
- Translation invariance
- Faster computation

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png" width="300">

>**🧠 Summary**

| Component     | Purpose                                 |
|---------------|------------------------------------------|
| `Conv2d`      | Detects local patterns in image patches |
| `ReLU`        | Enables non-linear transformations      |
| `MaxPool2d`   | Reduces size, adds robustness           |
| `Linear`      | Final decision-making (classification)  |

---
### LeNet Architecture 
LeNet-5 was developed by **Yann LeCun et al. in 1998** to classify **handwritten digits** (e.g., from the MNIST dataset). It was one of the **first successful convolutional neural networks (CNNs)** and laid the foundation for modern deep learning architectures used in computer vision.

LeNet introduced **three revolutionary ideas**:

1. ✅ **Local Receptive Fields**: Just like our eyes focus on small patches of a scene, LeNet looked at **local patches** of the image using **convolutional filters**.

2. ✅ **Shared Weights**: A single filter slides across the entire image. This means:
   - Fewer parameters
   - Detects the **same pattern** (e.g., vertical edge) *anywhere* in the image

3. ✅ **Hierarchical Feature Learning**: LeNet learned **low-level patterns** (like edges) in early layers and **high-level patterns** (like shapes) in deeper layers — just like human vision!

>**🧱 Original LeNet-5 Architecture (for 32×32 grayscale inputs)**

```text
Input (32×32×1 grayscale image)
  ↓
C1: Convolution (6 filters, 5×5 kernel) → Output: 28×28×6
  ↓
S2: Average Pooling (2×2) → Output: 14×14×6
  ↓
C3: Convolution (16 filters, 5×5 kernel) → Output: 10×10×16
  ↓
S4: Average Pooling (2×2) → Output: 5×5×16
  ↓
C5: Convolution (120 filters, 5×5) → Output: 1×1×120
  ↓
Flatten → Fully Connected Layer (84 units)
  ↓
Output Layer (10 units for classification)
```
>📊 LeNet Shape Transformations (Step-by-Step)

| Layer   | Input Shape  | Output Shape | Notes                                 |
|-------- |--------------|--------------|----------------------------------------|
| Input   | `32×32×1`    | —            | Grayscale image (MNIST-style input)    |
| Conv1   | `32×32×1`    | `28×28×6`    | 6 filters, 5×5 kernel, stride=1        |
| Pool1   | `28×28×6`    | `14×14×6`    | 2×2 **average pooling**                |
| Conv2   | `14×14×6`    | `10×10×16`   | 16 filters, 5×5 kernel, stride=1       |
| Pool2   | `10×10×16`   | `5×5×16`     | 2×2 **average pooling**                |
| Conv3   | `5×5×16`     | `1×1×120`    | 120 filters, 5×5 kernel                |
| Flatten | `1×1×120`    | `120`        | Flatten to 1D vector                   |
| FC1     | `120`        | `84`         | Fully connected layer                  |
| Output  | `84`         | `10`         | Output logits (10 classes)            |

>**🔍 Clarifications: How Do C1, C3, and C5 Work?**

**✅ Conv1:**
- Input: `32×32×1` (grayscale)
- Kernel: `5×5`, stride 1 → Output spatial size = `32 - 5 + 1 = 28`
- 6 filters → Output: `28×28×6`

**✅ Conv2 (sometimes labeled C3 in classic LeNet):**
- Input: `14×14×6`
- Each of the 16 filters has size `6×5×5` (depth spans all input channels)
- Output spatial size = `14 - 5 + 1 = 10`
- 16 filters → Output: `10×10×16`

**✅ Conv3 (sometimes labeled C5):**
- Input: `5×5×16`
- Each filter has size `16×5×5` (spanning depth + full spatial region)
- Output: `1×1×120` — like a 1x1 spatial grid with 120 feature maps
- This is effectively a **fully connected layer** implemented via convolution

📌 So, each Conv layer learns **increasingly abstract features**, while pooling reduces spatial size and increases robustness to shifts.

> 📌 **Note on Input Size**

Although MNIST images are 28×28, LeNet-5 was originally designed for **32×32 grayscale images**. The MNIST digits were **zero-padded** to 32×32 before being fed into the model. This was done to:

- Preserve edge information,
- Allow multiple convolution + pooling layers without shrinking the image too quickly.

🧠 So when we describe "original LeNet" in architectural diagrams, we usually start from 32×32 input — even if the raw MNIST data was 28×28.

> CIFAR-10 already comes as 3×32×32 RGB images, so it's a perfect fit for the adapted LeNet architecture — with adjustments for:
>- 3 channels instead of 1,
>- Slightly more filters,
>- ReLU instead of Tanh,
>- MaxPooling instead of AveragePooling.

➡️ Activation functions used were mainly Tanh, and pooling was average pooling, which differs from modern practices (ReLU and max pooling).

>**⚠️ Why We Need to Modify LeNet for CIFAR-10**

CIFAR-10 is more complex than MNIST:
- Color images (3 channels),
- More object variability (10 diverse categories),
- Natural scenes instead of centered digits.

So we must adapt LeNet by:
- Changing the input channels from 1 to 3.
- Increasing the number of filters slightly to handle color and variability.
- Using ReLU activations instead of Tanh.
- Replacing average pooling with max pooling (standard in modern CNNs).

> **🛠️ Adapted LeNet Architecture for CIFAR-10**

Here’s the modified architecture we'll implement:

```text
Input: 3×32×32 image
  ↓
Conv1: 3→6 filters, 5×5 kernel → ReLU → MaxPool (2×2)
  ↓
Conv2: 6→16 filters, 5×5 kernel → ReLU → MaxPool (2×2)
  ↓
Flatten
  ↓
FC1: 400 → 120 → ReLU
  ↓
FC2: 120 → 84 → ReLU
  ↓
Output: 84 → 10 (logits)
```
>📊 Adapted LeNet for CIFAR-10 — Shape Transformations

| Layer   | Input Shape   | Output Shape | Notes                                      |
|-------- |---------------|--------------|--------------------------------------------|
| Input   | `32×32×3`     | —            | RGB image (CIFAR-10)                       |
| Conv1   | `32×32×3`     | `28×28×6`    | 6 filters, 5×5 kernel, stride=1            |
| Pool1   | `28×28×6`     | `14×14×6`    | 2×2 max pooling                            |
| Conv2   | `14×14×6`     | `10×10×16`   | 16 filters, 5×5 kernel, stride=1           |
| Pool2   | `10×10×16`    | `5×5×16`     | 2×2 max pooling                            |
| Flatten | `5×5×16`      | `400`        | Flatten 3D tensor to 1D vector             |
| FC1     | `400`         | `120`        | Fully connected, followed by ReLU          |
| FC2     | `120`         | `84`         | Fully connected, followed by ReLU          |
| Output  | `84`          | `10`         | Output logits for 10 CIFAR-10 classes      |

</br>

>📌 This structure keeps the essence of LeNet but modernizes it for CIFAR-10. It remains relatively shallow, helping us:
- Learn CNN fundamentals,
- Train quickly,
- Establish a baseline for comparison with deeper networks.

### AlexNet Mini – A Deeper CNN

LeNet showed us that:

- Convolutions extract spatial patterns (edges, textures).
- Pooling reduces dimensionality while keeping key features.
- Even a **shallow CNN** outperforms MLPs on image tasks.

But CIFAR-10 is **much more complex** than MNIST:

- It has **color images** (3 channels instead of 1)
- Contains **more varied categories** (e.g., airplane, dog, ship, frog)
- Includes **diverse visual patterns and backgrounds**

To handle this complexity, we need **deeper networks** that learn **hierarchical features** — from simple edges to complex object parts.

>**📚 Enter AlexNet (2012)**

AlexNet made deep CNNs mainstream. It won the **ImageNet 2012 competition** by a huge margin and introduced:

- **More convolutional layers** for deeper abstraction  
- **Larger filter banks** to boost capacity  
- **ReLU** activations (faster training than Tanh)  
- **Dropout** for regularization  
- **GPU training** for scale


>🧠 Is LeNet Really That Shallow?

Yes — though technically it has **3 convolutional layers**, the final one (C5) acts like a **fully connected layer** (5×5 kernel on 5×5 input = 1×1 output).

So:
- It has **2 spatial convolution layers** + **1 conv-style FC layer**.
- Our **adapted LeNet** follows the same logic using PyTorch:  
  We flatten after the second pooling layer and use a dense layer instead of a 1×1 convolution.

>**🔄 Differences from LeNet**

| Feature               | LeNet (Adapted)     | AlexNet Mini            |
|-----------------------|---------------------|--------------------------|
| Input Size            | 32×32 RGB           | 32×32 RGB               |
| # Convolutional Layers| 2                   | 3                       |
| Filters per Layer     | 6 → 16              | 64 → 128 → 256          |
| Activation            | ReLU                | ReLU                    |
| Pooling               | 2×2 MaxPool         | 2×2 MaxPool             |
| Dropout               | ❌ Not used         | ✅ Yes                  |
| Output Head           | 400 → 120 → 10      | 256 → 128 → 10          |

</br>

>🧠 Why AlexNet Mini?

The original AlexNet was designed for **ImageNet** with **224×224 images** and millions of parameters. That’s **too heavy** for CIFAR-10.

So we use a **miniaturized version** of AlexNet, adapted for:

- Small image size (32×32)
- Lower memory needs
- Faster training (even on CPU/GPU)

>🧱 Architecture Summary

```text
Input (3×32×32)
↓
Conv1: 3→64, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Conv2: 64→128, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Conv3: 128→256, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Flatten
↓
Linear: 256 → 128 → ReLU → Dropout
↓
Linear: 128 → 10 → Output logits
```
**AlexNetMini Shape Transformation**

| Layer     | Input Shape     | Output Shape    | Notes                                    |
|-----------|------------------|------------------|------------------------------------------|
| Input     | `3×32×32`        | —                | RGB image                                |
| Conv1     | `3×32×32`        | `64×30×30`       | 64 filters, 3×3 kernel, stride=1         |
| MaxPool1  | `64×30×30`       | `64×15×15`       | 2×2 max pooling, stride=2                |
| Conv2     | `64×15×15`       | `128×13×13`      | 128 filters, 3×3 kernel, stride=1        |
| MaxPool2  | `128×13×13`      | `128×6×6`        | 2×2 max pooling, stride=2                |
| Conv3     | `128×6×6`        | `256×4×4`        | 256 filters, 3×3 kernel, stride=1        |
| MaxPool3  | `256×4×4`        | `256×2×2`        | 2×2 max pooling, stride=2                |
| Flatten   | `256×2×2`        | `1024`           | Flatten to vector (256×2×2 = 1024)       |
| Linear1   | `1024`           | `128`            | Fully connected layer → ReLU → Dropout   |
| Output    | `128`            | `10`             | Output logits for 10 classes             |

</br>

**AlexNetMini Training: Too Slow?**

While AlexNetMini is powerful, training it on CIFAR-10 can be time-consuming, especially on limited hardware.

So, let’s define a Lite version that: 
-    Trains faster ⏱️
-    Uses fewer filters and fewer fully connected neurons
-    Still improves over LeNet — just not as much as full AlexNetMini

>**Architecture Summary**
```text
Input: 3×32×32 image
↓
Conv1: 3 → 32 filters, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Conv2: 32 → 64 filters, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Conv3: 64 → 128 filters, 3×3 kernel → ReLU → MaxPool (2×2)
↓
Flatten
↓
Dense (2048 → 256) → ReLU
↓
Output Layer (256 → 10)
```

>**AlexNetMiniLite Shape Transformation**

| Layer     | Input Shape   | Output Shape   | Notes                              |
|-----------|---------------|----------------|------------------------------------|
| Input     | `3×32×32`     | —              | CIFAR-10 RGB image                 |
| Conv1     | `3×32×32`     | `32×32×32`     | 32 filters, 3×3 kernel, padding=1  |
| MaxPool1  | `32×32×32`    | `32×16×16`     | 2×2 pooling, stride=2              |
| Conv2     | `32×16×16`    | `64×16×16`     | 64 filters, 3×3 kernel, padding=1  |
| MaxPool2  | `64×16×16`    | `64×8×8`       | 2×2 pooling, stride=2              |
| Conv3     | `64×8×8`      | `128×8×8`      | 128 filters, 3×3 kernel, padding=1 |
| MaxPool3  | `128×8×8`     | `128×4×4`      | 2×2 pooling, stride=2              |
| Flatten   | `128×4×4`     | `2048`         | Flattened vector (128×4×4)         |
| FC1       | `2048`        | `256`          | Fully connected + ReLU             |
| Output    | `256`         | `10`           | Final logits (10 classes)          |

</br>

This architecture:
-    Uses 3 convolutional layers with increasing filter depths (32 → 64 → 128).
-    Applies MaxPooling after each convolution to reduce spatial dimensions.
-    Ends with one hidden dense layer before the output logits.
-    Has much fewer parameters than full AlexNetMini, making it faster to train.

>🧮 Parameter Count Comparison

| Model           | Approx. Parameters | Notes                          |
|-----------------|--------------------|--------------------------------|
| MLP             | ~393,000           | Fully connected only           |
| LeNet (CIFAR-10)| ~61,000            | Very small and fast            |
| AlexNetMini     | ~2.5 million       | Deeper, heavier CNN            |
| AlexNetMiniLite | ~300,000           | Lighter, faster to train       |

</br>

> ℹ️ Actual counts may vary slightly based on implementation. You can use `torchsummary` or `torchinfo` to get exact values.

