### Convolution Neural Networks

CNNs introduce three powerful ideas:

1. **Local Receptive Fields**: Rather than looking at the entire image at once, CNNs use small “windows” (like `3×3` or `5×5` grids) that **scan small parts of the image at a time**. Each window focuses on **local patterns**, such as detecting edges, corners, or simple textures. This is similar to how our eyes process only a small patch of the visual field at a time and then combine information from multiple patches.

2. **Parameter Sharing**: The same filter (set of weights) is **reused** as it slides across different parts of the image. For example, a filter that detects horizontal edges will work anywhere in the image — **top, bottom, or center**. This dramatically reduces the number of parameters compared to an MLP and ensures the model **remembers what it learned about an edge regardless of location**.

3. **Hierarchical Feature Learning**: CNNs build knowledge **layer by layer**:
   - Early layers detect **simple features** (e.g., edges, corners).
   - Middle layers detect **patterns or textures** (e.g., fur, eyes).
   - Deeper layers detect **complex objects** (e.g., faces, entire animals).

This hierarchy allows CNNs to construct an understanding of the image **from simple to complex**, mirroring how humans perceive visual information.

### 🐾 Image Representation and Dataset

- Images are stored as **tensors**: multi-dimensional arrays of numbers.
- In PyTorch, the format is usually **C × H × W**:
  - **C** = Number of channels (e.g., 3 for RGB, 1 for grayscale)
  - **H** = Height (number of rows)
  - **W** = Width (number of columns)

For example, a **32×32 RGB image** will have 3 channels (Red, Green, Blue), each channel having a 32×32 grid, and thus total shape: `3 × 32 × 32`.

Each pixel in a color image has **three values**: R (Red) intensity, G (Green) intensity, B (Blue) intensity.

Pixel values in the raw dataset range from 0 to 255. When we apply the `ToTensor()` transform provided by `torchvision`, these values are converted to floats in the range `0.0–1.0`. Later in the pipeline, we often apply **normalization** using the mean and standard deviation of the dataset to center and scale the pixel values, which helps with training stability.

**Toy Example: RGB Patch**

Imagine a **$3 \times 3$ patch** of an image (zoomed in):

**Red Channel ($3 \times 3$)** <--> **Green Channel ($3 \times 3$)** <--> **Blue Channel ($3 \times 3$)**


\begin{bmatrix}
123 & 50 & 90 \\
100 & 60 & 120 \\
90 & 30 & 60
\end{bmatrix}

\begin{bmatrix}
34 & 200 & 80 \\
20 & 150 & 70 \\
15 & 180 & 50
\end{bmatrix}

\begin{bmatrix}
90 & 10 & 255 \\
80 & 40 & 210 \\
60 & 30 & 190
\end{bmatrix}


When combined, these three matrices reconstruct the **color pixel information** for that patch.

**Why Does This Matter for CNNs?**: **Convolutional filters** will operate across **all channels** simultaneously. The way images are stored ($C \times H \times W$) directly affects how we design convolution layers. Understanding this format is crucial before we discuss **filters, strides, and feature maps**.

**Visual Comparison: Original vs Resized**

To better understand why we resize images for training:

- **Original image**: High resolution, varied sizes, more details.
- **Resized image (64×64)**: Standardized, smaller, faster to process.

We trade off some detail for **training speed and consistency**.

**Why Convolution Instead of Fully Connected Layers?**

In an image, meaningful features like edges, corners, or textures are **local** — we don’t need to connect every pixel to every neuron to detect them. Instead, we scan small regions (patches) of the image with **filters** (also called **kernels**) that specialize in finding specific patterns (e.g., vertical edges, diagonal lines, color gradients).

**How Does It Work?**

A convolution layer slides a **filter (kernel)** across the image. At each position, it performs **element-wise multiplication** followed by a **sum** to produce a single output value. Repeating this across the image forms the **feature map**.

Mathematically, a 2D convolution is:

$$
S(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)
$$

where:
- $ I $= input image  
- $ K $ = convolution filter  
- $ S $ = resulting feature map  

**Key Hyperparameters**

1. **Filter Size (Kernel Size)**  
   - Typically $3 \times 3$ or $5 \times 5$.  
   - Larger filters capture bigger patterns but are more computationally expensive.

2. **Stride**  
   - How far the filter moves per step.  
   - Stride = 1: maximum overlap. Stride = 2: skips pixels (downsampling).

3. **Padding**  
   - Adds zeros around the border to preserve image size after convolution.  
   - Without padding, feature maps shrink after each layer.


**Why This Matters**

- **Parameter Efficiency:** A $3 \times 3$ filter uses 9 parameters but scans the whole image.  
- **Translation Invariance:** A detected edge is recognized regardless of its location.  
- **Hierarchical Learning:** Early layers detect edges → middle layers detect textures → deeper layers recognize objects.


**Example: Edge Detection**

A simple horizontal edge filter:

$$
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

This highlights **horizontal changes** in intensity — transitions from dark to light.

**Why Multi-Channel Convolutions?**

Up to now, we’ve worked with **single-channel (grayscale)** images, where each pixel is represented by one intensity value.  
However, real-world images (like those in Oxford Pets) are **RGB** — meaning **3 channels** (Red, Green, Blue).

- Each channel is processed **independently** by the filter (kernel).
- The results are then **summed across channels** to form the final feature map.


**Mathematical Representation**

For a convolution filter with weights $W_R$, $W_G$, and $W_B$ applied to an RGB image:

$$
\text{Feature Map} = (W_R * I_R) + (W_G * I_G) + (W_B * I_B)
$$

Where:
- $I_R, I_G, I_B$ are the red, green, and blue channels of the image.
- $*$ denotes the 2D convolution operation.


**Multiple Filters → Multiple Feature Maps**

If we apply **N different filters**, we get **N feature maps**.  
Each map captures **different patterns** (e.g., vertical edges, horizontal edges, textures).

- Each filter learns to detect **specific features**.
- Stacking feature maps lets the network represent complex structures.

- By applying **different filters**, we generated **different feature maps**:
  - Horizontal edges → boundaries running left-to-right.
  - Vertical edges → boundaries running top-to-bottom.
- In CNNs, dozens (or even hundreds) of filters operate in parallel:
  - Capturing **complementary features** that, combined, form a rich understanding of the image.
- These feature maps are then passed to **pooling layers** (next section) to reduce size while keeping the most relevant patterns.

### Hierarchical Feature Learning

Up to now, we’ve explored the **building blocks** of CNNs: convolutions, multi-channel processing, and pooling. But the real power of CNNs emerges when we **stack these layers** — each layer takes the features detected by the previous one and combines them into something more complex. This creates a **hierarchy of features**, where simple patterns evolve into sophisticated representations of the input.

**How Does This Hierarchy Work?**

- **Early layers (closest to the raw image):**  
  These layers learn to detect the simplest building blocks — edges, lines, and basic color contrasts. This is exactly what we saw with our manual edge filters. At this stage, the network doesn’t “know” what an ear or paw is; it’s only identifying basic shapes and boundaries.

- **Middle layers:**  
  As we stack more convolutions, these edge detectors combine to form **textures and small patterns** — like fur, eyes, or whisker arrangements. Importantly, these are **not predefined by us**. The network *learns* which combinations of edges are useful for distinguishing classes during training.

- **Deeper layers:**  
  Further combinations form **larger parts** of the object — like ears, snouts, or tails. These parts are more abstract: the network begins to recognize “this curved edge + texture = ear,” even if the ear appears in different positions or sizes.

- **Final layers:**  
  The deepest layers integrate everything into **whole-object representations** (e.g., “This collection of textures and parts is a Persian cat”). By this point, the feature maps no longer resemble edges or textures — they encode high-level concepts useful for classification.

**Where Does Pooling Fit In?**

Pooling layers are interspersed between convolution layers to **gradually reduce spatial detail** while retaining the most important features. Think of it as zooming out: early layers focus on precise edge positions, while later layers care about *what* features exist rather than *exactly where* they are. This combination of convolution + pooling allows CNNs to be **both detailed and flexible**.

**Why Is This Powerful?**

- This hierarchy mimics **how human vision works** — from detecting edges to recognizing complex objects.  
- We don’t manually design “ear detectors” or “fur detectors.” The network learns what’s useful based on the training data.  
- Each layer **builds on previous layers**, which is why depth (stacking layers) makes CNNs capable of solving complex visual tasks.

### Summary

In this notebook, we explored the **foundational building blocks of CNNs** — understanding not just how they work, but why they are so effective for images.

**Convolutional Layers**  
We saw how convolutions use small filters to detect **local patterns** (like edges) while preserving the **spatial structure** of images. Parameter sharing makes this approach far more efficient than MLPs, which flatten images and lose spatial relationships.

**Multi-Channel Convolutions**  
Real images are RGB, so each filter operates across **all three channels** and sums the results to form a feature map. Stacking multiple filters lets the network detect a variety of patterns — for example, some filters may learn horizontal edges, others vertical edges, and others more complex textures.

**Pooling Layers**  
Pooling layers act as a way to **summarize information**: they downsample feature maps and make the network more robust to small shifts (translation invariance). This helps later layers focus on *what* features exist rather than *exactly where* they are.

**Hierarchical Feature Learning**  
By stacking convolution + pooling layers, CNNs build a **feature hierarchy**:  
- Early layers learn edges and color contrasts.  
- Middle layers combine these into textures and patterns.  
- Deeper layers recognize object parts (like ears or snouts).  
- Final layers put these parts together to recognize whole objects (e.g., specific dog or cat breeds).  

This progression is not manually designed — it emerges naturally during training, which is what makes CNNs so powerful.


```text
      Input Image (3×H×W)
               │
               ▼
   Convolution + Pooling Layers
      (edges → textures → parts)
               │
               ▼
    Fully Connected Layers (FC)
      (combine all features)
               │
               ▼
   Output (Breed Prediction)
```