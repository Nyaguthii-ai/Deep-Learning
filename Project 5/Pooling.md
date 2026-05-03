### Pooling Layers

**Why Pooling?**

When convolutional layers generate feature maps, they preserve most of the original spatial details — every small change in the input image is reflected in these maps. While this is useful for capturing fine details, it quickly becomes inefficient as networks get deeper: feature maps remain large, computations become heavier, and the model risks focusing too much on tiny pixel-level variations.

Pooling layers solve this by **condensing the information** in a feature map. Instead of looking at every pixel, pooling looks at small regions (like `2×2` blocks) and summarizes them — for example, by taking the **maximum value** (max pooling) or the **average value** (average pooling). This achieves two things at once:  

1. It **reduces the size** of the data flowing through the network, making computations faster and reducing the number of parameters in later layers.  
2. It makes the network more **robust to small shifts or distortions** in the input image — if an object moves slightly, the strongest features (like edges or textures) still appear in the pooled map.

In simpler terms, pooling acts like a way of **zooming out**: we keep the most important patterns and ignore small irrelevant variations, helping the network focus on **what** is present rather than **exactly where** it is.


**Types of Pooling**

1. **Max Pooling**
1. **Max Pooling**

Selects the **maximum value** in each window:

$$
\text{MaxPool}(2\times 2) \to \text{take max of each 2x2 block}
$$

2. **Average Pooling**

Takes the **average value** in each window:

$$
\text{AvgPool}(2\times 2) \to \text{average values in each 2x2 block}
$$


**Visual Intuition**

Example: $4 \times 4$ feature map pooled with $2 \times 2$ max pooling.

**Original Feature Map**


$$
\begin{bmatrix}
1 & 2 & 5 & 6 \\
3 & 4 & 7 & 8 \\
9 & 10 & 13 & 14 \\
11 & 12 & 15 & 16
\end{bmatrix}
$$

**After $2 \times 2$ Max Pooling**

$$
\begin{bmatrix}
4 & 8 \\
12 & 16
\end{bmatrix}
$$

Pooling reduces dimensions by a factor of 2 (from $4\times4$ to $2\times2$) while keeping the **strongest activations**.

**Reflection**

- **Max Pooling** preserves the **strongest activations** — useful for detecting whether a feature is present, regardless of exact position.
- **Average Pooling** provides **smoother representations** — less common in modern CNNs but sometimes useful for specific tasks.
- Pooling reduces spatial dimensions, enabling deeper networks without exploding computation.