# Convolution Neural Networks using PyTorch

**Transforms and Normalization**

**1. Resize & ToTensor**

- Convert variable-size images to a fixed **128×128** shape.
- Convert to PyTorch tensors (values in `[0,1]`).

**2. Normalization**

- Normalization centers the data and scales it to unit variance:
  $$
  x' = \frac{x - \mu}{\sigma}
  $$
- For CNNs, we normalize **per channel (R, G, B)** using dataset statistics (mean & std).
- This differs from earlier MLP examples, where we normalized **after flattening** (treating all pixels as one vector).

**Data Splits**

- The Oxford Pets dataset provides **`trainval`** and **`test`** splits.
- We will:
  - Use `trainval` for **training & validation** (split internally).
  - Use `test` for **final evaluation**.

**Baseline CNN Structure**

We will design a **simple yet powerful CNN** to classify the 37 pet breeds:

- **2 Convolution Layers** (extract low- and mid-level features)
- **ReLU Activation** (introduce non-linearity)
- **Max Pooling** (downsample feature maps)
- **Flatten Layer** (convert 3D feature maps to 1D vector)
- **2 Fully Connected Layers (FC)** (combine features for classification)
- **Output Layer (Softmax)** (37-class probability distribution)

**Architecture Overview (ASCII)**

```text
Input: 3 × 128 × 128
│
├── Conv1: 16 filters (3×3) → ReLU → MaxPool (2×2)
│
├── Conv2: 32 filters (3×3) → ReLU → MaxPool (2×2)
│
├── Flatten → FC1 (128 units) → ReLU
│
└── FC2 (37 units) → Softmax
```

**Assumptions for Shape Calculations**

- **Kernel size:** $3 \times 3$ (both Conv layers)  
- **Padding:** $1$ (keeps spatial size same after convolution)  
- **Stride (Conv):** $1$  
- **Pooling:** $2 \times 2$ max pooling (halves spatial dimensions)  

**Formula for Convolution Output Size**

For an input with spatial dimension $H \times W$:

$$
H_{out} = \frac{H + 2P - K}{S} + 1, \qquad
W_{out} = \frac{W + 2P - K}{S} + 1
$$

Where:
- $K$ = kernel size
- $P$ = padding
- $S$ = stride

For pooling ($2 \times 2$, stride 2):

$$
H_{out} = \frac{H}{2}, \qquad W_{out} = \frac{W}{2}
$$


**Shape Transformations (Step-by-Step)**

| Layer                           | Input Shape    | Output Shape   | Operation Details                    |
| ------------------------------- | -------------- | -------------- | ------------------------------------ |
| **Input**                       | $3 \times 128 \times 128$  | $3 \times 128 \times 128$  | Raw RGB image                        |
| **Conv1 (3→16, k=3,p=1,s=1)**   | $3 \times 128 \times 128$  | $16 \times 128 \times 128$ | Computed using formula: $128 = \frac{128 + 2(1) - 3}{1} + 1$ |
| **MaxPool1 (2×2)**              | $16 \times 128 \times 128$ | $16 \times 64 \times 64$   | Downsample: $128 / 2 = 64$           |
| **Conv2 (16→32, k=3,p=1,s=1)**  | $16 \times 64 \times 64$   | $32 \times 64 \times 64$   | Computed similarly with padding 1    |
| **MaxPool2 (2×2)**              | $32 \times 64 \times 64$   | $32 \times 32 \times 32$   | Downsample: $64 / 2 = 32$            |
| **Flatten**                     | $32 \times 32 \times 32$   | $32,768$                     | Convert 3D tensor to 1D vector       |
| **FC1 (32,768→128)**            | $32,768$                     | $128$                        | Combine features, apply ReLU         |
| **FC2 (128→37)**                | $128$                        | $37$                         | Class logits for 37 breeds           |

<br>

> **Why This Architecture?**

**Progressive Feature Learning**
- **First Conv layer:** captures edges and colors.
- **Second Conv layer:** captures textures and patterns (e.g., fur details).

**Pooling Layers**
- Reduce computational cost and add translation invariance.

**Fully Connected Layers**
- Integrate high-level features for final classification.


> **CNN vs MLP**

- **MLP:** Flattens input immediately → loses spatial information.  
- **CNN:** Preserves spatial structure → fewer parameters (weight sharing).

**Parameter Comparison (approx.)**

- **MLP on $128 \times 128$:** ~6M parameters (fully connected from start).  
- **CNN (2 conv layers + FC):** ~4M parameters, yet higher accuracy due to spatial features.

### Implementing the CNN (PyTorch Code)

**From Concept to Code**

We now translate our conceptual architecture into PyTorch:

- **Class-based model (`nn.Module`)**: Main approach — flexible, industry standard, good for learning `forward()` flow.
- **Sequential alternative (`nn.Sequential`)**: Concise version for straightforward feedforward networks.

**Parameter Awareness**

CNNs are **parameter-efficient** because:
- Convolutional filters are **shared across spatial locations** (few parameters).
- Fully connected layers dominate parameter count in small CNNs.

Example (128×128 input):
- After 2 conv + pool layers → feature map is `32×32×32 = 32,768` features.
- FC1 reduces `32,768 → 128` (≈ 4.2M params) — still far fewer than a pure MLP (~6M params).

**CNN vs MLP Parameter Count ($128\times128$ Input)**

Let’s estimate parameter counts if we built an **MLP** instead of this CNN:

- MLP scenario (eg., Input -> FC -> FC -> Output)

    - **Input size** = $3 \times 128 \times 128 = 49,152$ features  
    - **First FC layer (49,152 → 256 units):**

    $$
    49,152 \times 256 + 256 \approx 12.6\ \text{million parameters}
    $$

    - Fully connected layers dominate the model size.

- CNN scenario (this notebook)

    - After two Conv+Pool blocks:

    $$
    32 \times 32 \times 32 = 32,768 \text{ features}
    $$

    - **FC1 (32,768 → 128 units):**

    $$
    32,768 \times 128 + 128 \approx 4.2\ \text{million parameters}
    $$

    - Conv layers add only $\sim 13K$ parameters (tiny compared to FC).

        ```text
        MLP: 49,152 ──► FC ──► FC ──► Output (≈12M params)

        CNN: 3×128×128 ──► Conv+Pool ──► Conv+Pool ──► FC ──► Output (≈4M params)
        ```

- CNN is $\sim 3\times$ smaller in parameters while capturing **spatial patterns**.  
- This efficiency allows CNNs to handle **larger inputs** (e.g., $224\times224$ in ImageNet) without massive memory costs.

---
### Loss Function & Optimizer

**Why Cross-Entropy Loss?**

- Our task is **multiclass classification** (37 pet breeds).  
- We need a loss function that:
  - Compares predicted **class probabilities** with **true labels**.
  - Encourages high probability for the correct class and low for others.

**Cross-Entropy Formula:**

$$
\text{Loss} = - \sum_{c=1}^{C} y_c \log(\hat{p}_c)
$$

Where:
- $ y_c $ = 1 for correct class, 0 otherwise (one-hot target).
- $ \hat{p}_c $ = predicted probability for class $ c $.

**Important Detail (Softmax + Logits)**

- **`nn.CrossEntropyLoss` in PyTorch:**
  - Expects **raw logits** (no softmax applied manually).
  - Internally applies `LogSoftmax` + `NLLLoss`.
  - More **numerically stable** than applying `Softmax` separately.


**Flow: Logits → Probabilities → Loss**

```text
Model Output (logits) ──► Softmax ──► Probabilities ──► Compare with Target ──► Cross-Entropy Loss
```

**Optimizer Choice**

- We’ll start with **Adam**:
  - Adaptive learning rate → good for quick convergence.
  - Handles sparse gradients (common in images).

- Alternative: **SGD + momentum**
  - Often used in large-scale training (e.g., ImageNet).
  - Can generalize slightly better but needs careful LR tuning.


**Learning Rate Note**

- Initial LR = `1e-3` (good default for Adam).  
- We may adjust later (e.g., LR scheduler) if loss plateaus or oscillates.

### Training the CNN

**Training Process Overview**

We train our baseline CNN using a standard loop:

1. **Epoch iteration:**
   - **Training phase:** forward pass → loss → backward pass → optimizer step.
   - **Validation phase:** evaluate on held-out validation data (no gradient updates).

2. **Metrics tracked:**
   - Training & validation **loss** per epoch.
   - Training & validation **accuracy** per epoch.
   - **Best model checkpoint** saved (lowest validation loss).

**Key Implementation Details**

- **Mode switching:**  
  - `model.train()` during training (activates dropout/batch norm if present).  
  - `model.eval()` during validation (disables gradient updates).

- **Checkpoint saving:**  
  - Save best weights when validation loss improves.  
  - Prevents overfitting from later epochs overwriting good weights.

- **Visualization:**  
  - Plot training/validation loss and accuracy curves after training completes.