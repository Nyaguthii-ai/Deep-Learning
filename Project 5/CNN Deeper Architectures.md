### CNN Deeper Architectures
**Architectures We’ll Compare**

1. **Baseline CNN (NB02)**  
   - 2 convolution layers (3×3) + 2 fully connected layers.  
   - Designed for teaching fundamentals; very shallow.

2. **LeNet-like CNN (1998)**  
   - Classic architecture for MNIST digit recognition.  
   - 3 convolution layers + 2 fully connected layers.  
   - Slightly deeper; captures more **mid-level features** (edges → textures).

3. **AlexNet-mini (2012) / VGG-lite**  
   - Inspired by ImageNet-winning networks.  
   - 4–5 convolution layers + deeper FC layers.  
   - Captures **hierarchical features** (edges → textures → object parts → objects).

**High-Level ASCII Diagrams**

**(a) Baseline CNN (NB02)**

```text
Input (3×128×128)
│
├── Conv(3→16) → ReLU → MaxPool(2×2)
│
├── Conv(16→32) → ReLU → MaxPool(2×2)
│
├── Flatten → FC(32×32×32 → 128) → ReLU
│
└── FC(128 → 37) → Softmax
```

**(b) LeNet-like CNN**

```text
Input (3×128×128)
│
├── Conv(3→16) → ReLU → MaxPool(2×2)
│
├── Conv(16→32) → ReLU → MaxPool(2×2)
│
├── Conv(32→64) → ReLU → MaxPool(2×2)
│
├── Flatten → FC(64×16×16 → 256) → ReLU
│
└── FC(256 → 37) → Softmax
```

**(c) AlexNet-mini / VGG-lite**

```text
Input (3×128×128)
│
├── Conv(3→32) → ReLU → Conv(32→32) → ReLU → MaxPool(2×2)
│
├── Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool(2×2)
│
├── Conv(64→128) → ReLU → MaxPool(2×2)
│
├── Flatten → FC(128×16×16 → 512) → ReLU → Dropout
│
└── FC(512 → 37) → Softmax

```

**Parameter Count and Shape Transformations**

| Layer Type    | Baseline CNN                  | LeNet-like           | AlexNet-mini          |
|---------------|-------------------------------|----------------------|-----------------------|
| **Conv Layers** | Few params ($3\times3$ filters) | More filters (16→64) | Many filters (32→128) |
| **FC Layers**   | Dominant ($32k \to 128$)       | Moderate ($64\times16\times16 \to 256$) | Larger ($128\times16\times16 \to 512$) |
| **Total Params**| ~ 4$M                      | ~ 6$M            | ~ 10$–$12$M       |

**Receptive Field Expansion**

- **Baseline CNN**: Each neuron “sees” only small local patches.
- **LeNet-like**: Deeper layers combine local features → larger effective receptive field.
- **AlexNet-mini**: Stacks more conv layers; receptive field covers most of the image, capturing global context (e.g., full face of a dog).


**Why Compare These?**

- **Highlights trade-offs:**
  - Depth vs. computation (more convs = more params but richer features).
  - Pooling strategy (aggressive vs gradual downsampling).
  - Fully connected size (impact on parameter efficiency).

**Parameter Awareness**

- We print **parameter counts** for each model *before training*.
- This helps relate **model depth vs parameter count vs performance** later.
- Expect:
  - **Baseline**: small, fast to train.
  - **LeNet-like**: slightly deeper, similar parameter count.
  - **AlexNet-mini**: deepest, more capacity but also risk of overfitting.

#### **Feature Hierarchy in CNNs**

- **Shallow CNNs (Baseline):**
  - First conv layer detects **edges and simple colors**.
  - Second layer combines edges into **basic textures** (e.g., fur patterns).

- **Deeper CNNs (LeNet-like, AlexNet-mini):**
  - Additional layers can capture **mid- and high-level features**:
    - **Edges → textures → object parts → full objects**
  - Theoretically better for **fine-grained classification** (e.g., 37 pet breeds).

  
**Does Deeper Always Mean Better?**

- **Our Results:**
  - LeNet-like slightly outperformed baseline (modest depth gain).
  - AlexNet-mini **did not improve** despite more layers:
    - Likely due to **limited training data** and **no augmentation**.
    - Deeper models require **more data** to learn effectively.

- **Overfitting Risk:**
  - Deeper models memorized training data quickly (high train accuracy).
  - Validation accuracy plateaued early → classic sign of **overfitting**.

**Parameter Efficiency vs Performance**

- Baseline vs LeNet-like: Similar parameter counts, slight accuracy boost.
- AlexNet-mini: More layers, **fewer fully connected parameters**, but weaker results → depth alone isn’t sufficient.

**Practical Considerations**

- **Training Cost:**  
  - More layers = more computation (even on CPU-only setup).  
  - Small gains may not justify cost unless dataset size increases.

- **Data Augmentation (NB03 Insight):**  
  - Crucial for enabling deeper models to generalize without overfitting.

- **Transfer Learning (Future Project):**  
  - Pretrained deeper networks (ResNet, VGG) can leverage learned features and outperform shallow models on small datasets.

**Takeaway**

Depth is **powerful but not magical**:  
- Without **enough data or regularization**, deeper CNNs fail to generalize.  
- Next steps (future notebooks) involve **augmentation + transfer learning** to unlock the full potential of deep architectures.
