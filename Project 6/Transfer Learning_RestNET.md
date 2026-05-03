### Transfer Learning - *RestNET*
**Why Transfer Learning?**

Rather than learning visual features from scratch, we can **leverage pretrained networks** (e.g., **ResNet18**) that were trained on the large-scale **ImageNet dataset (1.2M images, 1,000 classes)**. These pretrained models already encode **general low- and mid-level features** (edges, color blobs, textures) that are transferable to new tasks like plant species recognition.

Key advantages:

- **Faster convergence:** The network starts from *rich features* instead of random weights.
- **Better generalization:** Especially useful for **small datasets** with limited diversity.
- **Proven architectures:** ResNet family handles deep networks effectively via **residual connections** (mitigates vanishing gradients).

**Feature Extraction vs Fine-Tuning**

We will explore **two transfer learning strategies**:

1. **Feature Extraction (Frozen Backbone):**
   - Keep pretrained convolutional layers **frozen**.
   - Replace final fully connected layer with a new classifier head (12 classes).
   - Only train the head — faster, less prone to overfitting.

2. **Fine-Tuning (Unfreezing Some Layers) — optional in later notebooks:**
   - Unfreeze selected layers for **task-specific adaptation**.
   - Increases flexibility but also risk of overfitting.

**Key Requirements for Transfer Learning**

1. **Image Size:**  
   Pretrained models like ResNet18 expect **224×224 pixel** inputs (same as ImageNet).

2. **Normalization:**  
   - We use **ImageNet mean and standard deviation** (as the backbone was trained on ImageNet):  
     - Mean = `[0.485, 0.456, 0.406]`  
     - Std = `[0.229, 0.224, 0.225]`  
   - This ensures pixel intensity distributions match those used during pretraining.

3. **Class Mapping Consistency:**  
   - Must ensure **class order** (dataset.classes) is consistent across the learning techniques.

**Meet ResNet18 (Our Pretrained Model)**

ResNet18 is a famous CNN known for **residual connections** — skip pathways that help very deep networks train without “forgetting” earlier information (solves vanishing gradient problems).  
**Don’t worry about the math — think of it as “shortcuts” that let the network combine both simple and complex features as it gets deeper.**

Key structure (simplified):
- Starts with a **7×7 convolution** + pooling (quickly reduces image size).  
- Passes through **4 stages** of layers (called residual blocks). Each stage learns progressively richer features:
  - Stage 1: Edges and color contrasts.  
  - Stage 2: Textures and simple shapes.  
  - Stage 3: Larger patterns (leaf veins, stem shapes).  
  - Stage 4: Full object representations (entire plant).  
- Ends with **global average pooling** and a **fully connected layer** (originally 1,000 ImageNet classes).

**How We Adapt It**

We keep the **feature extractor part** (all convolutional + residual layers) frozen, and only **replace the last fully connected (fc) layer** with a new classifier that outputs.

**Visual Summary:**

```text
         Input (224×224×3)
                ↓
   Conv7×7 + Pooling (basic edges)
                ↓
Residual Stages 1–4 (textures → patterns → objects)
                ↓
      Global Average Pooling
                ↓
New Fully Connected Layer (12 species)
```

**Training Strategy**

- **Feature Extraction Mode**  
  - **Frozen Backbone:** Pretrained ResNet layers remain fixed (weights do not update).  
  - **Trainable Classifier Head:** Only the final `fc` layer learns (≈6K parameters).  
  - Advantage: Faster training, less risk of overfitting on small dataset.

- **Loss Function**  
  - `CrossEntropyLoss`: Suitable for multi-class classification.

- **Optimizer**  
  - Use `Adam` **only on trainable parameters**:
    ```python
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    ```

- **Learning Rate Scheduler (Optional)**  
  - `StepLR` can reduce learning rate after a fixed number of epochs to fine-tune convergence.

- **Training Loop**  
  - Similar structure to scratch CNN in NB02:
    - Alternate between **training** and **validation** phases.
    - Track loss and accuracy per epoch.
    - Save best model (lowest validation loss).