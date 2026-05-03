# Training Deep Convolutional GANs (DCGANs)

**DCGAN** is a widely adopted architecture that improves GAN training by using:

✔ Deeper convolutional & transposed convolutional layers  
✔ **BatchNorm2d** to stabilize gradients  
✔ **ReLU / LeakyReLU activations** for better gradient flow  
✔ Standardized **weight initialization** for stable convergence

This architectural shift allows GANs to:
- Learn **spatial hierarchies** (edges → textures → structures)  
- Achieve **more stable training**  
- Generate **visually coherent, high-quality images**

```text
Minimal Conv GAN (Notebook 1)
------------------------------
z (latent vector)
   ↓
Linear → reshape to feature map
   ↓
Few ConvTranspose2d layers
   ↓
Tanh output (3×64×64)

✅ Preserves basic spatial structure  
⚠️ Shallow network (limited detail)  
⚠️ No batch normalization (less stable)

DCGAN 
------------------------------
z (latent vector)
   ↓
Deeper ConvTranspose2d stack
   ↓
BatchNorm2d + ReLU (stabilize training)
   ↓
ConvTranspose2d → Tanh output (3×64×64)

✅ Learns rich textures and styles  
✅ More stable training  
✅ Generates higher-quality images

```
To produce **higher-quality, more stable, and more detailed images**, we now move to a **deeper convolutional architecture — DCGAN (Deep Convolutional GAN)**.

**Depth builds hierarchical visual features**

Shallow conv nets capture only simple edges or blobs.  
**Deep** conv nets stack many layers, allowing the model to learn:

- **Low-level features:** edges, colors  
- **Mid-level features:** textures, brushstrokes  
- **High-level features:** shapes, artistic structure

This hierarchy is **essential for generating realistic art**.

**Batch Normalization stabilizes GAN training**

GANs are notoriously hard to train.  
**DCGAN introduces BatchNorm2d**, which:

- Normalizes activations
- Smooths gradient flow
- Prevents one network from overpowering the other
- Reduces training collapse

✅ Result: **More stable and consistent training**

**Balanced architecture for Generator and Discriminator**

DCGAN follows a carefully designed architecture:

- Symmetric depth in **Generator** and **Discriminator**
- Standardized **weight initialization** (mean = 0, std = 0.02)
- **ReLU** in Generator, **LeakyReLU(0.2)** in Discriminator

✅ This balance helps both networks learn together effectively.

**Learnable upsampling with ConvTranspose2d**

Instead of flattening and using fully connected layers, DCGAN uses **transposed convolutions** to *learn* how to upsample.

**Important clarification:**  
✅ We do **not** flatten images.  
✅ We **only** reshape the latent vector `z` into a 4D tensor so ConvTranspose2d can process it.

**Generator flow (aligned with Section 4):**

```text
z (N, 100) → reshape → (N, 100, 1, 1)
→ ConvTranspose2d + BatchNorm + ReLU → (N, 256, 4, 4)
→ ConvTranspose2d + BatchNorm + ReLU → (N, 128, 8, 8)
→ ConvTranspose2d + BatchNorm + ReLU → (N, 64, 16, 16)
→ ConvTranspose2d + BatchNorm + ReLU → (N, 32, 32, 32)
→ ConvTranspose2d + Tanh → (N, 3, 64, 64)
```

At each stage:
✅ Spatial size increases (4 → 8 → 16 → 32 → 64)  
✅ Channels decrease (256 → 128 → 64 → 32 → 3)  
✅ Detail and texture improve

### Define the DCGAN Generator

**Generator** of our DCGAN. The Generator starts from a **latent noise vector** $z \in \mathbb{R}^{100}$ sampled from $\mathcal{N}(0,1)$ and **progressively upsamples** it into a **$3\times 64\times 64$** RGB image.

**Layer-by-layer flow**
```text
Latent vector (z)
    ↓ reshape to (N, z_dim, 1, 1)
ConvTranspose2d → BatchNorm → ReLU
    ↓
ConvTranspose2d → BatchNorm → ReLU
    ↓
ConvTranspose2d → BatchNorm → ReLU
    ↓
ConvTranspose2d → Tanh
    ↓
Generated image (N, 3, 64, 64)
```
Each layer “paints” more structure — like an artist refining a sketch into a detailed painting.

---
- **Output:** $3\times 64\times 64$ image in $[-1, 1]$

```text
z (N,100,1,1)
  → ConvT(100→256, k=4, s=1, p=0) → BN → ReLU      → (N,256,4,4)
  → ConvT(256→128, k=4, s=2, p=1) → BN → ReLU      → (N,128,8,8)
  → ConvT(128→ 64, k=4, s=2, p=1) → BN → ReLU      → (N, 64,16,16)
  → ConvT( 64→ 32, k=4, s=2, p=1) → BN → ReLU      → (N, 32,32,32)
  → ConvT( 32→  3, k=4, s=2, p=1) → Tanh           → (N,  3,64,64)
```

**Why these choices?**
- **Upsampling with transposed convolutions**: Each ConvTranspose2d layer increases spatial size (e.g., $4!\to!8!\to!16!\to!32!\to!64$) while learning where and how to place texture and structure.
- **BatchNorm**: Normalizes activations within each mini-batch, making gradients more stable and accelerating learning.
- **ReLU in hidden layers: Encourages strong, sparse activations that help synthesize vivid structures.
- **Tanh at the output**: Ensures pixel values lie in $[-1,1]$, matching our input normalization; this alignment is crucial so that the Discriminator compares images in the same range.

### Define the DCGAN Discriminator

**Discriminator** — a convolutional network that receives an image of size **$3\times 64\times 64$** and outputs a **probability** that the image is *real* (from the dataset) rather than *fake* (from the Generator).

**Layer intuition (top–down)**  
We progressively **downsample** the image with strided convolutions, increasing channels while reducing spatial size. At each stage we use:

- **Conv2d (stride=2)** to halve spatial dimensions and extract richer features.  
- **BatchNorm** (except the very first layer) to stabilize gradients.  
- **LeakyReLU** activations (slope $=0.2$) so gradients can still flow when activations are negative.

Finally, we flatten to a single logit and apply **Sigmoid** to obtain a probability in $(0,1)$.

```text
Input: (N, 3, 64, 64)
 → Conv2d(3→64,  k=4, s=2, p=1)       → LeakyReLU(0.2)         → (N, 64, 32, 32)
 → Conv2d(64→128,k=4, s=2, p=1)       → BatchNorm → LeakyReLU  → (N,128, 16, 16)
 → Conv2d(128→256,k=4, s=2, p=1)      → BatchNorm → LeakyReLU  → (N,256,  8,  8)
 → Conv2d(256→512,k=4, s=2, p=1)      → BatchNorm → LeakyReLU  → (N,512,  4,  4)
 → Flatten → Linear(512·4·4 → 1)      → Sigmoid                 → (N, 1)
```

**Why LeakyReLU (and not ReLU)?**

Standard ReLU sets negative activations to zero, which can cause **dead features** and **vanishing gradients** when the Discriminator becomes confident. **LeakyReLU** preserves a small negative slope, keeping gradients alive and making the adversarial game more stable.

### Initialize Weights and Setup Training

Before we train a DCGAN, we need to set up **sensible defaults** that make the adversarial game learnable and stable. Three ingredients matter here:

**Weight Initialization (DCGAN recipe)**  
Convolutional GANs are very sensitive to initial weights. The original DCGAN paper recommends initializing convolutional and transposed-convolutional **weights** from a **zero-mean Gaussian** with **small variance**:
$$
W \sim \mathcal{N}(0,\; 0.02^2)
$$
This keeps early activations in a reasonable range and helps avoid **vanishing** or **exploding** gradients. For **BatchNorm** layers, it is common to initialize the **scale** (gamma) near 1 and the **bias** (beta) at 0:
$$
\gamma \sim \mathcal{N}(1,\; 0.02^2), \quad \beta = 0
$$

**Loss Function (Binary Cross-Entropy)**  
We model the Discriminator’s output as a probability that an image is **real**. The **Binary Cross-Entropy** loss measures how well $D$ recognizes real images and how well $G$ fools $D$:
- For real images: target label = 1  
- For generated images: target label = 0 (for $D$’s update)  
- For $G$’s update, we *want* $D(G(z)) \to 1$, so target label = 1

**Optimizer (Adam with tuned betas)**  
DCGANs typically use **Adam** with:
- learning rate $\text{lr} = 0.0002$
- momentum terms $(\beta_1,\beta_2) = (0.5, 0.999)$

Lowering $\beta_1$ from the default 0.9 smooths the moving averages and often stabilizes the adversarial updates.

**Fixed Noise for Monitoring**  
To *see* learning progress, we keep a fixed batch of latent vectors $z$ sampled once at the start. After each epoch, we generate images from this fixed $z$ so changes in quality are easy to compare across time.
