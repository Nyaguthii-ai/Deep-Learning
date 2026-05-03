# Generative Adversarial Networks (GANs)

GANs have transformed how we think about creativity and artificial intelligence. Instead of simply classifying or predicting, GANs *create* — they can generate **new, realistic, and original data** such as images, music, and even art.

**🎭 The Duel Between Two Networks**

A **Generative Adversarial Network (GAN)** is built on the idea of **competition and cooperation** between two neural networks:

- 🧑‍🎨 **Generator (G):**  
  The Generator starts with random noise and tries to **create fake images** that resemble real ones.  
  Its goal is to *fool* the Discriminator.

- 🕵️ **Discriminator (D):**  
  The Discriminator is like an art critic — it receives both real images (from the dataset) and fake ones (from the Generator)  
  and tries to **distinguish between real and fake**.

Through this constant back-and-forth competition(continuous **adversarial game**), both networks improve — the Generator becomes a better artist, and the Discriminator becomes a better critic.  
Eventually, the Generator learns to produce outputs that can **fool the Discriminator** — and us!

**🖼️ GAN Concept Diagram**

```text
        ┌──────────────────┐
        │   Random Noise   │
        │   (z ~ N(0,1))   │
        └──────┬───────────┘
               │
               ▼
        ┌────────────────┐
        │   Generator    │
        │     G(z)       │
        └──────┬─────────┘
               │   Fake Images
               ▼
        ┌───────────────┐
        │ Discriminator │
        │     D(x)      │
        └───────────────┘
        ▲         │
        │         ▼
  Real Images   "Fake or Real?"
        │
        └─────────────────────────→ Feedback to Generator
```

This loop continues until both models reach a dynamic equilibrium:
</br>the Discriminator can’t easily tell real art from generated art — that’s when we know the Generator has *learned something meaningful* about the artistic distribution.

**⚔️ The Adversarial Objective**

The training of GANs can be described as a **two-player minimax game**:

$$
\min_G \max_D V(D, G) =
\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] +
\mathbb{E}_{z \sim p_z(z)} [\log (1 - D(G(z)))]
$$

Let’s unpack this:

- The **Discriminator** $D(x)$ outputs a probability — how “real” the input looks.  
- The **Generator** $G(z)$ maps random noise $z$ (from a simple prior, e.g. $\mathcal{N}(0,1)$) to the data space.  
- The **first term** rewards $D$ for correctly identifying real images.  
- The **second term** penalizes $D$ for being fooled by $G$’s fake images.  
- Meanwhile, the Generator tries to *minimize* this overall loss, pushing $D(G(z))$ to be close to 1.

Before we begin training our GAN, we must ensure that the input images and noise vectors are properly prepared for the Generator and Discriminator.  
This step is critical because **GANs are highly sensitive to data scaling and normalization** — incorrect preprocessing can make training unstable or even prevent convergence altogether.

**Why Normalize Images?**  
Most GAN architectures (including ours) use a **tanh** activation function in the final layer of the Generator.  
The `tanh` function outputs values in the range **[-1, 1]**, so to match this, we must normalize our real training images to the **same range**.  
This ensures that both real and generated images occupy a consistent value space, allowing the Discriminator to make meaningful comparisons.

Mathematically, normalization transforms pixel values from **[0, 1] → [-1, 1]** as:
$$
x_{norm} = 2x - 1
$$

**Latent Noise (z):**  
The Generator doesn’t start with an image — it starts with **random noise**.  
This noise vector, denoted as $ z $, is sampled from a standard normal distribution $ \mathcal{N}(0, 1) $ and typically has **100 dimensions**.  
During training, the Generator learns to transform this random noise into realistic images.

### Building a Minimal GAN Architecture

To make our first GAN concrete, we will implement a **very shallow CNN** pair — a **Generator** that upsamples random noise into $64\times 64$ color images, and a **Discriminator** that judges whether an input image looks real or fake. We keep the design intentionally small so we can focus on the **adversarial idea** rather than heavy engineering.

**Design intuition we will follow**

- **Generator ($G$):** start from a latent vector $z\in\mathbb{R}^{100}$ sampled from $\mathcal{N}(0,1)$, then **upsample** with `ConvTranspose2d` blocks, using **ReLU** activations and a **$\tanh$** at the end so outputs live in $[-1,1]$ (matching our preprocessing).
- **Discriminator ($D$):** take a $64\times 64$ RGB image and **downsample** with `Conv2d` blocks, using **LeakyReLU** (slope $=0.2$) and a final **Sigmoid** to output a probability.

**Layer-by-layer shapes**

```text
Generator G (input: z ∈ ℝ^100)
z  → Linear → 128×8×8  → ReLU
   → ConvTranspose2d(128→64, k=4, s=2, p=1)  → 64×16×16  → ReLU
   → ConvTranspose2d(64→32,  k=4, s=2, p=1)  → 32×32×32  → ReLU
   → ConvTranspose2d(32→3,   k=4, s=2, p=1)  → 3×64×64   → Tanh  (range [-1, 1])

Discriminator D (input: image 3×64×64)
3×64×64  → Conv2d(3→32,  k=4, s=2, p=1)  → 32×32×32 → LeakyReLU(0.2)
         → Conv2d(32→64, k=4, s=2, p=1)  → 64×16×16 → LeakyReLU(0.2)
         → Conv2d(64→128,k=4, s=2, p=1)  → 128×8×8  → LeakyReLU(0.2)
         → Flatten → Linear(128·8·8 → 1) → Sigmoid (probability “real”)
```

**Why these activations?**

- In $G$, **ReLU** encourages strong, sparse feature activations during upsampling, while $\tanh$ maps pixel predictions to $[-1,1]$ to align with our normalized dataset.
- In $D$, **LeakyReLU** avoids “dead” neurons during downsampling and helps gradients flow even when activations are negative. Sigmoid outputs a calibrated probability $\in (0,1)$.

We also use the common **DCGAN weight initialization** heuristic: convolutional weights $\sim \mathcal{N}(0, 0.02^2)$, biases $=0$. This tends to stabilize early adversarial training.

**Training dynamics in a nutshell**

- Step D: show $D$ real images (label 1) and fake images $G(z)$ (label 0), compute the Binary Cross Entropy loss, and update $D$ to better separate real vs fake.
- Step G: sample fresh noise $z$, generate $G(z)$, ask $D$ to score them as real (label 1), and update $G$ so its fakes look more real to $D$.

**Why fixed noise helps**  
If we evaluate $G$ on *the same set of $z$ vectors* after each epoch, any improvement or regression in image quality becomes easy to spot.

### Understanding Adversarial Dynamics

A well-behaved GAN requires both $G$ and $D$ to learn at a comparable pace:

- If $D$ becomes **too strong**, it perfectly separates real and fake samples, giving gradients close to zero. The Generator stops learning — a phenomenon called **vanishing gradients**.
- If $G$ becomes **too strong**, it produces samples that easily fool $D$, so $D$ fails to recover meaningful signals about reality.
- The sweet spot is a **dynamic equilibrium**, where $G$ and $D$ continuously challenge each other but neither fully dominates.

This is why GAN training often requires careful tuning of learning rates, batch sizes, and update frequencies.

**Non-convergence and oscillation**

Unlike standard supervised learning, GANs do not necessarily **converge** to a fixed minimum.  
Because $G$ and $D$ are optimizing **opposing objectives**, the training tends to **oscillate** — loss values may fluctuate even when the quality of generated images improves.
