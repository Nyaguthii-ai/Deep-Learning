# Variational Autoencoders (VAEs)

VAEs extend autoencoders by introducing a probabilistic latent space, allowing us to not only rebuild existing data but also generate entirely new, realistic samples.
This makes VAEs one of the most foundational generative models in deep learning.

VAEs make latent space continuous, structured, and sampleable by by treating the latent space as a **probability distribution**, not a fixed vector.  
Instead of encoding an input into a single point, we learn a **mean (μ)** and **variance (σ²)** that define a Gaussian distribution for each input.  
We can then *sample* from this distribution to produce diverse outputs.

> In essence, the VAE learns how to imagine — creating realistic new samples by understanding the distribution of the data itself.

### Conceptual Bridge: From AE to VAE

**The Motivation for a Probabilistic Latent Space**

To generate realistic new samples, we want our latent space to be:
- **Continuous** – small changes in `z` lead to smooth changes in output.
- **Structured** – nearby points correspond to similar inputs.
- **Sampleable** – we can draw new `z` values and still get valid outputs.

This is where **Variational Autoencoders (VAEs)** come in.  
Instead of mapping each input to a *single latent vector*, a VAE maps it to a **distribution** in the latent space.

**From Deterministic Codes to Distributions**

| Model | Latent Representation | Key Property |
|--------|------------------------|---------------|
| **Autoencoder (AE)** | $ z = f_\theta(x) $ | Single point (deterministic) |
| **Variational Autoencoder (VAE)** | $ z \sim \mathcal{N}(\mu, \sigma^2) $ | Distribution (probabilistic) |

In a VAE:
- The **encoder** outputs two vectors — a mean $ \mu $ and a log-variance $ \log\sigma^2 $.  
- Together, they define a **Gaussian distribution** for each input image:
  $$
  q_\theta(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x))
  $$
- The **decoder** then reconstructs from a **sample** of this distribution:
  $$
  \hat{x} = g_\phi(z)
  $$

This introduces controlled randomness that helps the model learn **smooth, continuous latent spaces**.


**Schematic Overview**

Below is a conceptual diagram (to be visualized in the notebook):

   >| Autoencoder (AE) | ------------------------|-----------------------| Variational Autoencoder (VAE) |
```text
         x                                                  x
         ↓                                                  ↓
   ┌────────────┐                                    ┌────────────┐
   │  Encoder   │                                    │  Encoder   │
   └────────────┘                                    └────────────┘
         ↓                                                  ↓
      z = f(x)                                          μ(x), σ(x)
                                                            ↓
                                                   z ~ N(μ, σ²)   ← Sample
                                                            ↓
                                                      ┌────────────┐
                                                      │  Decoder   │
                                                      └────────────┘
                                                            ↓
                                                            x̂

```

**Why This Matters**

By making the latent representation *probabilistic*:
- We encourage **smooth transitions** in latent space.
- We can **sample new z values** from $ \mathcal{N}(0, I) $ to create novel faces.
- We gain a model that not only reconstructs but **generates**.

In short:
> The VAE transforms the Autoencoder from a *compressor* into a *creator.*

### The Reparameterization Trick

How can we **sample** from that distribution *while keeping the model differentiable*?

**The Challenge: Non-Differentiable Sampling**

 **Sampling directly** from this distribution breaks the computational graph — the randomness makes it impossible for gradients to flow back through the sampling step.  
That means the model can’t learn effectively via backpropagation.

> 🧩 **Problem:** Sampling is stochastic and non-differentiable.  
> We need a way to “sample” that still allows gradients to pass through the encoder.

To solve this, we **separate the randomness** from the deterministic parameters (μ, σ).  
We introduce a random variable $ \epsilon $ drawn from a standard normal distribution:
$$
\epsilon \sim \mathcal{N}(0, I)
$$
and compute:
$$
z = \mu + \sigma \odot \epsilon
$$

This equation expresses the sample `z` as a **deterministic transformation** of μ, σ, and a random ε. 
 
Because ε is the *only* source of randomness, gradients can still flow through μ and σ.

> 💡 The model doesn’t directly sample from the distribution —  
> it *constructs* a sample by shifting and scaling a noise vector.
> “Take a random standard-normal vector (ε),  
> stretch it by σ, shift it by μ — and you now have a sample z.”

This way:
- The **noise** provides stochasticity.  
- The **parameters (μ, σ)** remain differentiable.  

Hence, we get **stochastic sampling + gradient flow** — the best of both worlds!
>✨ The reparameterization trick is what makes VAEs trainable

### VAE Architecture Definition
$$
x \xrightarrow[\text{encode}]{f_\theta} (\mu, \log\sigma^2)
\quad \xrightarrow[\text{sample via reparam}]{z = \mu + \sigma \odot \epsilon}
\quad \xrightarrow[\text{decode}]{g_\phi} \hat{x}
$$

- **Encoder:** Maps input image → mean and log-variance vectors.  
- **Reparameterization:** Samples `z` in a differentiable way.  
- **Decoder:** Reconstructs the image from latent vector `z`.  

The **output** of the decoder uses a **Sigmoid activation** since pixel intensities lie in [0, 1].

**Network Dimensions**

| Component | Layer Structure | Description |
|------------|----------------|--------------|
| **Encoder** | 2304 → 512 → 128 → (μ, logσ²) | Compress input and estimate latent distribution |
| **Latent space** | 128 | Sampled from Gaussian via reparameterization |
| **Decoder** | 128 → 512 → 2304 → Sigmoid | Reconstruct image from latent sample |

Each face image is 48 × 48 grayscale → flattened into a **2,304-dimensional vector**.

**Conceptual Note**

This architecture looks like a regular AE — but conceptually it’s different:

| Aspect                | Autoencoder                  | Variational Autoencoder                 |
| --------------------- | ---------------------------- | --------------------------------------- |
| Latent representation | Deterministic vector         | Probabilistic (mean + variance)         |
| Sampling              | None                         | Differentiable random sampling          |
| Objective             | Minimize reconstruction loss | Minimize reconstruction + KL divergence |
| Capability            | Reconstruction               | Reconstruction + Generation             |

### VAE Loss Function

Our Variational Autoencoder is trained with a **composite loss** that balances two goals:
- Reconstruct each input image faithfully.
- Shape the latent space into a smooth, continuous distribution that we can sample from.

Formally, we minimize
$$
L = L_\text{recon} + L_\text{KL}
$$

**Reconstruction Loss (Fidelity)**
The reconstruction term measures how close the output $\hat{x}$ is to the input $x$.

- **MSE** (mean squared error), suitable for grayscale images in \[0, 1\]:
$$
L_\text{recon}^{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{HW}\sum_{p}(x_{i,p}-\hat{x}_{i,p})^2
$$

- **BCE** (binary cross-entropy), also common for \[0, 1\] pixel intensities with a Sigmoid output:
$$
L_\text{recon}^{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\frac{1}{HW}\sum_{p}\big[x_{i,p}\log \hat{x}_{i,p} + (1-x_{i,p})\log(1-\hat{x}_{i,p})\big]
$$

Either choice encourages faithful reconstructions. For our 48×48 grayscale faces, **MSE** is a simple, robust default.

**KL Divergence (Latent Regularization)**
The encoder outputs $(\mu, \log\sigma^2)$ for each input, defining a Gaussian in latent space.  
To make the latent space **smooth and sampleable**, we regularize it toward the standard normal $\mathcal{N}(0, I)$ using the KL divergence:
$$
L_\text{KL} = -\frac{1}{2} \sum \big( 1 + \log\sigma^2 - \mu^2 - \sigma^2 \big)
$$

**Putting It Together**
We combine both terms:
$$
L = L_\text{recon} + L_\text{KL}
$$

Optionally, we can introduce a weighting factor $\beta$ (the **beta-VAE** idea) to control the trade-off between reconstruction fidelity and latent regularization:
$$
L = L_\text{recon} + \beta \, L_\text{KL}
$$

- Larger $\beta$ strengthens latent structure (more disentanglement, potentially blurrier reconstructions).
- Smaller $\beta$ prioritizes reconstruction sharpness (less regularized latent space).

> **Plotting Loss Curves (Total, Recon, KL)**
![Plot Loss Curves (Total, Recon, KL)](<VAE Loss Reconstruction.png>)

**How to Read These Curves**

- When reconstruction loss decreases, the decoder is learning to produce outputs closer to inputs — better reconstructions.
- When KL divergence stabilizes at a reasonable scale, the encoder is learning a latent distribution that is well-regularized and smooth, which we can sample from to generate new faces.
- When total loss decreases and validation trends follow training, we are moving toward a useful generative model.

**Core Insights**
- **Autoencoders (AEs)** focus on *reconstruction* — compressing and decompressing data deterministically.
- **Variational Autoencoders (VAEs)** go further — they *learn a distribution* of latent representations, enabling *generation* of entirely new, unseen examples.
- The **reparameterization trick** bridges random sampling and gradient-based optimization, allowing us to train probabilistic models end-to-end.
- The **KL divergence term** encourages a well-structured latent space that is continuous, smooth, and suitable for interpolation and synthesis.
- The **reconstruction loss** ensures that generated outputs remain faithful to real images.

**Why It Matters**

A VAE doesn’t just memorize — it *understands* the underlying structure of the data.  
This makes VAEs powerful tools for:
- Representation learning  
- Controlled generation (e.g., faces with varying expressions)  
- Semi-supervised learning and anomaly detection  

### Optional Extensions

Now that we understand the VAE fundamentals, we can explore several meaningful extensions and experiments.

**(a) Convolutional VAE (CVAE)**
- Replace dense layers with **Conv2d / ConvTranspose2d** layers.
- Captures **spatial hierarchies** in images → sharper, more coherent reconstructions.
- Ideal for face or object image datasets.

**(b) Latent Dimensionality Experiments**
- Try smaller (e.g., 8) or larger (e.g., 32, 64) latent vector sizes.  
- Observe trade-offs between:
  - Reconstruction sharpness  
  - Latent smoothness and diversity  

**(c) Latent Traversal Visualization**
- Vary one dimension of the latent vector \( z \) while keeping others fixed.
- Decode and visualize outputs → observe gradual transformation in expressions or features (smile intensity, head tilt, etc.).
- This helps us interpret what each latent variable represents.

**(d) β-VAE Experiment**
- Modify total loss:  
  $$
  L = L_\text{recon} + \beta \, L_\text{KL}
  $$
- Increasing **β > 1** enforces stronger disentanglement between latent factors, at the cost of reconstruction quality.
- Helps us explore **information bottleneck** and **disentangled representation learning**.