# Anomaly detection on Autoencoders using facial expressions

Autoencoders are not just theoretical constructs — their ability to reconstruct inputs with varying accuracy can be used for **practical applications**.  
In this notebook, we will focus on two of the most common and powerful ones:

1. **Anomaly Detection:**  
   Detecting unfamiliar or rare expressions by analyzing reconstruction error.  
   When the autoencoder encounters a face it hasn’t learned before, it typically fails to reconstruct it accurately — signaling an anomaly.

2. **Denoising Autoencoder:**  
   Training the model to remove noise from corrupted inputs.  
   This demonstrates how autoencoders can learn *robust, structure-preserving representations* of data.

> Through these applications, we will see how the reconstruction process becomes not only a way to measure how well the model has learned, but also a **tool** for identifying novelty and restoring degraded data.

### Anomaly Detection

The underlying principle is simple yet powerful:

> An autoencoder trained on **familiar data** (faces or expressions it has seen during training) learns to reconstruct them accurately.  
> However, when it encounters **unseen or unusual patterns**, it struggles — leading to a **higher reconstruction error**.

This difference in reconstruction quality provides a useful **signal for anomaly detection**.

When we feed it unfamiliar expressions, the model cannot represent them well in its latent space, resulting in **blurred reconstructions** and **larger reconstruction errors**.

Hence:

- **Known features →** low reconstruction error  
- **Unknown features →** high reconstruction error 

**Mathematical Expression**

The **reconstruction error** for an input image $ x $ and its reconstruction $ \hat{x} $ is computed as:

$$
\text{Reconstruction Error} = \| x - \hat{x} \|^2
$$

This measures the pixel-wise difference — the larger the error, the less familiar the image is to the autoencoder.

**Conceptual Flow**

Below is a schematic overview of the process:

```text
Input Image ──► Encoder ──► Latent Space ──► Decoder ──► Reconstructed Image
      │                                                       │
      └───────────────────── Compute Reconstruction Error ────┘
```
The **latent space** acts as a learned representation of “normal” data.  
When a new image doesn’t fit well within that representation, the **error increases**, flagging it as an anomaly.

### Denoising Autoencoders

**Denoising autoencoders (DAEs)** — a powerful variant designed to **recover clean data from noisy or corrupted inputs**.

A denoising autoencoder learns to **filter out this noise** and **reconstruct the original, clean face**.  

This ability makes DAEs more than just reconstruction tools — they become **robust feature learners**, capable of understanding the essential structure of the data.

> In practice, this robustness helps downstream models perform better on imperfect or real-world data.

**How It Works**

We start with an original input image $x$.  
We then create a **noisy version** $x'$ by adding some random Gaussian noise:

- **Input to the model:** $x'$ (corrupted image)  
- **Target output:** $x$ (clean original image)  
- **Predicted output:** $\hat{x} = f_\theta(x')$

The model is trained to minimize the difference between the clean target and its reconstruction of the noisy input.

**Objective Function**

The loss function captures how close the reconstructed clean face $\hat{x}$ is to the true clean face $x$:

$$
L = | x - f_\theta(x') |^2
$$

This encourages the network to *ignore noise* and focus on the stable, consistent features that define the underlying facial structure.

**Connection to Learning Robust Representations**

By forcing the model to reconstruct the *clean version* from *noisy inputs*, we implicitly teach it to:
- Focus on **global structure** over fine noise patterns.
- Develop a **smoother latent space** — where small input changes don’t cause large representation changes.
- Generalize better to unseen or imperfect data.

> In other words, denoising acts as a **form of regularization**, strengthening the model’s understanding of essential data features — a stepping stone toward more advanced generative models like **Variational Autoencoders (VAEs)** and **GANs**.

By feeding noisy inputs $x'$ and asking the model to produce clean outputs $x$, we force the network to learn stable, structure-focused features.
Instead of memorizing pixels, the encoder learns a representation that is resilient to noise, and the decoder learns how to reconstruct essential facial geometry.

Visualization helps us confirm whether the model has truly learned to **remove noise** while **preserving facial structure**.

We will display **triplets** of images:
> **Noisy Input → Reconstructed Output → Original Clean Image**

As we examine the triplets:

- Noisy Inputs show random speckles or grain introduced by Gaussian noise.
- Reconstructed Outputs appear much smoother and cleaner — the autoencoder learned to filter out noise while retaining key facial features.
- Originals serve as the reference for what the model aims to recover.

> A little smoothness or blurring is expected — it reflects the model’s trade-off between removing noise and preserving detail.
The DAE prioritizes structure over texture, which is why we see stable, recognizable faces even after strong noise injection.

**Summary of What We Learned**

- **Autoencoders as Unsupervised Learners:**  
  They learn compressed, latent representations of data *without labels*, capturing the key structures and variations within facial images.

- **Anomaly Detection:**  
  - The reconstruction error serves as a **measure of familiarity**.  
  - Known emotions (training classes) → **low error** → the model can “rebuild” them well.  
  - Unknown emotions → **high error** → the model struggles, signaling **novelty or anomaly**.  
  - This principle extends to use cases like fault detection, fraud detection, and out-of-distribution monitoring.

- **Denoising Autoencoders:**  
  - By training on **noisy inputs** and **clean targets**, the model learns to **filter noise** and focus on **essential structure**.  
  - This process leads to **robust feature learning** — a foundation for handling imperfect or real-world data.  
  - Slight smoothness is expected: it represents stability over pixel-level precision.

> Together, these two experiments demonstrate how autoencoders can *interpret, detect, and restore* — all without explicit supervision.

**Broader Conceptual Takeaway**

Autoencoders learn to model the **manifold of normal data** — a compressed, meaningful representation of what “makes sense” in the training domain.  
When something doesn’t fit that learned structure (noise, anomalies, unseen classes), the reconstruction process exposes it.

This ability makes them invaluable in:
- **Anomaly & novelty detection**
- **Image denoising & restoration**
- **Dimensionality reduction & visualization**
- **Pretraining for deeper models**
