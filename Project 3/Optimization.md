### **Optimization**

> ❓ How do we **train** our model to minimize that loss?

The answer lies in a powerful concept:  
🎯 **Optimization** — the art of adjusting parameters (weights and biases) to make better predictions.

>**The Goal of Optimization**

Every model has **parameters** (like weights in a neural network).  
During training, we want to **adjust these parameters** to **minimize the loss** on our dataset.

Mathematically:

$$
\min_{\theta} \ \mathcal{L}(\theta)
$$

Where:
- $\theta$ = parameters of the model
- $\mathcal{L}(\theta)$ = loss function (e.g., cross-entropy)

This is where **gradient-based optimization** comes in.

>**The Idea of a Loss Surface**

Imagine a hilly landscape — valleys and mountains.  
Each point on this surface corresponds to a different set of model parameters.  
Our goal is to **descend** to the lowest point — the global (or local) **minimum** of the loss.


> The **loss surface** is often very complex, especially for deep networks — with many bumps and valleys.


>**Gradient Descent – A Quick Intuition**

Gradient Descent is an algorithm to "walk downhill" the loss surface.

Each update looks like:

$$
\theta := \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

Where:
- $\eta$ = learning rate (step size)
- $\nabla_\theta \mathcal{L}$ = gradient of loss w.r.t parameters

> **🚦 But Wait... Which Gradient?**

This leads to different **variants** of Gradient Descent based on how we compute the gradient:

| Method               | How Gradient is Computed                 | Speed      | Stability | Memory |
|----------------------|------------------------------------------|------------|-----------|--------|
| **Batch Gradient Descent** | Use the **entire dataset** to compute gradient | Slow        | Stable    | High   |
| **Stochastic Gradient Descent (SGD)** | Use **1 random sample** at a time           | Fast updates | Noisy     | Low    |
| **Mini-batch SGD**   | Use **a few samples (batch)**            | Good tradeoff | Balanced  | Medium |

We’ll explore these in detail with code and plots.

>🔍 What You'll Learn in This Notebook

- How different optimizers behave on real training data
- How **batch size** affects training speed and accuracy
- How to **interpret training curves** (loss/accuracy over epochs)
- When to choose **SGD vs. Mini-Batch SGD vs. Full GD**

In neural networks, we use **gradient-based optimization**. The most common approach is **Stochastic Gradient Descent (SGD)** and its variants.

But what does that really mean?

Let’s explore three core optimization strategies:

>**🌐 1. Gradient Descent (Full-Batch)**

- Uses **all training data** to compute the gradient.
- **Very stable**, but **slow and memory-intensive**.
- Gradient direction is exact but computation is expensive for large datasets.

>Update Rule

Let $ \theta $ be model parameters, $ \eta $ the learning rate:

$$
\theta = \theta - \eta \cdot \nabla J(\theta) \quad \text{where } J(\theta) = \text{loss on entire dataset}
$$

>**🔄 2. Stochastic Gradient Descent (SGD)**

- Uses **one sample at a time** to compute the gradient.
- **Very fast**, but **noisy updates**.
- Faster learning but erratic convergence.

>Update Rule

$$
\theta = \theta - \eta \cdot \nabla J(\theta^{(i)}) \quad \text{where } J(\theta^{(i)}) = \text{loss on single sample}
$$

>**📦 3. Mini-Batch SGD**

- Compromise: use **small batches** (e.g., 32 samples).
- **Balances speed and stability**.
- Widely used in practice.

>📉 Update Rule

$$
\theta = \theta - \eta \cdot \nabla J(\theta^{(i:i+k)}) \quad \text{for a mini-batch of } k \text{ samples}
$$

> **🌍 Optimizer Landscape**

| Optimizer | Gradient Computation | Update Frequency | Pros | Cons |
|----------|----------------------|------------------|------|------|
| **Gradient Descent (GD)** | Uses *all* training data | Once per epoch | Stable updates | Slow, memory-heavy |
| **Stochastic Gradient Descent (SGD)** | Uses *one sample* at a time | Per sample | Fast, frequent updates | Noisy, unstable |
| **Mini-Batch SGD** | Uses *a small batch* | Per mini-batch | Balance of both worlds | Hyperparameter tuning needed |

>Why Does This Matter?
- Each optimizer affects **convergence speed**, **stability**, and **generalization**.
- We’ll compare them side-by-side in upcoming sections.

> **🧠 Summary**

- **Full-Batch GD** is deterministic but slow.
- **SGD** is fast and noisy — good for large datasets, but can oscillate.
- **Mini-Batch SGD** is widely used — it balances **speed** and **stability**.

- **Loss Convergence**:
  - **SGD (Per Sample)** shows the **fastest loss drop** initially, but the curve is noisy.
  - **Mini-Batch SGD** shows a **smooth and fast** descent, balancing speed and stability.
  - **Full-Batch GD** converges steadily but slowly — its updates are too infrequent for quick learning.

- **Accuracy Over Epochs**:
  - SGD achieves high accuracy **early** but shows **fluctuations** due to its noisy nature.
  - Mini-Batch SGD provides **consistent and steadily improving accuracy**.
  - Full-Batch GD improves slowly and reaches **lower final accuracy** than the other two.

>🧠 Key Insight

- **Mini-Batch SGD** offers the best compromise:
  - Efficient GPU usage
  - Stable gradients
  - Faster convergence than Batch GD
  - Smoother than per-sample SGD

This is why **mini-batch training is standard** in modern deep learning!

