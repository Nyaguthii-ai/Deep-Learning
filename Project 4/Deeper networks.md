# Deeper Networks

A **deeper network** — one with more layers — has more **representational power**. In theory, it can:

- Learn more **complex, non-linear patterns**,
- Capture **hierarchical features** (e.g., edges → textures → objects),
- Offer **better generalization** on large, rich datasets like CIFAR-10.

But deeper isn’t always better — deeper models are **harder to train**.

They often suffer from:

- ⚠️ **Vanishing gradients**: updates become too small to be useful.
- ⚠️ **Exploding gradients**: updates become too large and unstable.
- ⚠️ **Unstable training dynamics**: losses oscillate or fail to converge.

So before diving deeper, we need a solid pipeline in place — and that starts with **preparing the dataset**.

>**Recap: CIFAR-10 Preprocessing**

We already saw that CIFAR-10 images have shape `[3, 32, 32]` — 3 color channels and 32×32 pixels.

For MLPs, we **flatten** these into 1D vectors of length **3072 = 3×32×32**. We also **normalize** pixel values from `[0, 255]` to `[0.0, 1.0]` using `transforms.ToTensor()`.


![Loss and Accuracy curves showing training progress across epochs, with loss decreasing over time and accuracy improving, demonstrating the training dynamics of a deep neural network on CIFAR-10 dataset](<LossandAcccuracy curves.png>)

>⚠️ Common Problems in Deep Networks

We’ll explore these in the next few sections:

| Problem             | Symptom in Curves                | Possible Cause       |
| ------------------- | -------------------------------- | -------------------- |
| Vanishing Gradients | Loss stays flat, low accuracy    | Gradients too small  |
| Exploding Gradients | Spikes in loss, erratic accuracy | Gradients too large  |
| Poor Initialization | Any of the above                 | Random weight scales |

> **ReLU vs. Tanh — A Quick Recap**

**ReLU (Rectified Linear Unit)**: ReLU is simple: it outputs the input if it’s positive, and 0 otherwise. This means neurons are "active" only when they see positive signals. Think of a ReLU neuron like a light switch that turns on only for positive values — simple and efficient.

Why ReLU works well:
- Its gradient is either **0 or 1**, so gradients don’t shrink as easily.
- It **avoids saturation** — there’s no upper cap like Tanh or Sigmoid.
- **Training is usually faster** and more stable with ReLU.

However, some neurons might "die" if they only receive negative inputs forever — these never activate and stop learning. This is called the **Dying ReLU problem**.

**Tanh (Hyperbolic Tangent)**: Tanh squashes inputs to the range between **-1 and 1**. It’s **symmetric around zero**, which can help early in training. Think of Tanh as a soft squashing spring — it compresses all signals to stay within a narrow zone.

But here's the issue:
- In deep networks, after several layers, signals can get **squashed too much**.
- This causes **gradients to shrink** as they move backward — a problem known as the **vanishing gradient**.
- The network struggles to learn because the updates become too tiny.

In summary, ReLU is like a sharp-edged filter: fast, sparse, and generally stable — but watch for inactive neurons. Tanh is smooth and bounded — better for shallow nets, but not great for deep ones due to vanishing gradients.

![Graphs depicting loss and accuracy over time for RelU vs Tanh](<RelU vs Tanh.png>)
- **ReLU** converges faster and reaches **higher accuracy** than Tanh.
- **Tanh** suffers from **slower learning** and **lower performance**, especially in deep networks.
- This is due to **vanishing gradients** caused by Tanh's saturation behavior (squashing values to -1 and 1).

>Key Takeaways

- In deep architectures, **ReLU is usually the better default**.
- Tanh and Sigmoid can cause vanishing gradients, especially when weights aren’t carefully initialized.

One major culprit is poor gradient flow. Let's understand this through intuition and code.
⚡ What Is Gradient Flow?

When we backpropagate errors during training, gradients flow from the output layer to earlier layers.

If gradients are:

- Very small (close to 0): early layers learn very little.
    → This is called the vanishing gradient problem.

- Very large: weights change erratically, leading to instability.
    → This is called the exploding gradient problem.

Both issues make training slow, unstable, or completely ineffective.

> 🧠 **The Problem with Default Initialization in PyTorch**

When we define a model using `nn.Linear` layers in PyTorch, the weights are **automatically initialized** using a method called:

```python
nn.init.kaiming_uniform_  # a.k.a. He Uniform
```

This is **a variant of He Initialization**, designed for **ReLU-based activations**, and is a solid choice *if* we use ReLU.

But here's the catch:

> 🔎 PyTorch applies this **regardless of the activation** we plan to use later.

This means:
- Even if we use **Tanh** (or **Sigmoid**), the layers still get initialized with **He Uniform**.
- But He initialization is **not well-suited** for these activations — and this mismatch leads to **poor learning** in deep networks.

>🔍 What’s Actually Going Wrong?

Let’s say we start with an input vector **x** and pass it through many linear layers:

$$
\text{Layer 1: } x_1 = W_1 x \\
\text{Layer 2: } x_2 = W_2 x_1 \\
\text{... and so on}
$$

If the **variance** of activations or gradients grows or shrinks across layers, learning becomes unstable.  
We want each layer to **preserve the scale** of both:
- **Forward activations**
- **Backward gradients**

That’s where **smart initialization strategies** come in.

>**✅ Xavier and He: Designed to Preserve Variance**

Researchers proposed two initialization methods tailored to specific activation functions:

| Init Method | Designed For | Goal |
|-------------|--------------|------|
| **Xavier (Glorot)** | Tanh / Sigmoid | Preserve both forward and backward variance |
| **He (Kaiming)**    | ReLU / Leaky ReLU | Compensate for zeroed-out negative values |


</br>

>**📌Xavier (Glorot) Initialization**

- Designed for symmetric activations like **Tanh** and **Sigmoid**.
- Assumes we want to preserve **equal variance** in both directions (forward & backward).
- Balances the scale by averaging the number of inputs and outputs.

**Formula (Uniform distribution):**

$$
W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$

👉 **If you want to try Sigmoid**, change the activation in your model from `Tanh()` to `Sigmoid()` — and use Xavier initialization just the same.

>**📌He (Kaiming) Initialization**

- Built for **ReLU-based** activations.
- ReLU discards all negative inputs — so signal is lost.
- He initialization compensates by increasing weight variance slightly.

> 🧮 **He Initialization: Two Variants**

There are **two forms** of He initialization depending on the distribution used:

1. **He Normal**

Weights are drawn from a **normal distribution**:

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

This is what we typically refer to when we say "He Initialization" in most theoretical discussions.

2. **He Uniform**

Weights are drawn from a **uniform distribution** over the range:

$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}} \right]
$$

This variant is used by **default in PyTorch** for `nn.Linear` layers when using ReLU.

> ✅ If we write:
```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```
PyTorch will automatically apply He Uniform tailored for ReLU.

> 🎯 In summary:

| Variant        | Distribution Type | Formula | When to Use         |
|----------------|-------------------|---------|----------------------|
| **He Normal**  | Normal            | 𝑁(0, 2/𝑛ᵢₙ) | When you want to explore theoretical or experimental variations |
| **He Uniform** | Uniform           | 𝕌[−√(6/𝑛ᵢₙ), √(6/𝑛ᵢₙ)] | PyTorch's default for ReLU activations |

**As we said, if we don’t explicitly set initialization, PyTorch by default uses He Uniform.**

Which means:
- If you're using **ReLU**, this default is **okay** (though not optimal).
- If you're using **Tanh** or **Sigmoid**, this is **mismatched** — use Xavier instead!

With the right initialization:
- Activations stay within a healthy range — no NaNs or zeros.
- Gradients propagate cleanly to early layers.
- Training becomes **faster**, **more stable**, and **more effective**.

We have now compared:

- **He Normal Init** vs. **Default Init (He Uniform)** for ReLU
- **Xavier Init** vs. **Default Init (He Uniform)** for Tanh

![Visualize Results - ReLU](<RelU Training Results.png>)
>**🧪 ReLU-Based Deep MLP**

| Observation | Insight |
|-------------|---------|
| 🔵 **Default Init - ReLU Uniform** and 🟠 **He Normal** show **very similar performance** in both training loss and accuracy. | This is expected — because PyTorch’s default for `nn.Linear` + ReLU is already **He Uniform**, which is well-suited for ReLU. |
| No clear gain from switching to **He Normal**. | While both preserve gradient flow, the difference between **uniform** and **normal** sampling is often minimal in practice. |

✅ **Conclusion**: For ReLU activations, PyTorch’s default initialization is already well-optimized. Explicit He Normal init shows similar outcomes.

![Visualize Results - Tanh](<TanH Training Results.png>)
>**🧪 Tanh-Based Deep MLP**

| Observation | Insight |
|-------------|---------|
| 🔵 **Default Init - Tanh** underperforms on both training loss and accuracy. | This is because the default He Uniform is **not suitable for Tanh**, which needs symmetric gradient handling. |
| 🟠 **Xavier Init** converges **faster** and achieves **better accuracy**. | Xavier was designed to preserve both forward and backward variance — making it ideal for symmetric activations like Tanh or Sigmoid. |

✅ **Conclusion**: For **Tanh (or Sigmoid)** activations, using **Xavier Initialization** gives clear improvements over the default.

---


