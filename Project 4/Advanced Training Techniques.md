### ** Advanced Training Techniques**
> The reality is: initialization is necessary, but not sufficient.

As we go deeper, we face new challenges that affect not just **whether learning happens**, but also **how fast and stable** that learning is — and how well the model generalizes to unseen data.

> **🚧 Why Deep Networks Still Struggle**

1. **Unstable Gradients**
- Gradients can still **vanish** (become tiny) or **explode** (become huge), especially across many layers.
- This leads to:
  - **Stalled or slow learning**
  - **Divergence** (loss becomes NaN or shoots up)
  - Layers updating at **different speeds**, leading to instability

2. **Fragile Training Dynamics**
- Deeper networks often need **careful tuning** and **more epochs** to converge.
- A single mistake — like a bad learning rate — can **ruin training**.

3. **Internal Covariate Shift**
- As parameters update during training, the **distribution of activations** changes layer-by-layer.
- This forces deeper layers to **constantly adapt** to a moving target — slowing learning and increasing variance.

>**🛠️ How Can We Help Deep Networks Learn?**

In this notebook, we’ll explore three **core techniques** that have become standard in modern deep learning:

| 💡 Idea                  | 🚀 How It Helps |
|--------------------------|-----------------|
| **Batch Normalization**   | Normalizes layer inputs to reduce internal covariate shift, stabilize and speed up training. |
| **Adaptive Optimizers**   | Optimizers like Adam and RMSProp adjust learning rates dynamically — helping the model learn faster. |
| **Regularization Methods**| Dropout and L2 penalty reduce overfitting and improve generalization performance. |

Each of these targets a **specific pain point** we often face when training deep models.

![Observing the limitations of the baseline MLP Model](<Baseline MLP Model.png>)
>🔍 What Did We Observe?

- **Training Loss** steadily decreased across epochs — showing that the model is learning from the data.
- **Validation Loss** initially followed training loss but later **flattened and fluctuated**, a common sign of **overfitting or unstable generalization**.
- **Accuracy curves** tell a similar story:
  - **Training Accuracy** continued to improve smoothly.
  - **Validation Accuracy** rose at first but then **plateaued or dipped**, failing to match the training performance.

> >**⚠️ Challenges Observed in the Baseline Model**

| Issue                     | What We Saw                                   | Why It Matters                           |
|--------------------------|-----------------------------------------------|------------------------------------------|
| **Slow Convergence**     | Gradual improvement in loss and accuracy      | Training takes longer, inefficient learning |
| **Overfitting Signals**  | Train accuracy >> Val accuracy after some epochs | Model may not generalize well             |
| **Fluctuating Validation** | Val loss/accuracy bounce around mid-training | Model may be sensitive to noise or lack regularization |
| **Poor Gradient Flow**   | Especially in early layers (as seen in NB02)  | Some layers may not be learning at all   |

</br>

>✅ Why This Baseline Is Important

This baseline highlights **real pain points** in training deep MLPs:
- No **Batch Normalization** → unstable training dynamics.
- No **weight regularization** → overfitting kicks in early.
- Simple **SGD optimizer** → slower adaptation compared to modern optimizers.

It gives us a solid reference point for improvement.

### Add Batch Normalization

**🧠 Why Batch Normalization?**

As we saw in our baseline training:

- Deeper networks struggle with **unstable gradients** and **slow learning**.
- Even with good initialization, learning can be **fragile and noisy**.

**Batch Normalization (BatchNorm)** was proposed to address these challenges. It works by **normalizing the inputs to each layer** within a mini-batch during training.

>🔬 What Does BatchNorm Do?

Think of each hidden layer as receiving inputs that keep changing during training — this is called **internal covariate shift**. BatchNorm fixes this by:

1. **Normalizing activations**: It ensures each layer receives inputs with **mean ≈ 0** and **std ≈ 1**.
2. **Stabilizing gradients**: Makes backpropagation smoother across all layers.
3. **Allowing faster training**: Enables use of **higher learning rates**.
4. **Acting as a mild regularizer**: Helps reduce overfitting (often reduces need for Dropout!).

>📊 Where Is BatchNorm Applied?

For fully connected (MLP) networks:
> We apply `BatchNorm1d` **after each `Linear` layer** and **before the activation** (`ReLU`, `Tanh`, etc.).

> Compare baseline MLP with the Batch Normalization MLP
![Comparison of baseline MLP with the Batch Normalization MLP](<Baseline MLP Vs Batch Normalization.png>)
Let’s interpret the comparison plots above:

- ✅ **Faster convergence**: The model with BatchNorm reaches higher accuracy **much earlier** than the baseline, both on training and validation sets.

- ✅ **Smoother training**: Training and validation **loss curves are more stable**, showing reduced noise and more consistent learning.

- ✅ **Better generalization**: The BatchNorm model not only trains faster — it also **achieves higher validation accuracy**, especially in early epochs. That means it’s learning patterns that **generalize better to unseen data**.

>**💡 Why Does BatchNorm Help?**

Batch Normalization improves training dynamics in several ways:

| 🔍 Mechanism                        | 🧠 How It Helps                                                    |
|------------------------------------|---------------------------------------------------------------------|
| **Normalizes activations**         | Keeps inputs to each layer in a **similar range** — avoids instability |
| **Reduces internal covariate shift** | Makes each layer less sensitive to changes in the previous layers   |
| **Stabilizes gradient flow**       | Avoids vanishing/exploding gradients — helps deeper layers learn    |
| **Acts like regularization**       | Adds noise during training — reduces overfitting slightly           |

### Introduce Optimizers
![Comparing the performances of SGD, SGD+ MOMENTUM, ADAM and RMSPROP](<Comparing the Optimizers.png>)

Based on the plots comparing `SGD`, `SGD + momentum`, `Adam`, and `RMSProp`, here’s what we observe:

**1. Training Loss (Top-Left Plot)**

- **Adam** and **RMSProp** show the **fastest and smoothest decrease** in training loss.
- **SGD + Momentum** improves over plain SGD, but still trails behind adaptive methods.
- **SGD** converges slowly — it struggles to efficiently update weights in deep networks.

**2. Validation Loss (Top-Right Plot)**

- All optimizers improve early on, but:
  - **Adam** and **RMSProp** reach **lower validation loss** faster.
  - **SGD** has **higher and more unstable** validation loss over time.
- After ~10 epochs, most curves start to **plateau or rise**, hinting at early **overfitting**.

**3. Training Accuracy (Bottom-Left Plot)**

- **Adam** and **RMSProp** again outperform others in training accuracy.
- **SGD + Momentum** makes a clear improvement over plain SGD.
- **SGD** stays behind — its updates are slower and less effective.

**4. Validation Accuracy (Bottom-Right Plot)**

- **Adam** and **RMSProp** achieve the **best final accuracy**, peaking around **0.54**.
- **SGD + Momentum** follows closely and performs respectably.
- **SGD** levels off early — showing weaker generalization.

>**📌 Summary Table**

| Optimizer         | 🏃 Train Speed | 🎯 Final Val Acc | 📉 Stability      | 🔧 Best For                        |
|------------------|----------------|------------------|-------------------|------------------------------------|
| **SGD**          | ❌ Slow         | ⚠️ Lower          | ❌ Fluctuates      | Simple models / intro-level demos  |
| **SGD + Momentum** | ⚠️ Moderate     | ⚠️ Fair           | ✅ More stable     | Safer alternative to plain SGD     |
| **Adam**         | ✅ Fast         | ✅ High           | ✅ Smooth          | Most practical default             |
| **RMSProp**      | ✅ Fast         | ✅ High           | ✅ Stable          | Similar to Adam, slightly noisier  |

### Refer to Dropout + L2 Regularization from Week 3

Regularization techniques aim to:
- **Prevent over-reliance** on specific neurons or weights
- **Encourage smoother, simpler** decision boundaries
- **Improve robustness** to noise or data shifts

>**🔎 Two Common Regularization Strategies**

>🔹 Dropout

*What it does:* Randomly “drops” neurons during training.

Each training step uses a slightly different sub-network by **turning off (zeroing out)** some neurons with probability `p`. This forces the network to:
- Not rely too heavily on specific neurons
- Learn **redundant representations**
- Act like **an ensemble of smaller networks**

Mathematically, if `rᵢ ∼ Bernoulli(p)` is a binary mask:
$$
\tilde{x}_i = x_i \cdot r_i
$$

During inference (testing), we **scale activations down** instead of applying dropout — to match the expected behavior.

🧠 Think of it as: “Shuffling which neurons are allowed to speak each time — so no one neuron becomes a dictator.”

>🔹 L2 Weight Decay (a.k.a. Ridge Regularization)

*What it does:* Penalizes **large weights** in the loss function.

A secondary term is added to the loss:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}} + \lambda \sum w^2
$$

Why it helps:
- Encourages **smaller weight magnitudes**
- Reduces model complexity (smoother decision boundaries)
- Helps prevent overfitting, especially with many features or layers

**Note**
In our model, we’re applying **two complementary regularization techniques**:

>Dropout (in the Model)
- Dropout is applied **during the forward pass**.
- It randomly "turns off" neurons (e.g., 20%) to prevent **co-adaptation**.
- This forces the network to learn **robust and redundant features**.

>L2 Regularization (in the Optimizer)
- L2 is applied **during the optimizer step** via the `weight_decay` parameter.
- It penalizes large weights using:
  $$
  \text{Loss}_{\text{total}} = \text{Loss}_{\text{CE}} + \lambda \sum w^2
  $$

![Regularization Impact: BN+Adam vs BN+Adam+Dropout+L2](<Comparison with regularization.png>)
>📉 Training Loss & Accuracy

- The **BN + Adam** model (without regularization) shows **lower training loss** and **higher training accuracy** throughout.
- This is expected — **regularized models are intentionally penalized** during training to encourage better generalization.

>📊 Validation Loss & Accuracy

- Despite underperforming on training metrics, the **BN + Adam + Dropout + L2** model maintains **competitive validation accuracy** and **lower validation loss** in later epochs.
- This suggests that regularization is helping the model **avoid overfitting**, especially toward the end of training.

>🧠 **What This Tells Us**

| Metric         | BN + Adam ✅                     | BN + Adam + Dropout + L2 ✅ |
|----------------|----------------------------------|-----------------------------|
| Train Loss     | Lower (fits training well)       | Higher (due to Dropout & L2) |
| Train Accuracy | Higher (possibly overfitting)    | Lower (regularized learning) |
| Val Loss       | Rises in later epochs            | Remains lower & stable      |
| Val Accuracy   | Peaks early, then fluctuates     | Slightly more stable        |

➡️ **Regularization may slow learning**, but helps prevent the model from "memorizing" training data — especially when we care about **performance on unseen data**.

> **⚠️ When Regularization Might Not Help**

There are cases where Dropout or L2 regularization might **hurt performance**:

- When the model is already **underfitting** (not learning well even on training data)
- When the **dataset is small** or very **clean and simple**
- When **training is short** and the model hasn’t yet started to overfit

In such cases, regularization might **suppress learning unnecessarily**.

![Evaluate the accuracy, F1 Score and Inference Time through the models](<Advanced Techniques Eval.png>)

Let’s compare the final models based on:

>**Evaluation Metrics**

| Metric         | What It Measures                                                                 |
|----------------|-----------------------------------------------------------------------------------|
| ✅ **Test Accuracy** | Overall proportion of correct predictions.                                 |
| 🧠 **Macro F1 Score** | Harmonic mean of precision and recall, **averaged across all classes**.     |
| ⏱️ **Inference Time** | Time taken to predict the entire test set. Useful for comparing efficiency. |

After training and evaluating multiple deep MLP variants, here’s what we observed on the test set:

| Model                        | Test Accuracy | Macro F1 Score |
|-----------------------------|---------------|----------------|
| **Baseline (SGD)**          | 51.56%        | 0.5147         |
| **BN (SGD + Momentum)**     | 51.85%        | 0.5164         |
| **BN + Adam**               | 52.72%        | 0.5240         |
| **BN + Adam + Dropout + L2**| **53.59%**    | **0.5298**     |

</br>

- **Each enhancement improved performance** over the baseline:
  - **BatchNorm** offered modest gains in both accuracy and stability.
  - **Adam** sped up convergence and improved generalization.
  - **Dropout + L2** further boosted test performance by reducing overfitting.

- **Regularization + Adam + BatchNorm** was the best combination in this experiment.
