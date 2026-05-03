### **Introduction to Overfitting**

>What Is Overfitting?

**Overfitting** occurs when a model learns the training data *too well* — including noise and fluctuations — instead of capturing the underlying general patterns.

When overfitting happens:
- ✅ Training accuracy becomes very high.
- ❌ But test/validation accuracy stagnates or worsens.
- ❌ The model performs poorly on unseen examples.

>Why Does Overfitting Happen?

Overfitting usually occurs when:
- The model is **too complex** (e.g., too many neurons, layers, or parameters).
- There's **not enough data** relative to model capacity.
- **No regularization** is applied (e.g., dropout, L2 penalty).
- The model trains for **too many epochs**.

![alt text](<Overfitting vs Good fit vs Underfitting.png>)

**Underfitting**: Too simple to capture the trend 

**Good Fit**: Captures the pattern without memorizing  

**Overfitting**: Memorizes training noise — poor generalization

- The **training loss** continues to drop, indicating the model is fitting the training data increasingly well.
- However, the **validation loss** begins to **rise noticeably after ~100 epochs**, showing that generalization is deteriorating.
- This widening gap between training and validation performance is a strong indicator of **overfitting**.

>⚠️ Clear Overfitting Observed

This behavior signals that:

- The model is either **too flexible** or **too powerful** for the dataset,
- Without regularization, it starts to memorize the specific values -- patterns and noise -- in the training data,
- This results in poor performance on unseen validation data.

📌 This is a **classic deep learning pitfall**. Left unchecked, overfitting can severely hinder a model's usefulness in the real world.

>What Is Regularization?

**Regularization** refers to a set of techniques we use to **prevent overfitting** and help the model **generalize better to unseen data**.

Think of it as a way to **discourage the model from becoming too confident or too complex** — like asking it to stay humble and avoid “memorizing” the training set too perfectly.

In practice, regularization works by:
- Adding a small **penalty** for complexity (e.g., very large weights),
- Forcing the model to **rely on multiple pathways** instead of fixating on just a few neurons,
- Helping the model remain **flexible but not overly sensitive** to training noise.

### **Add Dropout Regularization**


**Dropout** is a regularization technique that helps prevent overfitting in neural networks.

During training, Dropout randomly **deactivates (drops)** a subset of neurons in a layer — typically by setting their output to zero. This means:

- The network cannot rely on any single neuron too much.
- It is forced to learn **redundant representations** that are more robust.
- Think of it as **ensemble learning inside a single network**.

Dropout is only active during training — during evaluation (`model.eval()`), all neurons are active.

>📐 Where to Apply Dropout?

Typically, Dropout is inserted **after the activation function** of a layer. For example:

Linear → ReLU → **Dropout** → Linear → ...

We'll:

1. Add `nn.Dropout(p=0.5)` after the hidden layer.
2. Retrain the model for the same number of epochs.
3. Plot the **training vs. validation loss** to observe improvements.
4. Compare results with the no-dropout model from Section 3.

> What Happens After Dropout?

- Both **training and validation losses** decrease and level off at similar values.
- The **gap between training and validation loss is narrower** than before.
- The loss curves exhibit **mild fluctuations**, typical when using Dropout — due to the randomness introduced by dropping neurons.

> 🧠 What This Suggests:

- ✅ **Dropout is helping**: It has reduced the model’s tendency to overfit the training data.
- The model now generalizes **better to the validation set**, even after many epochs.
- This is a sign of improved **robustness** and **generalization ability**.

> 💡 How Dropout Works (Intuition)

During training, **Dropout randomly "turns off" a subset of neurons** in the network on each forward pass.

- Think of it like forcing the model to solve the task using **a different team of neurons every time**.
- This prevents it from becoming overly reliant on any **single “clever” neuron** or shortcut.
- Over time, **all neurons must learn to be useful** on their own and in different combinations.

✅ As a result, the network learns **redundant, distributed representations** — like having multiple backup strategies — which helps it **generalize better** when facing unseen data.

> 🧪 Reminder:

- Dropout is only applied during training — **all neurons are active at test time**.
- The output is scaled accordingly, so the model's predictions stay consistent.

### **What is L2 Regularization?**

L2 regularization, also called **weight decay**, is a technique to **penalize large weights** in the network.  
It adds a term to the loss function that grows with the **squared magnitude of weights**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_{i} w_i^2
$$

- $\mathcal{L}_{\text{data}}$: Original loss (e.g., cross-entropy)  
- $\lambda$: Regularization rate (also called regularization penalty or weight decay)  
- $w_i$: All trainable weights in the model

>🤔 Why Penalize Large Weights?

- Large weights can make the network overly sensitive to small input variations → leads to overfitting.  
- Penalizing weights encourages **simpler models** that are **less likely to memorize noise**.

>🔧 How Do We Use It in PyTorch?

We add `weight_decay=λ` to the optimizer — no changes needed in model architecture:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```
> 📉 What Happens after L2 Regularisation:

- **Training loss** continues to decline steadily, though slightly slower than in the no-regularization case.
- **Validation loss** initially drops, then begins to **flatten out and rise slightly** around epoch 700.
- Compared to the baseline, the model **overfits later and more gradually**.

> 🔬 What This Suggests:

- L2 regularization helped **delay the onset of overfitting** — by gently constraining the model’s capacity.
- However, since we haven’t used any additional regularization (like dropout or early stopping), **some overfitting still occurs**.
- The **gap between training and validation loss is narrower** than in the baseline, suggesting improved **generalization**.

> 🧠 How L2 Regularization Works (Intuition)

L2 regularization (also called **weight decay**) discourages the model from using **very large weights** by adding a small penalty term to the loss function:

$$
\text{New Loss} = \text{CrossEntropy} + \lambda \cdot \sum w^2
$$

- It acts like a **soft brake** on weight growth — pulling large weights back toward zero.
- By discouraging large weights, the model tends to find **simpler, more balanced solutions**.
- You can think of it as telling the model: “You can fit the data, but try not to **overreact** to every detail.”

✅ This often helps the network avoid overfitting to noise or rare patterns, especially in smaller or imbalanced datasets.

> 📌 Key Insight:

L2 regularization doesn't prevent overfitting completely, but it:
- Encourages the model to find **more stable and smoother decision boundaries**,
- Helps the model resist fitting to noise,
- And improves **robustness and generalization**.

## Comparison of regularization strategies

![alt text](<Comparison of Regularization Strategies.png>)

> In summary

- **M1** (No Reg): Shows classic overfitting — loss drops early but then rises.
- **M2** (Dropout): Stabilizes training but may still fluctuate.
- **M3** (L2): Slower but more consistent convergence.
- **M4** (Dropout + L2): Best trade-off — smooth curve and generalizes well.

> **📈 2. Validation Loss Only (Overlayed for Direct Comparison)**

To make comparison even sharper, here’s a single chart showing just the validation losses of all four models:
![alt text](<Validation Loss Across Regularization Techniques.png>)

- 🔺 Look at when overfitting begins (validation loss rising).
- 🔽 See which curve stabilizes lowest — this indicates best generalization.
- 🟰 Dropout + L2 often offers a good balance of robustness and smoothness.

>**📋 Test Performance Summary**

Disclaimer: For illustrative purposes. The table results could differ from the ones you have obtained due to the randomness of some initializations.

| Model              | Test Accuracy | Macro F1 Score |
|-------------------|----------------|----------------|
| No Reg (M1)        | 58.30%         | 46.81%         |
| Dropout (M2)       | **62.78%**     | **63.01%**     |
| L2 (M3)            | 60.99%         | 62.58%         |
| Dropout + L2 (M4)  | 59.19%         | 46.35%         |


>🔍 What Do These Numbers Suggest?

✅ **Dropout alone (M2)** performs best across both metrics:
- It leads to the **highest test accuracy** and **strongest macro F1**, suggesting the model is **not only correct more often** but also **more balanced across classes**.
- Dropout seems to encourage the network to **learn diverse and redundant features**, which enhances generalization.

✅ **L2 regularization (M3)** also performs well, especially in terms of **macro F1**, indicating it effectively manages **class-level balance**.

⚠️ **No Regularization (M1)** shows both **lower accuracy and F1**, suggesting the model may be **overfitting** — it performs well on training data but struggles with generalization.

⚠️ **Dropout + L2 (M4)** underperforms in this setting. While both techniques are useful individually, combining them without tuning may result in **excessive regularization**, causing the model to underfit.

>🧠 Intuition Behind the Results

- **Dropout** acts like a training-time challenge: by randomly turning off neurons, it forces the network to **not rely too heavily on any single pathway**. Over time, this encourages **robust feature learning** and discourages memorization.
- **L2 regularization** discourages large weights, nudging the network toward **simpler, more stable solutions**.
- When used together (M4), these methods might **interact in a way that’s too restrictive**, limiting the network’s capacity to learn meaningful structure from the data — unless carefully tuned.

>🎯 Key Takeaways

- ✅ **Regularization is essential**, but choosing the right method — or combination — requires experimentation.
- ✅ **Macro F1 is especially important** for imbalanced tasks: it tells us how well we're serving *all* classes, not just the majority.
- ✅ **Dropout (M2)** appears to strike the best balance in this setup: the network generalizes well and handles class imbalance effectively.

>**What If We Change the Architecture?**

The current model uses **one hidden layer with 32 neurons** — a moderately expressive architecture. But what if we changed that?

- **Increase the number of neurons (e.g., 64)?**: The model becomes more expressive and can learn more complex patterns. But this also increases the risk of **overfitting**, especially if the dataset is small or noisy. Regularization techniques become even more important in such cases.

- **Decrease the number of neurons (e.g., 16)?**: The model becomes simpler and faster to train, but may struggle to capture the richness of the data — leading to **underfitting**. You might see both training and validation performance suffer.

- **Add more hidden layers?**: The model can learn **deeper hierarchical representations**, which is great for complex tasks — but also makes training more delicate. You may need **more data**, **better regularization**, and strategies like **batch normalization** or **residual connections** to ensure stable training.