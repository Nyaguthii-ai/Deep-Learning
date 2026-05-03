# Backpropagation 

**What is Backpropagation?**

> Backpropagation is the algorithm used to **compute the gradients** of a loss function with respect to all weights and biases in the network.

**Understanding Backpropagation Flow**

The diagram below illustrates the **forward and backward passes** in a feedforward neural network with:

- 1 hidden layer (ReLU activation)  
- 1 output layer (Sigmoid activation)  
- A scalar loss $L$

**Forward Pass (Purple Blocks)**

From left to right, the input $\mathbf{x}$ is transformed step by step:

1. **Hidden Layer (Linear)**  
   $$
   z^{[1]} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1
   $$

2. **Activation (ReLU)**  
   $$
   a^{[1]} = \text{ReLU}(z^{[1]})
   $$

3. **Output Layer (Linear)**  
   $$
   z^{[2]} = \mathbf{W}_2 a^{[1]} + \mathbf{b}_2
   $$

4. **Activation (Sigmoid)**  
   $$
   \hat{y} = a^{[2]} = \sigma(z^{[2]})
   $$

Finally, we compute the **loss** $L(y, \hat{y})$ — often binary cross-entropy.


**Backward Pass (Brown Blocks)**

Backpropagation starts from the loss and flows backward using the **chain rule**.  
At each step, we compute gradients:

- $\frac{\partial L}{\partial a^{[2]}}$, then  
- $\frac{\partial L}{\partial z^{[2]}}$, then  
- $\frac{\partial L}{\partial a^{[1]}}$, and so on...

Each red box represents a **partial derivative**, and the arrows represent **chaining gradients** together using multiplication.


**Gradient of Loss with Respect to $\mathbf{W}_1$**

At the bottom, we summarize the final gradient needed to update the first layer’s weights:

$$
\frac{\partial L}{\partial \mathbf{W}_1} = 
\frac{\partial L}{\partial a^{[2]}} \cdot
\frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot
\frac{\partial z^{[2]}}{\partial a^{[1]}} \cdot
\frac{\partial a^{[1]}}{\partial \mathbf{W}_1}
$$

Each component corresponds to one stage in the network,  
and their multiplication gives us the full gradient we need.

### 🧩 Key Learning Point

> **Backpropagation** breaks down a complex gradient into small, manageable steps — each one following the **chain rule**.

This is what allows us to **train deep neural networks efficiently**  
without computing complex symbolic derivatives from scratch.

**Goal: Learn via Gradient Descent**

We want to **minimize a loss function**, e.g., Mean Squared Error (MSE):

$$
\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2
$$

**Parameter Update Rule (Gradient Descent)**

When we train a neural network, we **adjust the parameters** to minimize the loss.

The general update rule is:

$$
\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}
$$

Where:

- $\theta$: a parameter (e.g., a weight or bias)  
- $\eta$: learning rate (a small positive scalar)  
- $\frac{\partial \mathcal{L}}{\partial \theta}$: the gradient of the loss with respect to that parameter

**📌 Note on Loss Function: When to Use**

The loss function:

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

is called **mean squared error (MSE)** and is commonly used for **regression tasks**, where the target $y$ is a **continuous number**.

✅ **Use this** when predicting **numerical values** (e.g., price, temperature, etc.)

- The $\frac{1}{2}$ is optional — it simplifies the derivative when doing backpropagation manually.


❗️ **Do NOT use this** for **binary classification** tasks (like disease vs. no disease).

➡️ For binary classification, we use:
- A **sigmoid** activation at the output
- A **binary cross-entropy** loss function:

$$
L = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
$$

This gives us **probabilities** and a more appropriate **loss surface** for classification tasks.

---


**Chain Rule in Action**

Our network is:

```text
x → z₁ = xW₁ + b₁ → h = ReLU(z₁) → ŷ = hW₂ + b₂ → L = ½(ŷ - y)²
```

We apply the **chain rule** to compute gradients backwards.


### Gradients for Output Layer

1. Derivative of loss w.r.t. prediction:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y
$$

2. Gradients for output weights $W_2$ and bias $b_2$:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = (\hat{y} - y) \cdot \mathbf{h}^\top
$$

$$
\frac{\partial \mathcal{L}}{\partial b_2} = \hat{y} - y
$$


**Gradients for Hidden Layer**

Let’s go backward into the hidden layer:

3. Derivative w.r.t. hidden layer activation:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = (\hat{y} - y) \cdot W_2^\top
$$

4. If we used **ReLU** activation:

$$
\frac{\partial \mathbf{h}}{\partial \mathbf{z}_1} =
\begin{cases}
1 & \text{if } z_1 > 0 \\
0 & \text{otherwise}
\end{cases}
$$

So:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \circ \text{ReLU}'(\mathbf{z}_1)
$$

($\circ$ means **element-wise multiplication**)

5. Gradients for first layer weights $W_1$ and bias $b_1$:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \mathbf{x}^\top \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}
$$

**Tracking Prediction for One Input Over Time**

```text
Input x = [x₁, x₂]  ← fixed sample (e.g., [2.0, -1.0])
        │
        ▼
Hidden Layer: Linear → ReLU
  z₁ = x · W₁ + b₁
  h = ReLU(z₁)
        │
        ▼
Output Layer: Linear → (optional) Sigmoid
  z₂ = h · W₂ + b₂
  ŷ = z₂  (or sigmoid(z₂))
        │
        ▼
Compare ŷ vs. y
        │
        ▼
Track prediction (ŷ) over epochs
        │
        ▼
Plot: ŷ vs. Epoch Number

