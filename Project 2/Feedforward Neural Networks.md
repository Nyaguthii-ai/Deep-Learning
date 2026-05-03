# Feedforward Neural Networks (FNNs)
A Feedforward Neural Network:

- Stacks multiple neurons into layers
- Allows intermediate hidden layers
- Uses non-linear activation functions to model complex patterns

Let’s say our input is a vector x ∈ ℝˆ2 (e.g., two features from the heart dataset).

We define a simple Feedforward Neural Network (FNN) with:

- One **hidden layer** of *3* neurons
- One **output neuron**

Let’s walk through the forward pass step by step:

Hidden layer linear transformation:
    hlin = x.W1 + b1

Apply activation (e.g., sigmoid or ReLU):
    h = ø(hlin)

Output layer:
    y = h.W2 + b2

The hidden layer acts like a **feature transformer**.

Non-linear activation functions (like ReLU or Sigmoid) allow the network to:
- Bend the decision boundaries
- Capture non-linearly separable patterns

**🔍 What is a Logit?**

In binary classification, the output of the neural network (before applying the sigmoid function) is called the **logit**.

This is the value we use to compute the probability:

$$
\text{Probability} = \text{sigmoid}(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}
$$

We plot the **raw logit values** in the plot above to understand how the network’s output varies across the input space. The predicted **logits** are visualized as colors across a 2D input grid, helping us see **how confident** the model is about its decision across different regions.




