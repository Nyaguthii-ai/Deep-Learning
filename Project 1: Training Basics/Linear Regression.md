# Linear regression 
Models the relationship between inputs and outputs by assuming a straight-line relationship between them.

A neural network with no hidden layers and no activation functions behaves exactly like a linear regression model.

Our equation is : 

**¥ = XW + b**

    *X* = Input features (matrix), (n,d)= n samples × d features

    *W* = Weights (vector/matrix), (d, 1)

    *b* = Bias (scalar or vector)	Scalar or (n, 1)

    *¥* = Predicted outputs (n, 1)	

So this model performs a linear transformation of the inputs, followed by a bias shift.
The multiplication gives us a weighted sum of features, and lets us adjust the output.

XW: Weighted sums -> shape (n X 1)

When building models with PyTorch—tensor shapes must match for the operations to work!

# Loss function
In supervised learning, a loss function compares the model's predictions ¥, to the true target values y, and returns a single number—a score that tells us how far off we are.

The most common loss function is the Mean Squared Error (MSE).

**MSE Loss = (1/n) * Σ(y_true - y_pred)^2**

Where:  
    yi = the actual value

    ¥i = the predicted value

    n = number of examples

# Training a linear model
When training we have a, Training set: used to update weights and a, Test set: used to measure generalization.
We use a typical **80-20** split. With the Training set taking the larger share.

If weights are too large at the beginning, the model might:
Produce massive outputs
Suffer from exploding gradients

We initialize:
    W as small random values (~N(0, 0.01))
    b as zero

Also, by setting requires_grad=True, PyTorch will track gradients for these variables during training.

# Gradient descent
To train the model using gradient descent, PyTorch needs to compute the gradients of the loss with respect to the parameters: **ø = {W,b}**

We tell PyTorch which tensors to track for gradients by setting:

    W = torch.randn((8, 1), requires_grad=True)

    b = torch.zeros((1,), requires_grad=True)

This enables **Automatic Differentiation** in PyTorch: PyTorch builds a computation graph behind the scenes and stores all operations involving W and b.

When we later call **.backward()** on the loss, PyTorch uses the chain rule to comput the gradient loss with respect to each parameter.

**The Goal: Minimize the Loss**
Our model is only useful if it makes accurate predictions.
To measure how far off it is, we use a loss function, which outputs a scalar value— the total error across the training data.

🎯 Our goal is to find the values of **ϕ** that make this loss as small as possible.

This is called an **optimization problem**.

The gradient is like a compass—it tells us:

    📉 Which direction will reduce the loss (the sign of the derivative)

    📏 How steeply the loss changes (the magnitude)

**Step-by-Step Update (Per Parameter)**

The learning rate α controls how fast we move toward the minimum:

When α:

Too small  ->  Learns very slowly

Too large  ->  Overshoots or diverges

Just right ->  Steady improvement in loss

    1. Compute loss  L[ϕ]      ← How far off are our predictions?
    2. Compute gradient ∇ϕ L   ← How should we change W and b?
    3. Take a step:            ← Update each parameter:
        ϕ ← ϕ - α ⋅ ∇ϕ L
    4. Repeat for many epochs
        X, y
            ↓
        forward pass: compute ŷ = XW + b
            ↓
           loss = MSE(ŷ, y)
            ↓
        backward pass: compute gradients ∇W, ∇b
            ↓
        update: 
            W ← W - α ⋅ ∇W
            b ← b - α ⋅ ∇b

#Training Loop

In every training epoch (iteration), we perform the following steps:

    1. Forward pass – Compute predictions: y = XW + b
    2. Loss computation – Measure how far off the predictions are
    3. Backward pass – Compute gradients via .backward()
    4. Parameter update – Adjust W and b using gradient descent
    5. Gradient reset – Clear old gradients before the next epoch

**NOTE:** We must reset the gradient and bias after every loop manually using:

    W.grad.zero_()
    b.grad.zero_()

❗ If we don’t clear the gradients, we'll add new gradients on top of the old ones, leading to incorrect updates during training.







