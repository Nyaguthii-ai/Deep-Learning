The **perceptron** is the simplest building block of a neural network.

It's a binary classifier that learns to separate two classes with a straight line (or hyperplane).
A perceptron takes **input features**, multiplies them with **weights**, adds a **bias**, and applies an **activation function** to decide the output (0 or 1). It's learning to find a line that best seperated two classes. 

**Perceptron Equation**

The perceptron computes:

    z = w(ˆT)x + b

Then applies the step activation function:

    y = {1  if z ≥ 0
        {0  otherwise
 
Where:

    x ∈ ℝˆ2 is the input vector (e.g., [feature1, feature2])
    w ∈ ℝˆ2 is the weight vector
    b is the bias (scalar)
    y is the predicted label (typically 0 or 1)

This is the decision boundary.

- If a point lies above the line → Class 1
- If a point lies below the line → Class 0

However,

- The perceptron can only learn linearly separable data.
- If the data is not linearly separable, the perceptron will not converge — it will keep updating weights forever!

The decision boundary is defined by the following equation: 

    x2 = -b ± w1x1
        _____________
            w2
            
Depending on how the model converges, the sign of w1 may be positive or negative.
Both forms describe the same decision boundary — just with flipped class sides.
