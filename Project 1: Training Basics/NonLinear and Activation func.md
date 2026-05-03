# Non Linear and Activation Functions

To give our models nonlinear expressive power, we need a way to bend and shape the learned function.

This is the role of **activation functions**.

An **activation function** is a nonlinear transformation applied to the output of a neuron (or layer).

                           Activation
            Linear Output -----------> Non Linear Output

# Common Activation Functions

Let’s explore three popular activation functions:

**🔹 ReLU (Rectified Linear Unit)**
Formula: f(x) = max(0,x)

- Outputs zero if x<0, and x otherwise
- Very simple and efficient
- Helps combat the vanishing gradient problem

If Input: tensor([-2., -1.,  0.,  1.,  2.])

ReLU Output: tensor([0., 0., 0., 1., 2.])

This confirms that:

- All negative values are converted to 0
- Positive values are passed through unchanged

**🔹 Sigmoid**
Formula: 

        f(x) = 1
            --------
            1+e^-x

- Squeezes output between 0 and 1
- Often used in binary classification problems
- Can saturate and cause gradients to vanish

If Input: tensor([-2., -1.,  0.,  1.,  2.])

Sigmoid Output: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

This confirms:

- Values < 0 are squeezed toward 0
- Values > 0 are squeezed toward 1
- The midpoint (0) maps to 0.5

**🔹 Tanh (Hyperbolic Tangent)**
Formula: 
    
        f(x) = tanh(x) = e^x - e^-x
                        ------------
                        e^x + e^-x

- Squeezes output between -1 and 1
- Like sigmoid, but centered at 0, which often helps training

Each of these functions introduces nonlinearity, enabling neural nets to stack and compose complex feature transformations.

Their activation curves look as follows:
- **ReLU** Range [0, ∞], Flat for x < 0, linear for x > 0

- **Sigmoid**	Range [0, 1], S-curve, flattens at both ends

- **Tanh** Range [ -1, 1], S-curve centered around zero

