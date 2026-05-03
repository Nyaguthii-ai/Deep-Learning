# Multi-Layer Perceptron 
An MLP is a type of neural network with:

- One or more hidden layers
- Each layer performs a linear transformation followed by a non-linear activation
- A final output layer makes predictions

**🧠 Simple Neural Network Architecture**

Inputs (8 features from dataset)
⬇️

Linear Layer (8 → 16)
⬇️

ReLU Activation
⬇️

Linear Layer (16 → 1)
⬇️

Output: Predicted concrete strength

In deep learning, we train models by adjusting internal weights and biases — these are called **parameters**.
But there’s another kind of setting that we must choose ourselves before training begins:
those are called **hyperparameters**.

**Definition:**

✅ Hyperparameters are configuration values not learned from data.

We define them before training to control how the model behaves and learns.

These choices greatly affect:
- How fast the model learns 🐢⚡
- Whether it overfits or underfits the data 🎯
- How well it generalizes to new data 🌍

**Common Hyperparameters (Cheat Sheet)**

        HYPERPARAMETER                  WHAT IT CONTROLS	                                TYPICAL VALUES
        Learning Rate (lr)	        How big a step we take to update weights	            0.001 to 0.1
        Number of Epochs	        How many times we loop through the training set	        100, 300, 1000+
        Hidden Layer Size	        How many neurons in each hidden layer	                16, 32, 64, ...
        Number of Hidden Layers	    Depth of the network (1 layer? 2? more?)	            1–3 (or more)
        Batch Size	                How many samples we use to compute gradients	        16, 32, 64, 128
        Activation Function	        The non-linearity we apply (ReLU, Tanh, Sigmoid etc.)	ReLU (common)
        Optimizer	                Algorithm for adjusting parameters	                    SGD, Adam, RMSprop

- Too large a learning rate? ❌ Model never converges.
- Too few epochs? ❌ Model underfits.
- Too small hidden layer? ❌ Can’t capture patterns.
- Wrong activation? ❌ Poor gradient flow.

**Why Use a Custom Class for an MLP?**

While **nn.Sequential** is a great shortcut for prototyping models, in most real-world scenarios we define our models using custom classes. 

This provides:
- Greater flexibility and control over architecture
- Easier debugging and extension
- More expressive forward passes

We define a model by creating a class that:

- Inherits from **nn.Module**
- Defines the layers in the __ _init_ __() method
- Describes the forward pass in a **forward()** method

