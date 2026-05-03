### Training the MLP on CIFAR-10

We now train our `SimpleMLP` using:

- **Loss**: `CrossEntropyLoss`, ideal for multiclass classification.
- **Optimizer**: `Adam`, a widely used optimizer combining momentum and adaptive learning rates.
- **Epochs**: We'll train for a few epochs and observe the trends in loss and accuracy.

**Key Concepts**

**Loss Function**: We use `nn.CrossEntropyLoss` which combines **log-softmax** and **negative log-likelihood**. It expects raw logits as model output and integer class labels as targets

**Optimizer**: We use `Adam` for efficiency and stability. We'll update weights based on gradients computed by backpropagation.

**Metrics**: We’ll track **Training loss** (average per epoch) and **Training accuracy** (proportion of correct predictions).

After 10 epochs, here’s what we observed:

- **Initial loss**: 1.89 → **Final loss**: 1.52
- **Accuracy improved** from **32% to ~46%** over the epochs

The model is **learning steadily** — loss is decreasing and accuracy is improving. But performance is still **far from ideal**. With **only one hidden layer** and **no spatial awareness**, it’s expected that:

  - The MLP **struggles to capture image structure** (e.g., edges, shapes, textures)
  - It **treats pixels as independent features**, which is suboptimal for images
  - The model is **underfitting**, though improving

>🔧 What Can Help?

- **Deeper MLPs** (more hidden layers)
- **Better optimizers or learning rate tuning**
- **Convolutional Neural Networks (CNNs)** — designed to leverage **spatial locality**

### Why MLPs Struggle with Image Data

Even after training a fully connected neural network (MLP) on CIFAR-10:
- Accuracy plateaued around **45–46%**.
- Confusion matrix showed **lots of misclassifications**, especially for visually similar categories.

>**But Why?**

**1. 🧱 Loss of Spatial Structure**

MLPs require us to **flatten images** into long 1D vectors.

- A 32×32 color image becomes a **3072-length vector** (32 × 32 × 3).
- All **spatial relationships** between neighboring pixels are **destroyed**.
- This makes it harder for the model to learn **local patterns** (e.g., edges, corners, textures).

The model sees no 2D structure — just a long list of numbers.

**2. 📏 Too Many Parameters**

Because every pixel is connected to **every neuron**, the number of parameters explodes:

- For a simple MLP with 3072 inputs and a hidden layer of 256 neurons:
  → **3072 × 128 = 3,93,216 weights** in the first layer alone!

This leads to **High memory usage**, **Overfitting risk** (especially with limited data), and **Slower convergence** during training.

**3. 🚫 No Inductive Bias**

MLPs have **no built-in assumptions** about images. They treat all input features (pixels) as equally important, regardless of **position** or **locality**.

Contrast this with:
- **Convolutional Neural Networks (CNNs)**, which:
  - Exploit local patterns (like edges, textures)
  - Share weights (via kernels)
  - Have fewer parameters
  - Learn hierarchical representations