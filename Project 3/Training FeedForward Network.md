# Training FeedForward Networks

>**Why Standardization Matters**

Neural networks learn by adjusting weights based on gradients — changes in the loss with respect to input values. If one feature (say, mcg) ranges from 0 to 1, and another (vac) ranges from 0 to 1000, the network might focus more on the large-scale feature, simply because its values dominate the computation. This can cause:

- Unstable gradients: Larger input values may lead to exploding or vanishing gradients during backpropagation.

- Slower convergence: The optimizer has to work harder to balance updates across differently scaled features.

- Poor performance: The network may get stuck in suboptimal solutions or take longer to generalize well.

✅ Standardization brings all features to a similar scale (mean = 0, std = 1), which makes learning smoother, faster, and more reliable — especially when using gradient-based optimizers like SGD or Adam.

So, we standardize to bring all features to the same scale and create training and test sets: with stratified sampling to preserve class ratios.

**Training Loop**

**🎯 Goal:** Train our MLP model using a standard supervised learning loop in PyTorch.

>**The Training Workflow**

Training a neural network involves the following cycle for each epoch:

1. **Forward Pass**: Compute predicted outputs (`logits`) from the model.
2. **Loss Computation**: Compare predictions against the true labels using a loss function.
3. **Backward Pass**: Compute gradients of the loss w.r.t. model parameters.
4. **Parameter Update**: Adjust the weights using an optimizer like SGD or Adam.
5. **Repeat** for multiple epochs to minimize the loss.


>Loss Function: `CrossEntropyLoss`

For multiclass classification, we use: $$\text{Loss} = -\log \left( \frac{e^{z_{\text{true class}}}}{\sum_j e^{z_j}} \right)$$

- Input: **Raw logits** from model (not softmaxed)
- Target: Class indices (e.g., 0–9 for 10 classes)

>Optimizer: `torch.optim.Adam`

Adam combines momentum and adaptive learning rate for efficient training.  
We don’t tune hyperparameters here — we’ll discuss optimizers more deeply in the next notebook.

>Training Tips

- **Shuffle** the data during training (`DataLoader` helps with this in future notebooks).
- Always **zero out gradients** before calling `.backward()`.
- Track **training accuracy** along with loss to assess learning progress.
- Loss: Measures how far predictions are from true labels. We want it to decrease. Shows gradient behavior
- Accuracy: Percentage of correctly classified samples. We want it to increase. Helps detect overfitting
- Detect problems like underfitting (loss not decreasing) or overfitting (accuracy peaks then drops).

>**Confusion Matrix**, which shows us:
- Which classes are hardest to predict
- Where the model is most confused
- Whether rare classes are ignored

Each row represents the **true class**, and each column the **predicted class**.

- **Diagonal cells** represent **correct predictions** — higher values here mean the model is more accurate for that class.
- **Off-diagonal entries** reflect **misclassifications** — showing which classes are being confused.
