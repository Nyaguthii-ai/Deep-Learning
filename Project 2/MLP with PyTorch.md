## MLPs with PyTorch

### **1. From Manual Networks to PyTorch – Why Abstract?**

> 💡 Real-world models have:
> - Many layers and neurons
> - Millions of weights
> - Complex loss functions
> - Multiple datasets and devices
> - The need for rapid experimentation

Manually coding all that becomes:

❌ Tedious  
❌ Error-prone  
❌ Slow to iterate

**What PyTorch Automates For Us**

| Task                         | Manual (NB03)     | PyTorch (NB04)        |
|------------------------------|-------------------|------------------------|
| Forward pass                 | You wrote it      | Handled by `nn.Module` |
| Gradient computation         | You derived it    | Handled by `autograd`  |
| Weight update (SGD)          | You coded it      | Handled by `optimizer.step()` |
| Loss calculation             | You used MSE      | We’ll use `BCELoss` / `BCEWithLogitsLoss` |
| Batch handling / vector ops  | You looped        | Efficient tensor ops   |


**The Philosophy of Using Frameworks**

We don’t use PyTorch because we *can’t* do it ourselves —  
we use it because it lets us focus on **designing**, **training**, and **improving** models — not on reinventing gradient mechanics.

**Why `Sigmoid` at the Output?**

- Because we want the model’s output to be a **probability** between 0 and 1.
- This matches with the **binary target labels** (0 or 1).
- We’ll later use **`BCELoss`** which expects probabilities.

**Parameter Count**

- Layer 1 (Linear 13 → 16): 13 × 16 weights + 16 biases = 224
- Layer 2 (Linear 16 → 1): 16 weights + 1 bias = 17
- **Total trainable parameters** = 241

Loss Function: nn.BCELoss

We are doing binary classification, so we’ll use the Binary Cross-Entropy function, also known as the log loss function
    loss_fn = nn.BCELoss()

Why BCELoss?

It expects model outputs to be probabilities (which we get from Sigmoid)
It compares predicted probabilities with true binary labels (0 or 1)
Optimizer: torch.optim.Adam

For training the model, we use the Adam optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

Why Adam?

Adaptive learning rate for each parameter
Generally works well as a default choice
Less sensitive to initial learning rate

**Confusion Matrix**

The **confusion matrix** shows how many samples were:

- **Correctly classified** (True Positives and True Negatives)
- **Incorrectly classified** (False Positives and False Negatives)

**Interpretation:**

- High values on the **diagonal** = good!
- Off-diagonal values suggest **misclassifications**
- This helps us understand **model strengths and weaknesses**

> ✅ If your model shows decent accuracy and balanced predictions, that means it has **learned to generalize** — not just memorize.


