## Feature Extraction from Pretrained Models

We explored how a deep convolutional neural network (CNN) such as ResNet-18 learns hierarchical features from data.  
The early layers detect simple patterns like **edges**, **corners**, and **color gradients**, while deeper layers capture more **abstract and semantic concepts** such as shapes, textures, or even object parts.

By the time the model reaches the final fully connected layer, it has transformed raw pixel information into **high-level feature representations** that summarize the essential characteristics of an image.  
This rich representation is what allows us to reuse pretrained models for new tasks — a process we call **transfer learning**.

In NB01, we froze all pretrained layers and replaced the classifier head with a new one designed for our own dataset (10 Caltech-101 classes).  
Now, we will train only that head while keeping the rest of the network frozen.

We call this process **feature extraction** because we are not changing the features themselves — the pretrained backbone acts as a **fixed feature extractor**.  
Only the final classifier learns how to map those existing features to our new label space.

Mathematically, let the pretrained network be represented as:

$$
f(x) = h_\theta(x)
$$

where $h_\theta$ denotes the feature extractor (all convolutional layers) with parameters $\theta$.  

  
We then add a new classifier $g_\phi$ with parameters $\phi$ (our trainable layer).  
Our task is to learn $\phi$ while keeping $\theta$ constant:

$$
y = g_\phi(h_\theta(x))
$$

By setting `requires_grad=False` for all parameters in $\theta$, we ensure that gradients are not computed or propagated backward through the frozen layers.  
This saves computation and prevents altering the already learned representations.

###  Loss Function, Optimizer, and Training Loop Design

We now set up the **objective** and the **training loop** to learn only the **new classifier head** while keeping the pretrained backbone frozen.

**a) Criterion: Cross-Entropy Loss**
For a multi-class problem with one correct class per image, we use **cross-entropy loss**.  
If the model outputs logits $\mathbf{z} \in \mathbb{R}^K$ and the true class index is $y \in \{0,\dots,K-1\}$, the loss for one example is:
$$
\mathcal{L}(\mathbf{z}, y) = - \log \left( \frac{e^{z_y}}{\sum_{j=0}^{K-1} e^{z_j}} \right)
$$
Intuitively, this penalizes the model when the **softmax probability** assigned to the true class is low.

**b) Optimizer: Adam (only classifier parameters)**
We optimize **only** the classifier head parameters, e.g., `model.fc.parameters()`.  
The pretrained convolutional layers remain frozen (`requires_grad=False`) and therefore **do not receive gradients** or updates.

**c) Scheduler (Optional)**
A learning-rate scheduler (e.g., `StepLR`) gently reduces the learning rate as training progresses.  
This is optional for our short CPU-friendly runs, but helpful for demonstrating stable training practice.

**d) Training–Validation Loop**
We train for a **small number of epochs** (e.g., 5-10) to stay within a few minutes on CPU.  
Each epoch:
- **Train phase**: compute `train_loss` by averaging batch losses; update only the classifier head.
- **Validation phase**: compute `val_loss` and `val_acc` with `torch.no_grad()` (no gradient tracking); do **not** update weights.

We log `train_loss`, `val_loss`, and `val_acc` **per epoch** so we can visualize learning curves in the next section.  
Because we froze the backbone, we expect **rapid improvement** even with limited data.

**What happens to gradients here?**  

Only the **classifier head** parameters have `requires_grad=True`, so during backpropagation, **only** those weights receive gradient updates.  

The frozen backbone acts as a **fixed feature extractor**: its filters transform images into embeddings, but those filters are not altered.  

**Why is this efficient?**  
- Far fewer parameters are trained, so we see **fast convergence** and **short training time**.  
- With limited data, keeping the backbone fixed helps avoid **overfitting**.  

In the next section, we will visualize **loss and accuracy curves** to confirm that the model learns quickly with just the new head.  

Later, in NB03, we will experiment with **fine-tuning** by unfreezing selective deeper layers and using a smaller learning rate.

### Evaluating on the Test Set

After training the classifier head, we now evaluate its performance on the **held-out test set**.  
Our goal is to understand not only the **overall accuracy**, but also how performance varies **per class**, and which classes tend to be **confused** with one another.

We will compute:
- **Overall accuracy** — proportion of correctly classified test images.
- **Per-class precision, recall, and F1** — these give a richer view than accuracy alone:
  - Precision (for a class $c$): $ \text{Prec}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c} $
  - Recall (for a class $c$): $ \text{Rec}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c} $
  - F1 (for a class $c$): $ \text{F1}_c = \frac{2 \cdot \text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c} $
- A **confusion matrix heatmap** — this visualizes how predictions distribute across classes for each true label.

Interpreting these results will help us reason about:
- Which classes are **easier** (high recall and precision),
- Which classes are **harder** (lower scores),
- Where **systematic confusions** occur (e.g., visually similar categories).

We keep evaluation strictly with `torch.no_grad()` to ensure we do not alter model weights or track gradients during testing.

**How do we read these results?**

 - **Overall accuracy** provides a quick snapshot of performance, but it may hide class imbalances.  
 - **Per-class precision/recall/F1** reveal which categories are easier or harder.  
   - Low **recall** for a class means many of its images are missed (false negatives).  
   - Low **precision** for a class means many predictions for that class are incorrect (false positives).
 - The **confusion matrix** highlights systematic mistakes. If two classes are often confused with each other, it could be due to:
   - Visual similarity (e.g., similar textures or shapes),  
   - Background clutter or pose variation,  
   - Limited examples for the rarer class in our subset.

Because our backbone was **pretrained on ImageNet**, many categories that share common structures should already be well separated in feature space.  

### Visualizing Predictions and Model Confidence

To build intuition about what our classifier head has learned, we will **visualize predictions on a few test images**.  
For each image, we will display the **true label**, the **predicted label**, and the model’s **confidence** (the highest softmax probability).

Why is this useful?

- Seeing predictions next to images helps us connect **numbers to visual patterns**.  
- The **confidence score** offers a quick sense of how certain the model is about its decision.  
- When the model is wrong with **high confidence**, that often hints at **systematic confusion** between visually similar classes or strong **background biases**.  
- When the model is right with high confidence, it suggests that **pretrained features** already separate categories effectively in feature space, even though we trained only the final layer.

**What patterns do we observe in the grids**

 - **Correct, high-confidence predictions** suggest that the pretrained backbone already places those images in well-separated regions of feature space.  
 - **Low-confidence predictions** often occur when images are ambiguous, small, or have cluttered backgrounds.  
 - **Wrong but high-confidence predictions** usually indicate **systematic confusion** between visually similar classes or strong background cues that mislead the model.

These observations reinforce our key idea: **pretrained features** already provide a powerful representation.  

Training only the head leverages this representation effectively, but when confusions persist, we can try:
 - **Mild data augmentation** to improve robustness to viewpoint and background,  
 - **Class rebalancing** if the dataset is skewed,  
 - **Fine-tuning** deeper layers with a smaller learning rate in NB03 to adapt the representation to our specific categories.