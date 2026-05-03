# Multiclass Classification 

**multiclass classification**, where the output can belong to **one of many possible categories**.

>**What’s the Difference?**

| **Task Type**               | **Output Classes**                  | **Example**                             |
|----------------------------|-------------------------------------|------------------------------------------|
| Binary Classification      | 2 (e.g., {0, 1})                    | Is this email spam or not?               |
| Multiclass Classification  | 3 or more (e.g., {A, B, C, ...})    | What kind of protein localization site?  |

In binary classification, we used a **sigmoid activation** to squash outputs into a [0, 1] range.

But in multiclass settings, we use:

- **Softmax activation** in the output layer  
- **Cross-entropy loss** adapted for multiple classes  

Let’s visualize the difference:

> **Binary Classification**    : Input → NN → Single Output Node → Sigmoid → Probability (0 to 1)
>
> **Multiclass Classification**: Input → NN → One Output Node per Class → Softmax → Class Probabilities

**Softmax** ensures that all output values are between 0 and 1, and they sum up to 1 — forming a valid probability distribution over the classes.

Before we train any model, it’s critical to understand the **distribution of classes** in the target variable.

This helps us:
- Detect any **class imbalance** 🧯
- Decide whether we need to apply **stratification**, **resampling**, or **evaluation metrics** that handle imbalance

Before we build any predictive model, we need to **understand how each feature behaves** — both overall and across different target classes.

>**Why Analyze Feature Distributions Before Modeling?**

Before we train a neural network to predict subcellular localization, we need to understand what kind of signals we’re giving it to learn from. Feature distribution analysis helps us answer key questions about our data and its suitability for classification:

🔍 **Class Separation Insight**: By examining how feature values differ across classes, we can spot whether a feature might be informative. For example, if a feature has clearly different ranges for different localization sites, it likely holds predictive power.

🚨 **Outliers and Skewness**: Unusual values or strongly skewed distributions might cause the model to focus on unimportant patterns or slow down training. Early detection lets us consider normalization or other preprocessing steps.

🧠 **Learning-friendly Inputs**: Neural networks learn better when inputs have well-behaved distributions — not too flat, not too peaky, and with distinct patterns across classes. Violin plots help us quickly assess this.

🛠️ **Feature Engineering Decisions**: If we see that a feature barely changes across classes, it may not contribute much to learning. Conversely, strong between-class separation can inspire us to emphasize that feature in future modeling.

A **correlation matrix** helps us visualize the strength and direction of linear relationships between pairs of continuous features.

>**Understanding Feature Correlations Before Modeling**

Understanding feature correlations helps us make sense of the information structure in our data — which is especially useful when we are about to feed this information into a neural network.

Here’s why it’s worth paying attention to:

🔁 **Redundancy and Multicollinearity**: If two features are highly correlated, they might be giving us the same signal. Including both doesn’t necessarily hurt a neural network, but it can lead to inefficient learning or slower convergence, especially in smaller networks.

🧠 **Neural Network Efficiency**: Deep learning models, especially simple ones like the MLPs we're starting with, work better when features are diverse and non-repetitive. Low correlation means each feature is adding new information, which can help the model learn richer patterns.

🧱 **Better Model Design**: If we spot strong correlations, we might consider:
- Combining features through dimensionality reduction (e.g., PCA in future projects).
- Removing redundant features to simplify the model.
- Normalizing inputs more carefully (important for gradient-based learning).

🧪 **Domain Insight**: In our case, understanding which biological signals (like mitochondrial vs. nuclear scores) are related can offer insight into how different cellular destinations overlap or differ — both biologically and statistically.

This is not just a data check — it’s a window into the inner structure of the learning problem.


>**Pearson Correlation Coefficient**

The most common measure of correlation is the **Pearson correlation coefficient**:

$$
r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$

- $r = +1$: perfect positive correlation  
- $r = -1$: perfect negative correlation  
- $r = 0$: no linear correlation

We'll apply two encodings for two purposes:

>**Label Encoding (For Model Training)**

- Converts each class to a unique integer (e.g., `"MIT"` → 0, `"NUC"` → 1, …).
- Used for training classifiers like neural networks in PyTorch.

>**One-Hot Encoding (For Understanding & Multiclass Output)**

- Converts each label into a vector with a `1` at the class index, `0` elsewhere.
- Used to understand the structure of multiclass targets and to compute class-wise losses.

>**Summary of Both Encodings**

| Class Label | Label Encoded | One-Hot Encoded       |
|-------------|----------------|------------------------|
| MIT         | 0              | [1, 0, 0, ..., 0]       |
| NUC         | 1              | [0, 1, 0, ..., 0]       |
| CYT         | 2              | [0, 0, 1, ..., 0]       |
| ...         | ...            | ...                    |


>**What is Softmax?**

Given logits $z_1$, $z_2$, ..., $z_K$ for $K$ classes,  
softmax produces a probability distribution:

$$
P(y = k \mid \mathbf{z}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

>**Quick Recap: What Are Logits?**

Logits are the **raw outputs** of the last layer in a neural network — they can be **positive or negative**, and don't sum to 1.

- Exponentiation ensures **positivity**
- Division ensures probabilities **sum to 1**
- Higher logits → higher probability

>**Why Use Softmax?**

- We get **mutually exclusive probabilities** (i.e., pick one class)
- It's differentiable → compatible with gradient-based learning
- Helps us interpret model outputs as probabilities

>**Numerical Example: From Logits to Probabilities**

Let’s say our model outputs the following logits for a 3-class problem:  
`logits = [2.0, 1.0, 0.1]`

Let’s apply the **softmax** formula step by step:

> Step 1: Exponentiate the logits  
$$
e^{2.0} = 7.389,\quad e^{1.0} = 2.718,\quad e^{0.1} = 1.105
$$

>➕ Step 2: Sum the exponentiated values  
$$
\text{sum} = 7.389 + 2.718 + 1.105 = 11.212
$$

>🔗 Step 3: Divide each by the sum  
$$
P(\text{class 1}) = \frac{7.389}{11.212} \approx 0.659 \\
P(\text{class 2}) = \frac{2.718}{11.212} \approx 0.242 \\
P(\text{class 3}) = \frac{1.105}{11.212} \approx 0.099
$$

So, the model predicts **class 1** with the highest confidence, assigning it ~66% probability.

This example shows how **softmax** turns arbitrary logits into a **probability distribution**, and how changes in logits directly affect model predictions.

Now that we’ve introduced **Softmax** for multiclass classification, we need a loss function that works well with it — that’s where **Cross-Entropy Loss** comes in.

>**❌ Why Not MSE?**

In regression tasks, we often use **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}^i - y^i)^2
$$

But for classification, **MSE doesn't perform well**:
- It doesn’t penalize wrong class probabilities effectively.
- It interacts poorly with softmax gradients during backpropagation.
- It treats class outputs as if they were continuous values, which they’re not.

✅ For classification, we need a loss function that compares **probability distributions** — and that’s what **cross-entropy** does.


>Intuition: Cross-Entropy Measures Surprise**

Cross-entropy quantifies how "surprised" our model is by the actual label.

- The **more confident and correct** the model is, the **lower** the loss.
- The **more confident and wrong** it is, the **higher** the loss.

>**General Formula for Cross-Entropy Loss**

When using **one-hot encoded labels**, the general formula is:

$$
\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)
$$

Where:
- $K$ = number of classes  
- $y_k \in \{0, 1\}$: 1 if class $k$ is the true class, 0 otherwise  
- $\hat{p}_k$: predicted probability for class $k$

🟡 Only one of the $y_k$ values is 1 (for the true class), so this simplifies to:

$$
\mathcal{L}_{\text{CE}} = -\log(\hat{p}_{\text{true class}})
$$

>**Numerical Example**

Let’s say the true class is class 2, and the model outputs the following softmax probabilities:

$$
[\hat{p}_1, \hat{p}_2, \hat{p}_3] = [0.1, 0.8, 0.1]
$$

Then:

$$
\mathcal{L} = -\log(0.8) \approx 0.22
$$

Now imagine the model is confident but wrong:

$$
[\hat{p}_1, \hat{p}_2, \hat{p}_3] = [0.7, 0.2, 0.1]
$$

Then:

$$
\mathcal{L} = -\log(0.2) \approx 1.61
$$

✅ This shows how cross-entropy **heavily penalizes confident mistakes**, which helps the model learn to be both accurate and calibrated.

### **Evaluation Metrics for Multiclass Classification**

After training a classification model, we need to **evaluate how well it performs**.  
In multiclass classification, we go beyond just **accuracy** to get a deeper understanding of model behavior.

Let’s explore the most important metrics:

✅ **1. Accuracy**  : $ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$

- Measures the overall proportion of correct predictions.
- ❗ Doesn’t tell us **which classes are being confused**.


📐 **2. Recall (a.k.a. Sensitivity / True Positive Rate)**: $\text{Recall}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Negatives}_c}$

- Out of all **actual class** \( c \) samples, how many were correctly predicted?
- Measures the model’s ability to **catch** true instances of a class.


🎯 **3. Precision (a.k.a. Positive Predictive Value)**: $\text{Precision}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Positives}_c}$

- Out of all samples **predicted as class** \( c \), how many were actually correct?
- Tells us how much we can **trust the model’s predictions** for a class.


🧪 **4. Specificity (a.k.a. True Negative Rate)**: $\text{Specificity}_c = \frac{\text{True Negatives}_c}{\text{True Negatives}_c + \text{False Positives}_c}$

- Out of all samples **not belonging to class** \( c \), how many were correctly rejected?
- Important when false alarms are costly.

⚖️ **5. F1 Score**: $\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$

- Balances both precision and recall.
- Useful when we care about **both completeness and correctness**.


📊 **6. Confusion Matrix**

The **confusion matrix** shows how predictions are distributed:

- **Rows**: Actual labels  
- **Columns**: Predicted labels  
- Diagonal = correct predictions  
- Off-diagonal = misclassifications


🧾 **7. Averaging Methods in Multiclass Metrics**

In multiclass settings, we summarize metrics across all classes using:

- **Macro average**: Treats all classes equally.  
- **Weighted average**: Weights each class by how frequent it is.  
- **Micro average**: Aggregates total TP, FP, FN globally (treats as binary).


> **📚 Binary Classification Example**

|                          | Predicted: Positive | Predicted: Negative |
|--------------------------|---------------------|---------------------|
| **Actual: Positive (1)** | 40 (TP)             | 10 (FN)             |
| **Actual: Negative (0)** | 30 (FP)             | 120 (TN)            |

**Now compute:**

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN} = \frac{40 + 120}{200} = 0.80$

- **Precision**: $\frac{TP}{TP + FP} = \frac{40}{40 + 30} \approx 0.571$

- **Recall (Sensitivity)**: $\frac{TP}{TP + FN} = \frac{40}{40 + 10} = 0.80$

- **Specificity**: $\frac{TN}{TN + FP} = \frac{120}{120 + 30} = 0.80$

- **F1 Score**: $2 \cdot \frac{0.571 \cdot 0.80}{0.571 + 0.80} \approx 0.666$

> 🔍 **Insight**: Accuracy looks good, but low precision indicates many **false positives**. F1 gives a more realistic picture of model performance.


> **📊 Multiclass Example**

Confusion matrix:

|              | Pred: A | Pred: B | Pred: C |
|--------------|---------|---------|---------|
| **Actual A** | 30      | 5       | 5       |
| **Actual B** | 10      | 25      | 5       |
| **Actual C** | 0       | 10      | 20      |

**Per-class Precision**:

- A: $\frac{30}{30 + 10} = 0.75$

- B: $\frac{25}{25 + 5 + 10} = 0.625$

- C:  $\frac{20}{20 + 10} = 0.666$

**Per-class Recall**:

- A: $\frac{30}{30 + 5 + 5} = 0.75$

- B: $\frac{25}{25 + 10 + 5} = 0.625$

- C: $\frac{20}{20 + 10} = 0.666$

**Macro Precision**: $\frac{0.75 + 0.625 + 0.666}{3} \approx 0.68$

**Macro Recall**: $\frac{0.75 + 0.625 + 0.666}{3} \approx 0.68$

> 🎯 **Insight**: Macro metrics treat all classes equally — useful when class balance is important. Weighted metrics may better reflect true performance when classes are imbalanced.

> **🧠 Terminology Bridge for Bio/ML Students**

| Clinical/Bio Name       | ML Equivalent              |
|-------------------------|----------------------------|
| **Sensitivity**         | **Recall**                 |
| **Specificity**         | **True Negative Rate**     |
| **Precision**           | **Positive Predictive Value (PPV)** |