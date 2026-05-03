# Evaluation Metrics

- **Overall accuracy** is a good starting point, but for large data sets it hides important details:
  - Some data might be predicted correctly most of the time.
  - Others might rarely be recognized, even if overall accuracy seems acceptable.

**Per-Class Accuracy**

- Helps identify **imbalances** or **systematic errors**:
  - Which are **well-classified**?
  - Which are **frequently confused**?

**Precision, Recall, and F1-Score**

- **Precision:** Out of predicted positives, how many were correct?  
- **Recall:** Out of actual positives, how many were recovered?  
- **F1-Score:** Harmonic mean of precision and recall — balances the two.

For **multi-class problems**:
- We use **macro-averaging** (treats all classes equally) and **micro-averaging** (weighted by frequency).

**What We’ll Do in Code**

1. Compute **overall accuracy** (baseline reference).  
2. Calculate **per-class accuracy** and display **top 5 best/worst breeds**.  
3. Generate a **classification report** (precision, recall, F1-score) using `sklearn.metrics.classification_report`.

**Why Visualize the Confusion Matrix?**

- **Accuracy** alone doesn’t show *which* classes are misclassified.  
- A confusion matrix highlights:
  - **Clusters of errors** (e.g., similar cat breeds misclassified as each other).
  - **Biases** (e.g., misclassifying small dogs as a single dominant breed).
  - Opportunities for **targeted improvements** (data augmentation, class weighting).

Visualize it with a **heatmap**:
   - Rows = true classes
   - Columns = predicted classes
   - Color intensity = frequency of predictions
