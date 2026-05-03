import pandas as pd
# Load the dataset
CT_df = pd.read_csv("yeast.csv")

# Task 1 – Dataset Dimensions and Columns
CT_df_shape = CT_df.shape  # Tuple of (rows, cols)
CT_feature_cols = CT_df.columns[:-1].tolist() # List of column names (excluding target)

print(f"CT_Dataset shape: {CT_df_shape}")
print(f"CT_Feature columns: {CT_feature_cols}")

#Exercise 1: Count number of samples in each localization class

#Class Distribution Dictionary
CT_class_counts = CT_df.iloc[:, -1].value_counts().to_dict() # dict with class label -> count
print("CT_Class distribution:", CT_class_counts)

from sklearn.preprocessing import LabelEncoder

#--------------------------------------------------------------------------------------------------
#Exercise 2:Compute the correlation between each feature and the label-encoded localization target. 

#Correlation of features with target
CT_label_encoder = LabelEncoder()
CT_target_numeric = CT_label_encoder.fit_transform(CT_df['localization_site'])
CT_df_corr = df[CT_feature_cols].copy()
CT_df_corr["label_numeric"] = CT_target_numeric

CT_feature_target_corr = CT_df_corr.corr()["label_numeric"].drop("label_numeric")
print("CT_Feature-target correlations:")
print(CT_feature_target_corr)

#--------------------------------------------------------------------------------------------------
#Exercise 3: Label and One-hot encode the target variable
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Copy the original dataframe to avoid modifying it in-place
df_encoded = df.copy()

# Label Encoding
label_encoder = LabelEncoder()
df_encoded['label_encoded'] = label_encoder.fit_transform(df_encoded['localization_site'])

# One-Hot Encoding (for illustrative purposes)
one_hot = pd.get_dummies(df_encoded['localization_site'], prefix="class")

# Display first few rows
print("✅ Label Encoded Classes:")
print(df_encoded[['localization_site', 'label_encoded']].head())

print("\n✅ One-Hot Encoded Output (first 5 rows):")
print(one_hot.head())
#label_encoded: assigns a unique integer to each class — needed for model training.
#class_* columns: form the one-hot encoded representation — useful to visualize class structure.

#--------------------------------------------------------------------------------------------------
#Exercise 4: Implement softmax against logits
import numpy as np
import matplotlib.pyplot as plt

def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Vary logit of Class 1, fix Class 2 = 0, Class 3 = -1
z_vals = np.linspace(-2, 4, 100)
logits_list = [[z, 0, -1] for z in z_vals]
logits_array = np.array(logits_list)

probs = softmax(logits_array)

plt.figure(figsize=(7, 5))
plt.plot(z_vals, probs[:, 0], label="Class 1")
plt.plot(z_vals, probs[:, 1], label="Class 2")
plt.plot(z_vals, probs[:, 2], label="Class 3")
plt.xlabel("Logit of Class 1")
plt.ylabel("Probability")
plt.title("Softmax Probabilities as Class 1 Logit Varies")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------------
#Exercise 5: MSE vs Cross-Entropy Loss for Multiclass Classification

# Range of predicted probabilities for the correct class
p = np.linspace(0.001, 0.999, 100)

# Loss calculations
cross_entropy = -np.log(p)
mse = (1 - p)**2

# Plot
plt.figure(figsize=(8, 6))
plt.plot(p, cross_entropy, label='Cross-Entropy Loss', linewidth=2)
plt.plot(p, mse, label='MSE Loss', linestyle='--')
plt.xlabel("Predicted Probability for True Class")
plt.ylabel("Loss")
plt.title("Loss vs. Confidence for the Correct Class")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------------
#Exercise 6: Evaluate model using accuracy, macro-averaged precision, recall, F1 score, and confusion matrix.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import numpy as np
import matplotlib.pyplot as plt

# True and predicted labels from a multiclass model
y_true = np.array(['MIT', 'NUC', 'CYT', 'ME3', 'MIT', 'NUC', 'CYT', 'POX', 'MIT', 'NUC'])
y_pred = np.array(['MIT', 'NUC', 'NUC', 'ME3', 'CYT', 'NUC', 'CYT', 'ME3', 'MIT', 'CYT'])

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

# Macro-averaged precision, recall, F1 (with zero_division handled)
macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Macro F1 Score:", macro_f1)

# Full classification report (per-class metrics)
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# Confusion Matrix
labels = np.unique(y_true)
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Multiclass)")
plt.show()

# True and predicted labels from a multiclass model
y_true = np.array(['MIT', 'NUC', 'CYT', 'ME3', 'MIT', 'NUC', 'CYT', 'POX', 'MIT', 'NUC'])
y_pred = np.array(['MIT', 'NUC', 'NUC', 'ME3', 'CYT', 'NUC', 'CYT', 'ME3', 'MIT', 'CYT'])

#--------------------------------------------------------------------------------------------------
#Exercise 7: Manual compute precision and recall for a specific class (e.g., "MIT") using the true and predicted labels.

CT_class = "MIT"
# True Positives: predicted MIT AND actually MIT
CT_TP = sum((yt == CT_class) and (yp == CT_class) 
            for yt, yp in zip(y_true, y_pred))

# False Positives: predicted MIT but actually NOT MIT
CT_FP = sum((yt != CT_class) and (yp == CT_class) 
            for yt, yp in zip(y_true, y_pred))

# False Negatives: actually MIT but predicted NOT MIT
CT_FN = sum((yt == CT_class) and (yp != CT_class) 
            for yt, yp in zip(y_true, y_pred))

CT_precision = CT_TP / (CT_TP + CT_FP) if (CT_TP + CT_FP) > 0 else 0
CT_recall = CT_TP / (CT_TP + CT_FN) if (CT_TP + CT_FN) > 0 else 0

print(f"CT_Precision for {CT_class}: {CT_precision:.2f}")
print(f"CT_Recall for {CT_class}: {CT_recall:.2f}")

#--------------------------------------------------------------------------------------------------
#Exercise 8: Implement softmax function and apply it to example logits to get predicted probabilities and classes.

# Define softmax
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Two example logits (as if from a neural net with 4 output classes)
logits_example_1 = np.array([2.0, 1.0, 0.1, -1.0])
logits_example_2 = np.array([0.5, 0.6, 0.4, 0.55])

# Apply softmax
probs_1 = softmax(logits_example_1)
probs_2 = softmax(logits_example_2)

# Predicted classes
pred_1 = np.argmax(probs_1)
pred_2 = np.argmax(probs_2)

print("Example 1 Logits:", logits_example_1)
print("Softmax Probabilities:", probs_1)
print("Predicted Class:", pred_1)

print("\nExample 2 Logits:", logits_example_2)
print("Softmax Probabilities:", probs_2)
print("Predicted Class:", pred_2)