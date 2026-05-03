import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


class MLP_Dropout_L2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP_Dropout_L2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model_dropout_l2 = MLP_Dropout_L2(input_dim, hidden_dim=32, output_dim=num_classes, dropout_rate=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_dropout_l2.parameters(), lr=0.001, weight_decay=1e-4)

train_losses_dropout_L2 = []
val_losses_dropout_L2 = []

for epoch in range(1000):
    model_dropout_l2.train()
    y_pred = model_dropout_l2(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    train_losses_dropout_L2.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model_dropout_l2.eval()
    with torch.no_grad():
        val_output = model_dropout_l2(X_val_tensor)
        val_loss = criterion(val_output, y_val_tensor)
        val_losses_dropout_L2.append(val_loss.item())
        # Optional: Print every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# Plot training vs. validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses_dropout_L2, label="Training Loss (Dropout + L2)")
plt.plot(val_losses_dropout_L2, label="Validation Loss (Dropout + L2)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss (Dropout + L2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate all models on the test set
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1)

    y_true = y_test_tensor.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

# Evaluate No Regularization (M1)
acc_m1, f1_m1 = evaluate_model(model_base, X_test_tensor, y_test_tensor)

# Evaluate Dropout Only (M2)
acc_m2, f1_m2 = evaluate_model(model_dropout, X_test_tensor, y_test_tensor)

# Evaluate L2 Regularization Only (M3)
acc_m3, f1_m3 = evaluate_model(model_l2, X_test_tensor, y_test_tensor)

# Evaluate Dropout + L2 (M4)
acc_m4, f1_m4 = evaluate_model(model_dropout_l2, X_test_tensor, y_test_tensor)

results = {
    "Model": ["No Reg (M1)", "Dropout (M2)", "L2 (M3)", "Dropout + L2 (M4)"],
    "Test Accuracy": [acc_m1, acc_m2, acc_m3, acc_m4],
    "Macro F1 Score": [f1_m1, f1_m2, f1_m3, f1_m4]
}

results_df = pd.DataFrame(results)
print(results_df)

