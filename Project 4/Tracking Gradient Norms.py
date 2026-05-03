def track_gradient_norms(model, train_loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    gradient_history = []

    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Record L2 norm of gradients for each layer
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.detach().norm(2).item()
                    grad_norms.append(grad_norm)
            gradient_history.append(grad_norms)

            optimizer.step()
            break  # Only one batch per epoch for gradient tracking (faster)

        print(f"Epoch {epoch+1}/{num_epochs} ✅ Gradient norms tracked.")

    return gradient_history

#Plot gradient norms
def plot_gradient_norms(gradient_history):
    import numpy as np
    import matplotlib.pyplot as plt

    grad_array = np.array(gradient_history)  # shape: (epochs, num_params)
    plt.figure(figsize=(10, 6))

    for i in range(grad_array.shape[1]):
        plt.plot(grad_array[:, i], label=f"Layer {i+1}")

    plt.title("Gradient Norms Across Layers (Over Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("Gradient L2 Norm")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def track_gradient_norms(model, train_loader, num_epochs=10, lr=0.001):
    # import necessary libraries
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    gradient_history = []

    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Record only weight gradients (not biases)
            grad_norms = []
            for name, param in model.named_parameters():
                if "weight" in name and param.grad is not None:
                    grad_norm = param.grad.detach().norm(2).item()
                    grad_norms.append(grad_norm)

            gradient_history.append(grad_norms)
            break  # One batch per epoch

        print(f"Epoch {epoch+1}/{num_epochs} ✅ Tracked {len(grad_norms)} weight gradients.")

    return gradient_history

# Use our original deep MLP (with default init)
gradient_history = track_gradient_norms(relu_deep_mlp, train_loader, num_epochs=15)
plot_gradient_norms(gradient_history)
