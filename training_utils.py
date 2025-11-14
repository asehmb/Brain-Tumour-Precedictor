# =============================================================================
# This file is part of Brain-Tumour Predictor.
#
# Copyright (C) 2025 Ajaydeep Sehmbi and Kiet Huynh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def check_accuracy(loader, model):
    """Calculates model accuracy on a given DataLoader."""
    num_correct = 0
    num_samples = 0
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)

    with torch.no_grad():  # Do not calculate gradients during evaluation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            # Get the index of the max value (the predicted class)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = float(num_correct) / float(num_samples) * 100
    model.train()  # Set model back to training mode
    return accuracy

def permutation_test(model, X, y, permutations=1000):
    """
    Permutation Ttest Implementation
    model = trained model
    X = input data (images)
    y = true labels (labels)
    permutations = number of permutations to perform
    """
    model.eval()
    with torch.no_grad():
        real_accuracy = (model(X).argmax(dim=1) == y).float().mean().item()
    
    permutation_accuracies = []
    for _ in range(permutations):
        permuted_y = y[torch.randperm(y.size(0))]
        with torch.no_grad():
            permuted_accuracy = (model(X).argmax(dim=1) == permuted_y).float().mean().item()
        permutation_accuracies.append(permuted_accuracy)

    pvalue = np.mean([s >= real_accuracy for s in permutation_accuracies])
    return real_accuracy, permutation_accuracies, pvalue

def classification_summary(model, X, y_true, class_names=None):
    """
    Prints a classification report and plots a heatmap confusion matrix.

    Parameters:
        y_true: array-like, true labels
        y_pred: array-like, predicted labels
        class_names: list of class name strings (optional)
    """
    model.eval()
    with torch.no_grad():
        scores = model(X)
        _, y_pred = scores.max(1)
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()

    # Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # If no class names provided, auto-generate them
    if class_names is None:
        num_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

def train_model(model, train_loader, test_loader, epochs, learning_rate, train_set, model_name="model"):
    """
    Training function for any model
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store stats for plotting later
    history = {'train_loss': [], 'test_accuracy': []}

    for epoch in range(epochs):
        running_loss = 0.0

        print(f'\n--- Epoch {epoch + 1}/{epochs} ---')

        # Training loop
        model.train()  # Ensure model is in training mode
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # Accumulate weighted loss

        # Calculate and record statistics
        epoch_loss = running_loss / len(train_set)
        history['train_loss'].append(epoch_loss)

        # Check test accuracy
        test_acc = check_accuracy(test_loader, model)
        history['test_accuracy'].append(test_acc)

        # Output stats
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

    print('\nFinished Training')

    # Final evaluation (optional, as it was done on the last epoch)
    final_test_acc = check_accuracy(test_loader, model)
    print(f'Final Test Accuracy: {final_test_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), f'{model_name}.pth')

    return history

def plot_training_history(history):
    """Plot training loss and test accuracy"""
    # plot training loss and test accuracy to see trends and check for overfitting
    fig, ax1 = plt.subplots()

    # Plot training loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(history['train_loss'], color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Test Accuracy (%)', color=color)
    ax2.plot(history['test_accuracy'], color=color, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss & Test Accuracy over Epochs")
    fig.tight_layout()
    plt.show()

def show_img(img):
    img = img * 0.5 + 0.5 # remove normalizations for visualization
    npimg = img.numpy() #convert tensor to numpy so it can be transposed
    plt.imshow(npimg.T)
    plt.show()
