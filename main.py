import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import permutation_test_score

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#TODO: find out how to setup up mlx or the apple one
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(device)

# HYPER PARAMETERS
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 4

data_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5,), (0.25,0.25,0.25,)),
     transforms.Resize((224, 224)),
     ])

train_set = ImageFolder(root='./Training', transform=data_transforms)
test_set = ImageFolder(root='./Testing', transform=data_transforms)

# shape = (BATCH_SIZE, 3, 224, 224)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)




class ResidualBlock(nn.Module):
    """
    Basic ResNet-style block:
    input:  (B, in_channels, H, W)
    output: (B, out_channels, H/stride, W/stride)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main conv path (F(x))
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Skip path: identity or 1×1 conv if shape changes
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        # Input: (B, 3, 224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Output after stem: ~ (B, 64, 56, 56) for 224×224 input
        self.dropout = nn.Dropout(p=0.5)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # TODO: is it worth adding more layers or changing num_blocks?
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # After layer2 (with 224×224 input): (B, 128, 28, 28)

        # ---- Global average pool + FC ----
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 128, 1, 1)
        self.fc = nn.Linear(512, num_classes)            # -> (B, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Build a stage: first block can downsample (stride>1), others keep size.
        Returns an nn.Sequential of ResidualBlocks.
        """
        layers = []
        # First block: may change channels and/or stride
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))

        # Remaining blocks: same channels, stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: test dropout placement, maybe add more pooling?
        # x: (B, in_channels, H, W) e.g. (B, 3, 224, 224)
        x = self.stem(x)      # -> (B, 64, 56, 56)
        x = self.layer1(x)    # -> (B, 64, 56, 56)
        x = self.layer2(x)    # -> (B, 128, 28, 28)
        x = self.layer3(x)    # -> (B, 256, 14, 14)
        x = self.layer4(x)    # -> (B, 512, 7, 7)

        x = self.global_pool(x)   # -> (B, 128, 1, 1)
        x = torch.flatten(x, 1)   # -> (B, 128)
        x = self.dropout(x)
        x = self.fc(x)            # -> (B, num_classes) logits
        return x

# not optimized, preliminary testing
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        """
        :param in_channels: number of input channels --- 1 maybe for grayscale 3 for rgb
        :param num_classes: number of output classes --- 4 for this project
        """
        super().__init__()
        # out_channels = amount of features to extract, kernel_size = output res (4x4) stride = step of kernel
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, padding=1, stride=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        # in_features = out_chanels*(kernel_size*kernel_size)
        self.fc1 = nn.Linear(in_features=193600, out_features=num_classes)

    def forward(self, x):
        """
        :param x: input image
        """
        x = F.relu(self.conv1(x))       # activate conv1
        x = self.pool(x)                # max pooling
        x = F.relu(self.conv2(x))       # apply 2nd conv layer and ReLu activation
        x = self.pool(x)                # apply max pooling
        x = x.reshape(x.shape[0], -1)   #flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))         # apply full layer
        return x

def show_img(img):
    img = img * 0.5 + 0.5 # remove normalizations for visualization
    npimg = img.numpy() #convert tensor to numpy so it can be transposed
    plt.imshow(npimg.T)
    plt.show()


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


if __name__ == "__main__":
    # Ensure model and data are on the same device
    model = SmallResNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize lists to store stats for plotting later
    history = {'train_loss': [], 'test_accuracy': []}

    for epoch in range(EPOCHS):
        running_loss = 0.0

        print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')

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


    ### TESTING ####
    # pytorch imageloader labels are in alphabetical order
    classification_summary(model, images, labels, class_names=['glioma', 'meningioma', 'no_tumor', 'pituitary'])

    # Permutation test
    real_accuracy, permutation_scores, pvalue = permutation_test(
        model, images, labels, permutations=1000
    )

    print(f"real_accuracy score: {real_accuracy:.4f}")
    print(f"Permutation scores: {permutation_scores[:10]}...")  # print first 10 scores
    print(f"P-value: {pvalue:.4f}")

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
