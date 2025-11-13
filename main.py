import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

#TODO: find out how to setup up mlx or the apple one
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER PARAMETERS
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.01
NUM_CLASSES = 4

data_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(num_output_channels=1),
     transforms.Normalize((0.5,), (0.5,)),
     transforms.Resize((224, 224)),
     ])

train_set = ImageFolder(root='./Training', transform=data_transforms)
test_set = ImageFolder(root='./Testing', transform=data_transforms)

# shape = (BATCH_SIZE, 3, 224, 224)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
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
        x = F.relu(self.fc1(x))         # apply full layer
        return x

def show_img(img):
    img = img * 0.5 + 0.5 # remove normalizations for visualization
    npimg = img.numpy() #convert tensor to numpy so it can be transposed
    plt.imshow(npimg.T)
    plt.show()


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


if __name__ == "__main__":
    # Ensure model and data are on the same device
    model = CNN(in_channels=1, num_classes=NUM_CLASSES).to(device)

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

    # Example of how to use the tracked history for plotting
    # plt.figure(figsize=(10, 4))
    # plt.plot(history['train_loss'], label='Training Loss')
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()