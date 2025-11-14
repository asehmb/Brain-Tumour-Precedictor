import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import torchvision
from matplotlib.pyplot import figure
from sympy.physics.units import momentum
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.functional import F
from tqdm.auto import tqdm

device = "cuda" if t.cuda.is_available() else "cpu"

#label the image data,
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.25,0.25,0.25))])

#create data folder
train_dataset = datasets.ImageFolder('Training', transform=transform)
test_dataset = datasets.ImageFolder('Testing', transform=transform)

#load data
train_loader = DataLoader(dataset=train_dataset, batch_size= 256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 256, shuffle=True)

#visualize the image
def show_img(img):
    img = img * 0.5 + 0.5
    npimp = img.numpy()
    plt.imshow(npimp.T)
    plt.show()

print(len(train_dataset))
print(len(test_dataset))
train_iter = iter(train_loader)
images, labels = next(train_iter)
print(labels)
show_img(make_grid(images))

#shape
''' 
    Shape: (256, 4, 224, 224)
'''

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        # input layer
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(p = 0.5)

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)    # 32, 224, 224
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)   # 64, 112, 112
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)  # 128, 56, 56
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1) # 256, 28, 28
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1) # 256, 14, 14

        # output layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 256 * 7 * 7, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))     # B, 32, 112, 112
        x = self.pool(self.relu(self.conv2(x)))     # B, 64, 56, 56
        x = self.pool(self.relu(self.conv3(x)))     # B, 128, 28, 28
        x = self.pool(self.relu(self.conv4(x)))     # B, 256, 14, 14
        x = self.pool(self.relu(self.conv5(x)))     # B, 256, 7, 7
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def check_accuracy(loader, model):
    """Calculates model accuracy on a given DataLoader."""
    num_correct = 0
    num_samples = 0
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)

    with t.no_grad():  # Do not calculate gradients during evaluation
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



LEARNING_RATE = 0.001
model = Model( num_classes = 4).to(device)
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()             #using cross entropy loss
optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize lists to store stats for plotting later
history = {'train_loss': [], 'test_accuracy': []}

for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f'\n--- Epoch {epoch+1}/{EPOCHS} ---')
    
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
    epoch_loss = running_loss / len(train_dataset)
    history['train_loss'].append(epoch_loss)

    # Check test accuracy
    test_acc = check_accuracy(test_loader, model)
    history['test_accuracy'].append(test_acc)

    # Output stats
    print(f"Training Loss: {epoch_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

print('\nFinished Training')

# Final evaluation
final_test_acc = check_accuracy(test_loader, model)
print(f'Final Test Accuracy: {final_test_acc:.2f}%')

print(model.fc2.weight)
print(model.fc2.bias)
