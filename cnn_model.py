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

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import train_model, plot_training_history, classification_summary, permutation_test

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

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


if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # HYPER PARAMETERS for CNN
    EPOCHS = 8
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-3
    NUM_CLASSES = 4

    # Note: Using grayscale for this CNN model
    data_transforms = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.25,)),
         ])

    train_set = ImageFolder(root='./Training', transform=data_transforms)
    test_set = ImageFolder(root='./Testing', transform=data_transforms)

    # shape = (BATCH_SIZE, 1, 224, 224) for grayscale
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Ensure model and data are on the same device
    model = CNN(in_channels=1, num_classes=NUM_CLASSES).to(device)
    
    # Train the model
    history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, train_set, "cnn_model")
    
    # Plot training history
    plot_training_history(history)
    
    # Testing
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # pytorch imageloader labels are in alphabetical order
    classification_summary(model, images, labels, class_names=['glioma', 'meningioma', 'no_tumor', 'pituitary'])

    # Permutation test
    real_accuracy, permutation_scores, pvalue = permutation_test(
        model, images, labels, permutations=1000
    )

    print(f"real_accuracy score: {real_accuracy:.4f}")
    print(f"Permutation scores: {permutation_scores[:10]}...")  # print first 10 scores
    print(f"P-value: {pvalue:.4f}")
