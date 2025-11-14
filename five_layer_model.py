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
from training_utils import train_model, plot_training_history, classification_summary, permutation_test

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# 5 layers Model
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


if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # HYPER PARAMETERS for 5-layer Model
    EPOCHS = 15
    BATCH_SIZE = 512
    LEARNING_RATE = 8e-4
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

    # Ensure model and data are on the same device
    model = Model(num_classes=NUM_CLASSES).to(device)
    
    # Train the model
    history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, train_set, "five_layer_model")
    
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
