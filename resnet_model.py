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

#TODO: refactor and clean up code, add comments where necessary
#TODO: see where and if relu should be replaced with gelu

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from training_utils import train_model, plot_training_history, classification_summary, permutation_test

#TODO: find out how to setup up mlx or the apple one
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

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
            nn.GELU(),
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
            nn.ReLU(inplace=True),      # I'll try GELU in this case
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Output after stem: ~ (B, 64, 56, 56) for 224×224 input
        self.dropout = nn.Dropout(p=0.5)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # TODO: is it worth adding more layers or changing num_blocks? # matter of fact letme change the layer a lil bit
        # then i'll add 1 more layer. And see how it goes
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.layer4 = self._make_layer(256, 256, num_blocks=2, stride=2)


        # After layer2 (with 224×224 input): (B, 128, 28, 28) --> (B, 64, 112, 112)

        # ---- Global average pool + FC ----
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 128, 1, 1)
        self.fc = nn.Linear(256, num_classes)            # -> (B, num_classes)

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


if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # HYPER PARAMETERS for SmallResNet
    EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.00012
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
    model = SmallResNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    
    # Train the model
    history = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, train_set, "small_resnet_model")
    
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
