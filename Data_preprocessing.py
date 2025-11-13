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
                                transforms.Normalize((0.5,), (0.5,))])

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
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, padding = 1, stride = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 4, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, padding = 1, stride = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 4, stride = 1)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features = 2985984, out_features = 128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 128, out_features = num_classes),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = t.flatten(x,1)
        x = self.fc_layer(x)
        return x

LEARNING_RATE = 0.01
model = Model(in_channels = 1, num_classes=4)
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()             #using cross entropy loss
optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    running_loss = 0
    print(f'Epoch {epoch+1}/{EPOCHS}')
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

print(model.fc_layer.weight)
print(model.fc_layer.bias)
