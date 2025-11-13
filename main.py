import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

data_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     transforms.Resize((224, 224)),
     ])

trainset = ImageFolder(root='./Training', transform=data_transforms)
testset = ImageFolder(root='./Testing', transform=data_transforms)

print(len(trainset))
print(len(testset))