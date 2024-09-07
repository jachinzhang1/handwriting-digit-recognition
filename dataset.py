import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms

path = './data'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
batch_size = 2048

trainData = torchvision.datasets.MNIST(
    path,
    train=True,
    transform=transform,
    download=True,
)
testData = torchvision.datasets.MNIST(
    path,
    train=False,
    transform=transform,
)

trainDataLoader = torch.utils.data.DataLoader(
    dataset=trainData,
    batch_size=batch_size,
    shuffle=True,
)
testDataLoader = torch.utils.data.DataLoader(
    dataset=testData,
    batch_size=batch_size,
)