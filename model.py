import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(7*7*64, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))

        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        output = self.softmax(x)

        return output


def test_model():
    model = Model()
    # print(model)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(output.shape)  # torch.Size([1, 10])


if __name__ == '__main__':
    test_model()
