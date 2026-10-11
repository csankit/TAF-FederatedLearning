"""Models used in the paper.

MNIST: two convolutional layers + two fully connected layers.
CIFAR-10: the CNN used for CIFAR in McMahan et al. (2017), derived from the
TensorFlow tutorial model (conv64-pool-conv64-pool-fc384-fc192-fc10).
"""
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.fc2(self.act(self.fc1(x)))


class CIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def build_model(dataset: str):
    if dataset in ("mnist", "fake_mnist"):
        return MNISTCNN()
    if dataset in ("cifar10", "fake_cifar10"):
        return CIFARCNN()
    raise ValueError(dataset)
