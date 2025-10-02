import torch
from torch import nn
from torch.nn import functional as F
# For testing

class MyCNN_Client(nn.Module):
  def __init__(self):
    super().__init__()
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    return x

class MyCNN_Server(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(44944, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def ClientModel(is_client, split_point=None):
    return MyCNN_Client()

def ServerModel(is_client, split_point=None):
    return MyCNN_Server()

def MyCNN(num_classes=10, split_point=None, is_client=None):
    if is_client:
        return MyCNN_Client()
    else:
        return MyCNN_Server()
