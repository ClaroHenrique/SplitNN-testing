import torch
from torch import nn
from torch.nn import functional as F

# Define the client model
class ClientModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.pool = nn.MaxPool2d(2, 2)
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.conv2 = nn.Conv2d(6, 16, 5)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = torch.flatten(x, 1)
      return x


class ServerModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x