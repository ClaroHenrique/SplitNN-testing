import torch
from torch import nn
from torch.nn import functional as F

# Define the client model:
class ClientModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 1, (2, 2))
    self.conv1.weight.data.fill_(1)
    self.conv1.bias.data.fill_(1)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    return x

class ServerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(961, 10)
    self.fc1.weight.data.fill_(1)
    self.fc1.bias.data.fill_(1)

  def forward(self, x):
    x = self.fc1(x)
    return x

client_model = ClientModel()
server_model = ServerModel()
