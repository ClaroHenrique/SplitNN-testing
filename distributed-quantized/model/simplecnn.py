import torch
from torch import nn
from torch.nn import functional as F

# Define the client model
class ClientModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 32, (3,3), padding=(1, 1))
      self.conv2 = nn.Conv2d(32,32, (3,3))
      self.pool1 = nn.MaxPool2d(2, 2)
      self.drop1 = nn.Dropout2d(p=0.25)


  def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.pool1(x)
      x = self.drop1(x)

      return x



class ServerModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv3 = nn.Conv2d(32,64, (3,3), padding=(1, 1))
      self.conv4 = nn.Conv2d(64,64, (3,3))
      self.drop2 = nn.Dropout2d(p=0.25)
      self.pool2 = nn.MaxPool2d(2, 2)

      self.dense1 = nn.Linear(2304, 512)
      self.dense2 = nn.Linear(512, 10)

  def forward(self, x):
      x = self.conv3(x)
      x = F.relu(x)
      x = self.conv4(x)
      x = F.relu(x)
      x = self.pool2(x)
      self.drop2 = nn.Dropout2d(p=0.25)
      x = torch.flatten(x, 1)

      x = self.dense1(x)
      x = F.relu(x)
      x = self.dense2(x)

      return F.softmax(x)
  

# client_model = ClientModel()
# server_model = ServerModel()