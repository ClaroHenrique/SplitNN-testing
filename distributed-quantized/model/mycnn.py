import torch
from torch import nn
from torch.nn import functional as F

# Define the client model
class ClientModel(nn.Module):
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

class ServerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(400, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  

client_model = ClientModel()
server_model = ServerModel()

############  QUANTIZATE MODEL ############
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

def calibrate(model, dataloader):
  model.eval()
  with torch.no_grad():
    for i, (x_batch, y_batch) in enumerate(dataloader):
      model(x_batch)
      if i >= 100: #TODO: limit accessed data
        break

def generate_quantized_model(model, calib_dataloader):
  # Quantization config
  qconfig = get_default_qconfig('x86')
  qconfig_mapping = QConfigMapping().set_global(qconfig)

  # Quantize model
  example_inputs = (next(iter(calib_dataloader)))
  prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
  calibrate(prepared_model, calib_dataloader)

  quantized_model = convert_fx(prepared_model)
  quantized_model




