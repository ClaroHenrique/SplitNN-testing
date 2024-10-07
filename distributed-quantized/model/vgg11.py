import torch
from torch import nn
from torch.nn import functional as F

# Define the client model
class ClientModel(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    # convolutional layers 
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
  def forward(self, x):
    out = self.conv_layers(x)
    # flatten to prepare for the fully connected layers
    out = out.reshape(out.size(0), -1)
    return out

class ServerModel(nn.Module):
  def __init__(self, num_classes=100):
    super().__init__()
    # fully connected linear layers
    self.linear_layers = nn.Sequential(
      nn.Linear(in_features=512*7*7, out_features=4096),
      nn.ReLU(),
      nn.Dropout2d(0.5),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Dropout2d(0.5),
      nn.Linear(in_features=4096, out_features=num_classes)
    )

  def forward(self, x):
    return self.linear_layers(x)
  

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
      if i >= 4: #TODO: limit accessed data
        print("calibrating", i)
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
  return quantized_model
