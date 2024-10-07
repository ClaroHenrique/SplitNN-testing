import torch
from torch import nn
from torch.nn import functional as F

# Define the client model
class ClientModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU())
    self.layer2 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(), 
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU())
    self.layer4 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer5 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU())
    self.layer6 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU())

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    return out

class ServerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer7 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer8 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer9 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer10 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer11 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer12 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU())
    self.layer13 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.fc = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(7*7*512, 4096),
      nn.ReLU())
    self.fc1 = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU())
    self.fc2= nn.Sequential(
      nn.Linear(4096, 10))

  def forward(self, x):
    out = self.layer7(x)
    out = self.layer8(out)
    out = self.layer9(out)
    out = self.layer10(out)
    out = self.layer11(out)
    out = self.layer12(out)
    out = self.layer13(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out
  

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
  return quantized_model
