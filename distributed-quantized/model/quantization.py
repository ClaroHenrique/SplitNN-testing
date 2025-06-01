############  QUANTIZATE MODEL ############
import torch
import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

def calibrate(model, dataloader):
    for i, (x_batch, y_batch) in enumerate(dataloader):
        model(x_batch)
        if i >= 4: #TODO: limit accessed data
            print("calibrating", i)
            break

def generate_quantized_model(model, calib_dataloader):
  # Quantization config
  # Copy model to CPU
  model = copy.deepcopy(model)
  model.to(torch.device('cpu'))
  qconfig = get_default_qconfig('x86')
  qconfig_mapping = QConfigMapping().set_global(qconfig) # fully quantize
  prepare_custom_config_dict = {'output_quantized_idxs': [0]} # set quantized output
  model.eval()
  with torch.no_grad():
    # Quantize model
    example_inputs = (next(iter(calib_dataloader)))
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs, prepare_custom_config_dict)
    calibrate(prepared_model, calib_dataloader)

    quantized_model = convert_fx(prepared_model)

  return quantized_model