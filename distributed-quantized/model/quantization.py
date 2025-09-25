############  QUANTIZATE MODEL ############
import torch
import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization import quantize_fx

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import get_default_qat_qconfig_mapping

def calibrate(model, dataloader, batch_limit=10):
    for i, (x_batch, y_batch) in enumerate(dataloader):
        model(x_batch)
        if i >= batch_limit - 1:  # Limit accessed data based on batch_limit
            print("calibrating", i)
            break

def generate_quantized_model_ptq(model, calib_dataloader, backend='x86'):
    # Quantization config
    # Copy model to CPU
    model = copy.deepcopy(model)
    model.to(torch.device('cpu'))
    qconfig = get_default_qconfig(backend)
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


def generate_prepared_model_qat(model, input_shape, backend='x86'):
    # # Per channel doesnt work in GPU
    # per_tensor_qat_qconfig = torch.quantization.QConfig(
    #     activation=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
    #         observer=torch.quantization.MovingAverageMinMaxObserver,
    #         quant_min=0,
    #         quant_max=255,
    #         dtype=torch.quint8,
    #         qscheme=torch.per_tensor_affine
    #     ),
    #     weight=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
    #         observer=torch.quantization.MovingAverageMinMaxObserver,
    #         quant_min=-128,
    #         quant_max=127,
    #         dtype=torch.qint8,
    #         qscheme=torch.per_tensor_affine
    #     )
    # )
    # qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    # qconfig_mapping.set_global(per_tensor_qat_qconfig)

    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.train()
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, torch.zeros(input_shape))
    return model_prepared

def generate_quantized_model_qat(model_prepared):
    # Model is already in QAT mode and trained
    model_prepared = copy.deepcopy(model_prepared)
    quantized_model = quantize_fx.convert_fx(model_prepared)
    #quantized_model.to(torch.device('cpu'))
    return quantized_model

def generate_quantized_model(model, calib_dataloader, quantization_type):
    if quantization_type == 'ptq':
        return generate_quantized_model_ptq(model, calib_dataloader)
    elif quantization_type == 'qat':
        return generate_quantized_model_qat(model)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")