from model.resnet import *
from model.resnet_custom import *
from model.test_model import *
from model.quantization import generate_prepared_model_qat
import torch


def get_model(name, quantization_type, split_point, is_client, input_shape, device, state_dict):
    if name == 'resnet18':
        model = resnet18(split_point=split_point, is_client=is_client)
    elif name == 'resnet34':
        model = resnet34(split_point=split_point, is_client=is_client)
    elif name == 'resnet50':
        model = resnet50(split_point=split_point, is_client=is_client)
    elif name == 'ResNet18_custom':
        model = ResNet18_custom(split_point=split_point, is_client=is_client)
    elif name == 'ResNet34_custom':
        model = ResNet34_custom(split_point=split_point, is_client=is_client)
    elif name == 'ResNet50_custom':
        model = ResNet50_custom(split_point=split_point, is_client=is_client)
    elif name == 'ResNet101_custom':
        model = ResNet101_custom(split_point=split_point, is_client=is_client)
    elif name == 'ResNet152_custom':
        model = ResNet152_custom(split_point=split_point, is_client=is_client)
    elif name == 'test_model':
        model = test_model(split_point=split_point, is_client=is_client)
    else:
        raise ValueError(f"Model {name} not supported.")
    
    if state_dict is not None:
        model.load_state_dict(state_dict)

    device = torch.device(device)

    if is_client and quantization_type == 'qat':
        model = generate_prepared_model_qat(model, input_shape)

    model.to(device)
    return model

def ServerModel(name, quantization_type, split_point, input_shape, device, state_dict=None):
    return get_model(name, quantization_type, split_point, False, input_shape, device, state_dict=state_dict)

def ClientModel(name, quantization_type, split_point, input_shape, device, state_dict=None):
    return get_model(name, quantization_type, split_point, True, input_shape, device, state_dict=state_dict)
