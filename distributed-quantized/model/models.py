from model.resnet import *
from model.mobilenetv2 import *
from model.resnet_32x32 import *
from model.test_model import *
from model.quantization import generate_prepared_model_qat
import torch


def get_model(name, num_classes, quantization_type, split_point, is_client, input_shape, device, state_dict):
    if name == 'ResNet18':
        model = resnet18(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet34':
        model = resnet34(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet50':
        model = resnet50(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'MobileNetV2':
        model = MobileNetV2(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet18_32x32':
        model = ResNet18_32x32(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet34_32x32':
        model = ResNet34_32x32(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet50_32x32':
        model = ResNet50_32x32(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet101_32x32':
        model = ResNet101_32x32(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'ResNet152_32x32':
        model = ResNet152_32x32(num_classes=num_classes, split_point=split_point, is_client=is_client)
    elif name == 'test_model':
        model = test_model(num_classes=num_classes, split_point=split_point, is_client=is_client)
    else:
        raise ValueError(f"Model {name} not supported.")
    
    if state_dict is not None:
        model.load_state_dict(state_dict)

    device = torch.device(device)

    model.to(device)
    return model

def ServerModel(name, num_classes, quantization_type, split_point, input_shape, device, state_dict=None):
    return get_model(name, num_classes, quantization_type, split_point, False, input_shape, device, state_dict=state_dict)

def ClientModel(name, num_classes, quantization_type, split_point, input_shape, device, state_dict=None):
    return get_model(name, num_classes, quantization_type, split_point, True, input_shape, device, state_dict=state_dict)
