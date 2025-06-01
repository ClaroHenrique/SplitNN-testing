from model.resnet import *
from model.resnet_custom import *
from model.test_model import *
import torch


def get_model(name, split_point, is_client):
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
    

    # If the server is running on a GPU, use it
    if torch.cuda.is_available() and not is_client:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    return model

def ServerModel(name, split_point):
    return get_model(name, split_point, is_client=False)

def ClientModel(name, split_point):
    return get_model(name, split_point, is_client=True)



