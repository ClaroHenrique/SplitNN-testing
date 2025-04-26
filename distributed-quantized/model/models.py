from model.resnet import *
import torch


def get_model(name, split_point, is_client):
    if name == 'resnet18':
        model = resnet18(split_point=split_point, is_client=is_client)
    elif name == 'resnet34':
        model = resnet34(split_point=split_point, is_client=is_client)
    elif name == 'resnet50':
        model = resnet50(split_point=split_point, is_client=is_client)
    # elif name == 'resnet101':
    #     model = resnet101(split_point=split_point, is_client=is_client)
    # elif name == 'resnet152':
    #     model = resnet152(split_point=split_point, is_client=is_client)
    else:
        raise ValueError(f"Model {name} not supported.")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    return model

def ServerModel(name, split_point):
    return get_model(name, split_point, is_client=False)

def ClientModel(name, split_point):
    return get_model(name, split_point, is_client=True)



