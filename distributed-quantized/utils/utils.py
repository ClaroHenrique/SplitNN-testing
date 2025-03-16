import os
import pickle
import time
import torch

def debug_print(*args, **kwargs):
    if os.getenv("DEBUG") == "1":
        print(*args, **kwargs)

def model_parameters_sum(model):
    param_sum = 0
    for param in model.parameters():
        param_sum += param.sum().item()
    return param_sum

def size_of_model(model):
    return len(pickle.dumps(model.state_dict()))

def save_state_dict(state_dict, model_name, dataset_name):
    torch.save(state_dict, f"./model-state/{model_name}_{dataset_name}.pth")

def load_model_if_exists(model, model_name, dataset_name):
    path = f"./model-state/{model_name}_{dataset_name}.pth"
    print(f"os.path.exists(path): {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"Loading model: {model_name}_{dataset_name}")
        model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))

def aggregate_measures_mean(measures):
    keys = list(measures[0].keys())
    n = len(measures)
    res = {}
    for k in keys:
        sum = 0
        for measure in measures:
            sum += measure[k]
        res[k] = sum / n
    return res
