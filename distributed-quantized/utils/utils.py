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

def save_state_dict(state_dict, model_name):
    torch.save(state_dict, f"./model-state/{model_name}.pth")

def load_model_if_exists(model, model_name):
    path = f"./model-state/{model_name}.pth"
    print(f"os.path.exists(path): {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"Loading model: {model_name}")
        model.load_state_dict(torch.load(path))
