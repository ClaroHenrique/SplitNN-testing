import os
import pickle
import time

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
