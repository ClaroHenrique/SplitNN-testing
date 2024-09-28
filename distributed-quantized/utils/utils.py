import os

def debug_print(*args, **kwargs):
    if os.getenv("DEBUG") == "1":
        print(*args, **kwargs)

def model_parameters_sum(model):
    param_sum = 0
    for param in model.parameters():
        param_sum += param.sum().item()
    return param_sum
