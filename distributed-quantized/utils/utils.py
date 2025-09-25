import os
import pickle
import time
import torch
import string
import random
import csv

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
    

def save_state_dict(state_dict, model_name, quantization_type, split_point, is_client, num_clients, dataset_name):
    if is_client:
        model_name = f"{model_name}_{quantization_type}_s{split_point}_client_n{num_clients}"
    else:
        model_name = f"{model_name}_{quantization_type}_s{split_point}_server_n{num_clients}"
    torch.save(state_dict, f"./model-state/{model_name}_{dataset_name}.pth")

def load_model_if_exists(model, model_name, quantization_type, split_point, is_client, num_clients, dataset_name):
    if is_client:
        model_name = f"{model_name}_{quantization_type}_s{split_point}_client_n{num_clients}"
    else:
        model_name = f"{model_name}_{quantization_type}_s{split_point}_server_n{num_clients}"
    path = f"./model-state/{model_name}_{dataset_name}.pth"
    print(f"os.path.exists({path}): {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"Loading model: {model_name}_{dataset_name}")
        model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))

def generate_run_id(length=8):
    chars = string.ascii_letters + string.digits
    return "run_" + "".join(random.choice(chars) for _ in range(length))

def save_training_results_in_file(file_name, run_id, start_time, epoch, full_accuracy, quant_accuracy, model_name, quantization_type, split_point, num_clients, dataset_name, optimizer_name, learning_rate):
    file_exists = os.path.isfile(file_name)
    duration = time.time() - start_time
    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "duration (s)", "epoch", "full_accuracy", "quant_accuracy",
                "model_name", "quantization_type", "split_point",
                "num_clients", "dataset_name", "optimizer_name", "learning_rate",
            ])
        writer.writerow([
            run_id,
            f"{duration:.1f}",
            epoch,
            f"{full_accuracy:.8f}",
            f"{quant_accuracy:.8f}",
            model_name,
            quantization_type,
            split_point,
            num_clients,
            dataset_name,
            optimizer_name,
            learning_rate,
        ])

def save_inference_measures_in_file(file_name, run_id, model_name, quantization_type, split_point, dataset_name, batch_size, measures):
    file_exists = os.path.isfile(file_name)

    columns = ["run_id", "model_name", "quantization_type", "split_point", "dataset_name", "batch_size"]
    measures_columns = [k for k in measures[0].keys()]

    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns + measures_columns)
        
        for measure in measures:
            writer.writerow(
                [
                    run_id,
                    model_name,
                    quantization_type,
                    split_point,
                    dataset_name,
                    batch_size,
                ] + [measure[k] for k in measures_columns]
            )

def aggregate_measures_mean(measures):
    keys = list(measures[0].keys())
    n = len(measures)
    res = {}
    print(f"Measures (count: {n}):")
    for k in keys:
        print(f"{k}", end="")
        sum = 0
        for measure in measures:
            print(f", {measure[k]}", end="")
            sum += measure[k]
        res[k] = sum / n
        print()
    return res
