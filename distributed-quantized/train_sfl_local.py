from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.abspath("proto"))
import pickle
import copy

##### CUSTOMIZE MODEL AND DATA #####
from model.models import ClientModel
from model.models import ServerModel
####################################
from optimizer.sgd import create_optimizer
from model.quantization import generate_quantized_model
from utils.utils import *
####################################
from config import experiment_configs

import torch
from torch import nn
from torch.nn import functional as F


from dataset.datasets import get_data_loaders #TODO use test dataset ??
import itertools

load_dotenv()
model_name = os.getenv("MODEL")
quantization_type = os.getenv("QUANTIZATION_TYPE")
dataset_name = os.getenv("DATASET")
split_point = int(os.getenv("SPLIT_POINT"))
learning_rate = float(os.getenv("LEARNING_RATE"))
client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
auto_save_models = int(os.getenv("AUTO_SAVE_MODELS"))
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
dataset_train_size = int(os.getenv("DATASET_TRAIN_SIZE"))
loss_fn = nn.CrossEntropyLoss()
global_request_id = 1
client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
num_clients = len(client_addresses)
device_client = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_quantized = "cpu"
device_server = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and optimizer
# import resnet18
server_model = None
client_models = None
server_optimizer, server_scheduler = None, None

client_optimizers_schedulers = None
client_optimizers = None
client_schedulers = None


train_test_data_loaders = None
train_data_loaders = None
train_iters = None
calib_data_loaders = None
test_data_loader = None


def load_model_and_data():
    global server_model, client_models
    global server_optimizer, server_scheduler
    global client_optimizers, client_schedulers
    global train_data_loaders, train_iters, test_data_loader

    server_model = ServerModel(model_name, quantization_type, split_point=split_point, device=device_server, input_shape=None)
    client_models = [ClientModel(model_name, quantization_type, split_point=split_point, device= device_client, input_shape=image_size) for _ in range(len(client_addresses))]
    server_optimizer, server_scheduler = create_optimizer(server_model.parameters(), learning_rate)

    client_optimizers_schedulers = [create_optimizer(client_model.parameters(), learning_rate) for client_model in client_models]
    client_optimizers = [optimizer for optimizer, _ in client_optimizers_schedulers]
    client_schedulers = [scheduler for _, scheduler in client_optimizers_schedulers]

    train_calib_test_data_loaders = [get_data_loaders(dataset_name, batch_size=client_batch_size, client_id=client_id, num_clients=num_clients, image_size=image_size) for client_id in range(len(client_models))]
    train_data_loaders = [train_data_loader for train_data_loader, _, _ in train_test_data_loaders]
    calib_data_loaders = [calib_data_loader for _, calib_data_loader, _ in train_test_data_loaders]
    train_iters = [iter(train_data_loader) for train_data_loader in train_data_loaders]
    test_data_loader = train_test_data_loaders[0][2]  # Use the first client's test data loader for testing

load_model_and_data()

def initialize_experiment_variables(experiment_config):
    global num_clients
    global model_name, quantization_type, split_point
    global dataset_name, image_size, dataset_train_size
    global client_batch_size

    num_clients = experiment_config["NUM_CLIENTS"]
    model_name = experiment_config["MODEL_NAME"]
    quantization_type = experiment_config["QUANTIZATION_TYPE"]
    split_point = experiment_config["SPLIT_POINT"]

    dataset_name = experiment_config["DATASET_NAME"]
    image_size = experiment_config["IMAGE_SIZE"]
    dataset_train_size = experiment_config["DATASET_TRAIN_SIZE"]
    client_batch_size = experiment_config["CLIENT_BATCH_SIZE"]

    load_model_and_data()



def get_next_train_batch(client_id):
    global train_iters
    batch = next(train_iters[client_id], None)
    if batch is None:
        train_iters[client_id] = iter(train_data_loaders[client_id])
        return next(train_iters[client_id])
    return batch


if auto_load_models:
        # server model
        load_model_if_exists(server_model, model_name, quantization_type, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
        print("Server model loaded")
        # client model
        for client_model in client_models:
            load_model_if_exists(client_model, model_name, quantization_type, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)


def server_forward(tensor_IR, labels):
    # update server model, returns grad of the input 
    # used to continue the backpropagation in client_model
    tensor_IR, labels = tensor_IR.to(device_server), labels.to(device_server)
    tensor_IR.requires_grad = True
    server_optimizer.zero_grad()
    outputs = server_model(tensor_IR)
    loss = loss_fn(outputs, labels)
    loss.backward()
    server_optimizer.step()
    return tensor_IR.grad.detach().to(device_server) #TODO: checar se não é device_client

def server_test_inference(tensor_IR, labels):
    # update server model, returns grad of the input
    # used to continue the backpropagation in client_model
    correct = 0
    total = 0
    loss = 0
    debug_print(torch.unique(labels, return_counts=True))
    with torch.no_grad():
        tensor_IR = tensor_IR.to(device_server).detach()
        tensor_IR.requires_grad = False
        outputs = server_model(tensor_IR).to('cpu')
        loss = loss_fn(outputs, labels)

        pred_class = torch.argmax(outputs, dim=1)
        correct += torch.eq(labels, pred_class).sum().item()
        total += len(labels)
        debug_print('tensor_IR.size()', tensor_IR.size())
        debug_print('outputs.size()', outputs.size())
        debug_print('pred_class.size()', outputs.size())
        debug_print('outputs', outputs)
        debug_print('labels', labels)
        debug_print('pred_class', pred_class)
    return correct, total, loss

def client_process_forward_query(batch_size, client_id): #TODO use batch_size
    client_optimizers[client_id].zero_grad()
    inputs, labels = get_next_train_batch(client_id)
    inputs = inputs.to(device_client)
    outputs = client_models[client_id](inputs)
    return outputs, labels

def client_process_backward_query(output, grad, client_id):
    #client_optimizers[client_id].zero_grad()
    # #TODO: ver isso aqui!!!!!!!!!!
    output.backward(grad)
    client_optimizers[client_id].step()


def aggregate_client_model_params():
    global_model = copy.deepcopy(client_models[0])

    for name, param in global_model.named_parameters():
        new_w = torch.zeros_like(param)
        for c_model in client_models:
            new_w += c_model.state_dict()[name]
        new_w /= num_clients
        global_model.state_dict()[name].copy_(new_w)

    global_model_state = global_model.state_dict()

    for client_model in client_models:
        client_model.load_state_dict(global_model_state)

def train_client_server_models():
    forward_tasks = []
    clients_IRs = []
    clients_labels = []
    clients_request_ids = []
    debug_print("Training models")

    # Zero gradients
    server_optimizer.zero_grad()
    for client_id in range(num_clients):
        client_optimizers[client_id].zero_grad()

    # Forward pass for each client
    debug_print("Waiting for client forward tasks")
    for client_id in range(num_clients):
        client_optimizers[client_id].zero_grad()
        inputs, labels = get_next_train_batch(client_id)
        inputs = inputs.to(device_client)
        tensor_IR = client_models[client_id](inputs)

        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)

    # Concatenate all client IRs and labels
    concat_IRs = torch.concatenate(clients_IRs).detach().to(device_server)
    concat_labels = torch.concatenate(clients_labels).detach().to(device_server)

    # Forward pass on server
    concat_IRs.requires_grad = True
    outputs = server_model(concat_IRs)

    # Backward pass on server
    loss = loss_fn(outputs, concat_labels)
    loss.backward()
    server_optimizer.step()
    
    # Get gradients from server
    concat_IRs_grad = concat_IRs.grad.detach().to(device_client)
    clients_IRs_grad = concat_IRs_grad.split(client_batch_size)

    for client_id, client_IR, client_IR_grad in zip(range(num_clients), clients_IRs, clients_IRs_grad):
        client_IR.backward(client_IR_grad)
        client_optimizers[client_id].step()
    
    debug_print("Aggregating client models")
    aggregate_client_model_params()


def print_test_accuracy(client_model, server_model, quantized=False):
    correct = 0
    total = 0
    loss = 0
    all_measures = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_data_loader):
            if quantized:
                inputs = inputs.to('cpu')
                tensor_IR = client_model(inputs)
                tensor_IR = tensor_IR.dequantize()
            else:
                inputs = inputs.to(device_client)
                tensor_IR = client_model(inputs)

            if device_client != device_server or quantized:
                tensor_IR = tensor_IR.to(device_server)
            outputs = server_model(tensor_IR).to('cpu')
            curr_loss = loss_fn(outputs, labels)
            pred_class = torch.argmax(outputs, dim=1)

            correct += torch.eq(labels, pred_class).sum().item()
            total += len(labels)
            loss += curr_loss.item()

        (quantized and (print("== Quantized Metrics ==") or True)) or print("== Full Precision Metrics ==")
        print(f"Accuracy: {correct} / {total} = {correct / total}")
        print(f"Loss: ", loss)
        print()
    return correct / total


#####################################################################################

def compare_full_and_quantized_model():
    client_model_quantized = generate_quantized_model(client_models[0], calib_data_loaders[0], quantization_type=quantization_type)
    full_acc = print_test_accuracy(client_model=client_models[0], server_model=server_model, quantized=False)
    print_test_accuracy(client_model=client_model_quantized, server_model=server_model, quantized=True)
    return full_acc


def run_experiments(experiment_config=None):
    print("="*100)
    if experiment_config:
        print("Experiment configuration dict:", experiment_config)
        initialize_experiment_variables(experiment_config)
    print("Running experiments with the following configuration:")
    print(f"Model: {model_name}, Quantization: {quantization_type}, Split Point: {split_point}")
    print(f"Dataset: {dataset_name}, Image Size: {image_size}, Train Size: {dataset_train_size}")
    print(f"Number of Clients: {num_clients}, Client Batch Size: {client_batch_size}")
    
    compare_full_and_quantized_model()


    #target_acc = float(input("Set target accuracy (def: 0.6): ") or 0.6)
    iterations_per_epoch = dataset_train_size // (num_clients * client_batch_size)
    print(f"Iterations per epoch: {iterations_per_epoch}")
    num_epochs = int(input("Set number of epochs (def: 200): ") or 200)
    
    epoch = 0
    i = 0
    print("Starting training")
    while True:
        i += 1

        # Training models
        print(f"Training iteration {i}", end="\r")
        train_client_server_models()

        if i % iterations_per_epoch == 0:
            if auto_save_models:
                save_state_dict(server_model.state_dict(), model_name, quantization_type, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
                save_state_dict(client_models[0].state_dict(), model_name, quantization_type, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
            full_acc = compare_full_and_quantized_model() #print_test_accuracy(client_model=client_models[0], server_model=server_model, quantized=False)
            epoch += 1

            server_scheduler.step()
            for client_scheduler in client_schedulers:
                client_scheduler.step()

            # Ensure lr from client equal to the server
            # for client_optimizer in client_optimizers:
            #     for param_group in client_optimizer.param_groups:
            #         param_group['lr'] = server_optimizer.param_groups[0]['lr']

            print(f"Server LR  {server_optimizer.param_groups[0]['lr']:.10f}")
            print(f"Client LR  {client_optimizers[0].param_groups[0]['lr']:.10f}")

            print(f"Epoch: {epoch}")
            print()

            stop_criteria = epoch >= num_epochs
            if stop_criteria:
                print(f"Accuracy {full_acc} reached")
                break

    compare_full_and_quantized_model()

#op = input("Do you want to run all the experiments? (y/n): ").strip().lower()
op = 'n'

if op == 'y':
    for experiment_config in experiment_configs:
        run_experiments(experiment_config=experiment_config)
elif op == 'n':
    run_experiments()
else:
    print("Invalid option")
