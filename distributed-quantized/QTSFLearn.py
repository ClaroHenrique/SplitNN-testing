#This is a prototype of the QSFLearn algorithm.
#This is not a distributed implementation, but it simulates one.

from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.abspath("proto"))
import pickle

##### CUSTOMIZE MODEL AND DATA #####
from model.models import ClientModel
from model.models import ServerModel
####################################
from optimizer.sgd import create_optimizer
from model.quantization import generate_quantized_model
from utils.utils import *

import torch
from torch import nn
from torch.nn import functional as F
import copy


load_dotenv()
model_name = os.getenv("MODEL")
dataset_name = os.getenv("DATASET")
learning_rate = float(os.getenv("LEARNING_RATE"))
client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
split_point = int(os.getenv("SPLIT_POINT"))
auto_save_models = int(os.getenv("AUTO_SAVE_MODELS"))
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
iterations_per_epoch = int(os.getenv("ITERATIONS_PER_EPOCH"))
loss_fn = nn.CrossEntropyLoss()
global_request_id = 1
client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
num_clients = len(client_addresses)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and optimizer
# import resnet18
server_model = ServerModel(model_name, split_point=split_point)
client_models = [ClientModel(model_name, split_point=split_point) for _ in range(num_clients)]
client_quantized_models = [None for _ in range(num_clients)]
server_optimizer, server_scheduler = create_optimizer(server_model.parameters(), learning_rate)

client_optimizers_schedulers = [create_optimizer(client_model.parameters(), learning_rate) for client_model in client_models]
client_optimizers = [optimizer for optimizer, _ in client_optimizers_schedulers]
client_schedulers = [scheduler for _, scheduler in client_optimizers_schedulers]

from dataset.datasets import get_data_loaders #TODO use test dataset ??
import itertools

train_test_data_loaders = [get_data_loaders(dataset_name, batch_size=client_batch_size, client_id=client_id, num_clients=num_clients, image_size=image_size) for client_id in range(len(client_models))]
train_iters = [itertools.cycle(train_data_loader) for train_data_loader, _ in train_test_data_loaders]
test_iters = [itertools.cycle(test_data_loader) for _, test_data_loader in train_test_data_loaders]


if auto_load_models:
        # server model
        load_model_if_exists(server_model, model_name, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
        print("Server model loaded")
        # client model
        for client_model in client_models:
            load_model_if_exists(client_model, model_name, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)


def server_forward(tensor_IR, labels):
    # update server model, returns grad of the input 
    # used to continue the backpropagation in client_model
    tensor_IR, labels = tensor_IR.to(device), labels.to(device)
    tensor_IR.requires_grad = True
    server_optimizer.zero_grad()
    outputs = server_model(tensor_IR)
    loss = loss_fn(outputs, labels)
    loss.backward()
    server_optimizer.step()
    server_scheduler.step()
    debug_print("updating server model")
    debug_print(torch.unique(labels, return_counts=True))
    debug_print("LR", server_scheduler.get_last_lr())
    return tensor_IR.grad.detach().to('cpu')

def server_test_inference(tensor_IR, labels):
    # update server model, returns grad of the input
    # used to continue the backpropagation in client_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    loss = 0
    debug_print(torch.unique(labels, return_counts=True))
    with torch.no_grad():
        tensor_IR = tensor_IR.to(device).detach()
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


def generate_client_quantized_models():
    quantized_model = generate_quantized_model(client_models[0], train_iters[0])
    global client_quantized_models
    client_quantized_models = [copy.deepcopy(quantized_model) for _ in range(num_clients)]


# Following forwards will not be used to update the model
def client_forward(model, batch_size, client_id):
    with torch.no_grad():
        inputs, labels = next(train_iters[client_id])
        outputs = model(inputs)
    return outputs, labels

def client_forward_quantized(batch_size, client_id):
    return client_forward(client_quantized_models[client_id], batch_size, client_id)

def client_forward_full_precision(batch_size, client_id):
    return client_forward(client_models[client_id], batch_size, client_id)

# Now we will use this forward to update the model
def client_forward_train(batch_size, client_id):
    client_optimizers[client_id].zero_grad()
    inputs, labels = next(train_iters[client_id])
    outputs = client_models[client_id](inputs)
    return outputs, labels

def client_process_backward_query(output, grad, client_id):
    client_optimizers[client_id].zero_grad()
    output.backward(grad)
    client_optimizers[client_id].step()
    #set LR manually
    for param_group in client_optimizers[client_id].param_groups:
        param_group['lr'] = server_optimizer.param_groups[0]['lr']


def aggregate_client_model_params():
    num_clients = len(client_models)

    client_model_states = [client_model.state_dict() for client_model in client_models]
    model_state_keys = client_model_states[0].keys()
    client_model_weights = [list(model_state.values()) for model_state in client_model_states]

    # aggregate parameters by mean
    aggregated_params = []
    for l, layer_key in enumerate(model_state_keys):
        layer_params = []
        #debug_print(layer_key, 'num_batches_tracked' in layer_key)

        if 'num_batches_tracked' not in layer_key:
            for client_w in client_model_weights:
                layer_params.append(client_w[l])
            layer_mean = torch.stack(layer_params, dim=0).mean(dim=0)
        else:
            layer_mean = client_model_weights[0][l].detach()
        aggregated_params.append(layer_mean)
    
    # get aggregated weights from server
    new_state_dict = dict(zip(model_state_keys, aggregated_params))
    for client_model in client_models:
        client_model.load_state_dict(new_state_dict)

# Train both client and server models with one train batch
def train_client_server_models(use_quant_clients: bool, train_clients: bool):
    if use_quant_clients and train_clients:
        raise ValueError("Cannot train quantized clients")

    forward_tasks = []
    clients_IRs = []
    clients_labels = []
    clients_request_ids = []
    num_clients = len(client_models)
    debug_print("Training models")

    debug_print("Waiting for client forward tasks")
    for client_id in range(num_clients):
        if use_quant_clients:
            tensor_IR, labels = client_forward_quantized(client_batch_size, client_id)
            tensor_IR = tensor_IR.dequantize()
        else:
            tensor_IR, labels = client_forward_train(client_batch_size, client_id)
        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)

    debug_print("Concat IRs and labels")
    # Concat IR and labels to feed and train server model
    concat_IRs = torch.concatenate(clients_IRs).detach()
    concat_labels = torch.concatenate(clients_labels).detach()
    debug_print("Feed server model with IRs")
    concat_IRs_grad = server_forward(concat_IRs, concat_labels)

    # Train client models
    if train_clients:
        debug_print("Sending backward requests to clients")
        backward_tasks = []
        clients_IRs_grad = concat_IRs_grad.split(client_batch_size)
        for client_id, client_IR, client_IR_grad in zip(range(len(client_models)), clients_IRs, clients_IRs_grad):
            client_process_backward_query(client_IR, client_IR_grad, client_id)
        debug_print("Aggregating client models")
        aggregate_client_model_params()


def print_test_accuracy(num_instances, model, quantized=False):
    correct = 0
    total = 0
    loss = 0
    all_measures = []

    while True:
        # TODO: use test dataset
        client_data_sample = next(test_iters[0])
        tensor_IR = model(client_data_sample[0])
        if quantized:
            tensor_IR = tensor_IR.dequantize()
        labels = client_data_sample[1]

        c_correct, c_total, c_loss = server_test_inference(tensor_IR, labels)
        correct += c_correct
        total += c_total
        loss += c_loss.item() / 1 #len(client_models)
        # all_measures.append(measure) TODO: get measures from cliente
        # print(f"Accuracy progress {correct} / {total} = {correct / total}")
        if total >= num_instances:
            break
        
        
    (quantized and (print("== Quantized Metrics ==") or True)) or print("== Full Precision Metrics ==")
    print(f"Accuracy: {correct} / {total} = {correct / total}")
    print(f"Loss: ", loss)
    #print(f"Measure mean: ", aggregate_measures_mean(all_measures))
    print()
    return correct / total



#####################################################################################
#client_model_quantized = generate_quantized_model(client_models[0], train_iters[0])
generate_client_quantized_models()
print_test_accuracy(num_instances=10000, model=client_models[0], quantized=False)
print_test_accuracy(num_instances=10000, model=client_quantized_models[0], quantized=True)

target_acc = float(input("Set target accuracy (def: 0.6): ") or 0.6)
metrics_per_epoch = {}
client_full_iterations, client_quant_iterations = 10, 20

iteration = 0
while True:
    iteration += 1
    print(f"Iteration: {iteration}")
    epoch = iteration // iterations_per_epoch + 1
    print("Epoch:", int(epoch))
    
    current_metrics = {}
    
    # Train server model and client models with full precision
    start_time = time.time()
    for _ in range(client_full_iterations):
        train_client_server_models(use_quant_clients=False, train_clients=True)
    full_precision_time = (time.time() - start_time) / client_full_iterations

    # Update client quantized models
    generate_client_quantized_models()

    # Train server model only, with quantized client models
    for _ in range(client_quant_iterations):
        start_time = time.time()
        train_client_server_models(use_quant_clients=True, train_clients=False)
    quantized_time = (time.time() - start_time) / client_quant_iterations
    print(f"Quantized training time: {quantized_time:.2f} seconds")

    if auto_save_models:
        save_state_dict(server_model.state_dict(), model_name, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
        save_state_dict(client_models[0].state_dict(), model_name, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
    
    
    print(f"Server LR  {server_optimizer.param_groups[0]['lr']:.10f}")
    print(f"Client LR  {client_optimizers[0].param_groups[0]['lr']:.10f}")

    
    
    if iteration % iterations_per_epoch == 0:
        full_acc = print_test_accuracy(num_instances=10000, model=client_models[0], quantized=False)
        quant_acc = print_test_accuracy(num_instances=10000, model=client_quantized_models[0], quantized=True)
        
        current_metrics["full_acc"] = full_acc
        current_metrics["quant_acc"] = quant_acc
        current_metrics["lr"] = server_optimizer.param_groups[0]['lr']
        current_metrics["full_precision_time"] = full_precision_time
        current_metrics["quantized_time"] = quantized_time
        metrics_per_epoch[epoch] = current_metrics
        print("Metrics:", metrics_per_epoch[epoch])
        
        stop_criteria = full_acc >= target_acc
        if stop_criteria:
            print(f"Accuracy {full_acc} reached")
            break
    
for epoch, metrics in metrics_per_epoch.items():
    print(f"Epoch {epoch:03d}: {metrics}")


# # send client_model weights to the server for aggregation
# client_model_state_dict = client_model.state_dict()
# client_model_params = list(client_model_state_dict.values())

# # get aggregated weights from server
# new_state_dict = dict(zip(client_model_state_dict.keys(), aggregated_params))
# client_model.load_state_dict(new_state_dict)
