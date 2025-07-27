from dotenv import load_dotenv
import asyncio
import os
import sys
# /SplitNN-testing/distributed-quantized/model-state/ResNet34_custom_s2_client_n8_cifar10_non_iid.pth

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

import torch
from torch import nn
from torch.nn import functional as F


from dataset.datasets import get_data_loaders #TODO use test dataset ??
import itertools

load_dotenv()
model_name = os.getenv("MODEL")
quantization_type = os.getenv("QUANTIZATION_TYPE")
dataset_name = os.getenv("DATASET")
learning_rate = float(os.getenv("LEARNING_RATE"))
client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
split_point = int(os.getenv("SPLIT_POINT"))
auto_save_models = int(os.getenv("AUTO_SAVE_MODELS"))
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
dataset_train_size = int(os.getenv("DATASET_TRAIN_SIZE"))
loss_fn = nn.CrossEntropyLoss()
global_request_id = 1
client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
num_clients = len(client_addresses)
device_client = "cpu"
device_server = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and optimizer
# import resnet18
server_model = ServerModel(model_name, quantization_type, split_point=split_point, device=device_server, input_shape=None)
client_models = [ClientModel(model_name, quantization_type, split_point=split_point, device= device_client, input_shape=image_size) for _ in range(len(client_addresses))]
server_optimizer, server_scheduler = create_optimizer(server_model.parameters(), learning_rate)

client_optimizers_schedulers = [create_optimizer(client_model.parameters(), learning_rate) for client_model in client_models]
client_optimizers = [optimizer for optimizer, _ in client_optimizers_schedulers]
client_schedulers = [scheduler for _, scheduler in client_optimizers_schedulers]


train_test_data_loaders = [get_data_loaders(dataset_name, batch_size=client_batch_size, client_id=client_id, num_clients=num_clients, image_size=image_size) for client_id in range(len(client_models))]
train_data_loaders = [train_data_loader for train_data_loader, _ in train_test_data_loaders]
train_iters = [iter(train_data_loader) for train_data_loader in train_data_loaders]
# test_iters = [itertools.cycle(test_data_loader) for _, test_data_loader in train_test_data_loaders]
test_data_loader = train_test_data_loaders[0][1]  # Use the first client's test data loader for testing


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
    return tensor_IR.grad.detach().to(device_server)

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
    client_optimizers[client_id].zero_grad()
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

    debug_print("Waiting for client forward tasks")
    for client_id in range(num_clients):
        tensor_IR, labels = client_process_forward_query(client_batch_size, client_id)
        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)

    debug_print("Concat IRs and labels")
    # Concat IR and labels to feed and train server model
    concat_IRs = torch.concatenate(clients_IRs).detach()
    concat_labels = torch.concatenate(clients_labels).detach()
    debug_print("Feed server model with IRs")
    concat_IRs_grad = server_forward(concat_IRs, concat_labels)
    debug_print(concat_IRs.sum())

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

    for batch_idx, (inputs, labels) in enumerate(test_data_loader):
        if quantized:
            tensor_IR = model(inputs)
            tensor_IR = tensor_IR.dequantize()
        else:
            inputs = inputs.to(device_client)
            tensor_IR = model(inputs)
        
        c_correct, c_total, c_loss = server_test_inference(tensor_IR, labels)
        correct += c_correct
        total += c_total
        loss += c_loss.item()  # / len(client_models)
        # all_measures.append(measure) TODO: get measures from cliente
        # print(f"Accuracy progress {correct} / {total} = {correct / total}")
        
        
    (quantized and (print("== Quantized Metrics ==") or True)) or print("== Full Precision Metrics ==")
    print(f"Accuracy: {correct} / {total} = {correct / total}")
    print(f"Loss: ", loss)
    #print(f"Measure mean: ", aggregate_measures_mean(all_measures))
    print()
    return correct / total



#####################################################################################
client_model_quantized = generate_quantized_model(client_models[0], train_iters[0], quantization_type=quantization_type)
#print_test_accuracy(num_instances=10000, model=client_models[0], quantized=False)
#print_test_accuracy(num_instances=10000, model=client_model_quantized, quantized=True)


#target_acc = float(input("Set target accuracy (def: 0.6): ") or 0.6)
num_epochs = int(input("Set number of epochs (def: 200): ") or 200)

iterations_per_epoch = dataset_train_size // (num_clients * client_batch_size)
epoch = 0
i = 0
while True:
    i += 1

    # Training models
    print(f"Training iteration {i}")
    train_client_server_models()


    if i % iterations_per_epoch == 0:
        if auto_save_models:
            save_state_dict(server_model.state_dict(), model_name, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
            save_state_dict(client_models[0].state_dict(), model_name, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
        full_acc = print_test_accuracy(num_instances=10000, model=client_models[0], quantized=False)
        epoch += 1
        print(f"Epoch: {epoch}")

        server_scheduler.step()
        for client_optimizer in client_optimizers:
            for param_group in client_optimizer.param_groups:
                param_group['lr'] = server_optimizer.param_groups[0]['lr']

        print(f"Server LR  {server_optimizer.param_groups[0]['lr']:.10f}")
        print(f"Client LR  {client_optimizers[0].param_groups[0]['lr']:.10f}")

        stop_criteria = epoch >= num_epochs
        if stop_criteria:
            print(f"Accuracy {full_acc} reached")
            break
       


# # send client_model weights to the server for aggregation
# client_model_state_dict = client_model.state_dict()
# client_model_params = list(client_model_state_dict.values())

# # get aggregated weights from server
# new_state_dict = dict(zip(client_model_state_dict.keys(), aggregated_params))
# client_model.load_state_dict(new_state_dict)
