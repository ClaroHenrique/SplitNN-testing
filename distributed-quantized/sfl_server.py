from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.abspath("proto"))
import grpc
import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2
import pickle

##### CUSTOMIZE MODEL AND DATA #####
from model.models import ClientModel
from model.models import ServerModel
from dataset.cifar10_non_iid import get_dataset_name
####################################
from optimizer.adam import create_optimizer
from utils.utils import *

import torch
from torch import nn
from torch.nn import functional as F

load_dotenv()
loss_fn = nn.CrossEntropyLoss()
learning_rate = float(os.getenv("LEARNING_RATE"))
client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
model_option = os.getenv("MODEL")
split_point = int(os.getenv("SPLIT_POINT"))
auto_save_models = int(os.getenv("AUTO_SAVE_MODELS"))
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
dataset_name = get_dataset_name()
global_request_id = 1


# Load model and optimizer
server_model, model_name = ServerModel(model_option, split_point=split_point)
_, client_model_name = ClientModel(model_option, split_point=split_point)
if auto_load_models:
    load_model_if_exists(server_model, model_name, dataset_name)
optimizer, scheduler = create_optimizer(server_model.parameters(), learning_rate)


def server_forward(tensor_IR, labels):
    # update server model, returns grad of the input 
    # used to continue the backpropagation in client_model
    tensor_IR.requires_grad = True
    optimizer.zero_grad()
    outputs = server_model(tensor_IR)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    #scheduler.step()
    debug_print("updating server model")
    debug_print(torch.unique(labels, return_counts=True))
    debug_print("LR", scheduler.get_last_lr())
    return tensor_IR.grad.detach()

def server_test_inference(tensor_IR, labels):
    # update server model, returns grad of the input
    # used to continue the backpropagation in client_model
    correct = 0
    total = 0
    loss = 0
    debug_print(torch.unique(labels, return_counts=True))
    with torch.no_grad():
        tensor_IR.requires_grad = False
        outputs = server_model(tensor_IR)
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

class DistributedClient(object):
    """
    Client for gRPC functionality
    """
    def __init__(self, address, message_max_size):
        options=[
            ('grpc.max_send_message_length', message_max_size),
            ('grpc.max_receive_message_length', message_max_size),
        ]
        self.channel = grpc.insecure_channel(address, options)            # instantiate a channel
        self.stub = pb2_grpc.DistributedClientStub(self.channel) # bind the client and the server

    def forward(self, batch_size, request_id):
        query = pb2.Query(batch_size=batch_size, request_id=request_id, status=0)
        response = self.stub.Forward(query)
        tensor_IR = pickle.loads(response.tensor)
        labels = pickle.loads(response.label)
        request_id = response.request_id
        debug_print("IR", tensor_IR, labels)
        return tensor_IR, labels, request_id

    def backward(self, grad, request_id): # TODO: implement request_id
        debug_print("GRAD", grad)
        grad = pickle.dumps(grad)
        tensor = pb2.Tensor(tensor=grad, label=None, request_id=request_id)
        return self.stub.Backward(tensor)
    
    def get_model_state(self):
        query = pb2.Empty()
        response = self.stub.GetModelState(query)
        model_state = pickle.loads(response.state)
        return model_state

    def set_model_state(self, model_state): # TODO: implement request_id
        model_state = pickle.dumps(model_state)
        model_state = pb2.ModelState(state=model_state)
        return self.stub.SetModelState(model_state)
    
    def generate_quantized_model(self):
        query = pb2.Empty()
        return self.stub.GenerateQuantizedModel(query)
    
    def test_inference(self, batch_size, quantized=False): # TODO: ForwardTest ao inves de test_inference
        query = pb2.Query(batch_size=batch_size, request_id=-1, status=0)
        if quantized:
            response = self.stub.TestQuantizedInference(query)
        else:
            response = self.stub.TestInference(query)

        tensor_IR = pickle.loads(response.tensor.tensor)
        labels = pickle.loads(response.tensor.label)
        measure = pickle.loads(response.measure.measure)
        if quantized:
            tensor_IR = tensor_IR.dequantize()
        return tensor_IR, labels, measure
    
    def test_quantized_inference(self, batch_size): # TODO: use batchsize
        return self.test_inference(batch_size, quantized=True)

async def train_client_server_models(clients):
    forward_tasks = []
    clients_IRs = []
    clients_labels = []
    clients_request_ids = []
    num_clients = len(clients)

    async def call_client_forward_async(client, batch_size, request_id):
        return client.forward(batch_size=batch_size, request_id=request_id)

    # Collect IR and Labels from the client
    for client in clients:
        global global_request_id
        #tensor_IR, labels, request_id = client.forward(batch_size=client_batch_size, request_id=global_request_id)
        forward_tasks.append(
            asyncio.create_task(
                call_client_forward_async(client, client_batch_size, global_request_id)
            )
        )
        global_request_id += 1
    
    for task in forward_tasks:
        tensor_IR, labels, request_id = await task
        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)
        clients_request_ids.append(request_id)

    # Concat IR and labels to feed and train server model
    concat_IRs = torch.concatenate(clients_IRs).detach()
    concat_labels = torch.concatenate(clients_labels).detach()
    concat_IRs_grad = server_forward(concat_IRs, concat_labels)

    if auto_save_models:
        save_state_dict(server_model.state_dict(), "server", dataset_name)

    debug_print(concat_IRs.sum())

    # Send IRs gradients back to the clients (train client model)
    async def call_client_backward_async(client, grad, request_id):
        return client.backward(grad=grad, request_id=req_id)

    backward_tasks = []
    clients_IRs_grad = concat_IRs_grad.split(client_batch_size)
    for client, client_IR_grad, req_id in zip(clients, clients_IRs_grad, clients_request_ids):
        backward_tasks.append(
            asyncio.create_task(
                call_client_backward_async(client, client_IR_grad, request_id=req_id))
        )
    await asyncio.gather(*backward_tasks)


def print_test_accuracy(clients, num_instances, quantized=False):
    correct = 0
    total = 0
    loss = 0
    all_measures = []

    while True:
        for client in clients:
            if quantized:
                tensor_IR, labels, measure = client.test_quantized_inference(batch_size=client_batch_size) #TODO fix batchsize
            else:
                tensor_IR, labels, measure = client.test_inference(batch_size=client_batch_size) #TODO fix batchsize

            c_correct, c_total, c_loss = server_test_inference(tensor_IR, labels)
            correct += c_correct
            total += c_total
            loss += c_loss.item() / len(clients)
            all_measures.append(measure)
            #print(f"Accuracy progress {correct} / {total} = {correct / total}")
            break
        if total >= num_instances:
            break
        
        
    (quantized and (print("== Quantized Metrics ==") or True)) or print("== Full Precision Metrics ==")
    print(f"Accuracy: {correct} / {total} = {correct / total}")
    print(f"Loss: ", loss)
    print(f"Measure mean: ", aggregate_measures_mean(all_measures))
    print()
    return correct / total

async def aggregate_client_model_params(clients):
    num_clients = len(clients)

    async def call_client_get_model_async(client):
        return client.get_model_state()

    get_model_tasks = [asyncio.create_task(call_client_get_model_async(client)) for client in clients]
    client_model_states = await asyncio.gather(*get_model_tasks)

    model_state_keys = client_model_states[0].keys()
    client_model_weights = [list(model_state.values()) for model_state in client_model_states]

    # aggregate parameters by mean
    aggregated_params = []
    for l, layer_key in enumerate(model_state_keys):
        layer_params = []
        debug_print(layer_key, 'num_batches_tracked' in layer_key)

        if 'num_batches_tracked' not in layer_key:
            for client_w in client_model_weights:
                layer_params.append(client_w[l])
            layer_mean = torch.stack(layer_params, dim=0).mean(dim=0)
        else:
            layer_mean = client_model_weights[0][l].detach()
        aggregated_params.append(layer_mean)
    
    # get aggregated weights from server
    new_state_dict = dict(zip(model_state_keys, aggregated_params))
    if auto_save_models:
        save_state_dict(new_state_dict, client_model_name, dataset_name)

    async def call_client_set_model_async(client, new_state_dict):
        return client.set_model_state(new_state_dict)

    # set update every client model
    set_model_tasks = [asyncio.create_task(call_client_set_model_async(client, new_state_dict)) for client in clients]
    await asyncio.gather(*set_model_tasks)

def generate_quantized_models(clients):
    for client in clients:
        client.generate_quantized_model()

if __name__ == '__main__':
    message_max_size = int(os.getenv("MESSAGE_MAX_SIZE"))
    client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
    clients = [DistributedClient(address, message_max_size) for address in client_addresses]

    op = ""
    generated_quantized = False
    while True:
        print("======== MENU ========")
        print("[1] - Train until accuracy reaches target")
        print("[2] - Quantize client model" + ((generated_quantized and ".") or (" (pendent)")))
        print("[3] - Show partial test dataset accuracy")
        print("[4] - Show full test dataset accuracy ")
        print("[0] - Sair")
        op = input()
        if op == "1":
            target_acc = input("Set target accuracy (def: 0.6): ")
            if target_acc == "":
                target_acc = 0.6
            else:
                target_acc = float(target_acc)

            i = 0
            while True:
                i += 1
                print(f"Training iteration {i}")
                # Training models
                asyncio.run(train_client_server_models(clients))
                # Aggregating client model parameters
                asyncio.run(aggregate_client_model_params(clients))
                # Estimate test dataset error
                generate_quantized_models(clients)
                full_acc = print_test_accuracy(clients, num_instances=client_batch_size, quantized=False)
                quant_acc = print_test_accuracy(clients, num_instances=client_batch_size, quantized=True)
                if full_acc >= target_acc:
                    print(f"Accuracy {full_acc} reached")
                    break
        elif op == '2':
            generate_quantized_models(clients)
            generated_quantized = True
        elif op == '22':
            generated_quantized = True
        elif op == '3':
            if generated_quantized:
                print_test_accuracy(clients, num_instances=client_batch_size)
                print_test_accuracy(clients, num_instances=client_batch_size, quantized=True)
            else:
                print("Must quantize client model")
        elif op == '4':
            if generated_quantized:
                print_test_accuracy(clients, num_instances=10000)
                print_test_accuracy(clients, num_instances=10000, quantized=True) #TODO: adjust num_clients per dataset
            else:
                print("Must quantize client model")
        elif op == '0':
            break
        else:
            print("Invalid option")


# # send client_model weights to the server for aggregation
# client_model_state_dict = client_model.state_dict()
# client_model_params = list(client_model_state_dict.values())

# # get aggregated weights from server
# new_state_dict = dict(zip(client_model_state_dict.keys(), aggregated_params))
# client_model.load_state_dict(new_state_dict)