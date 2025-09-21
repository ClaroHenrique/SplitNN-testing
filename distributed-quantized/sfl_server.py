from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.abspath("proto"))
import grpc
import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2
import pickletools

from optimizer.sgd import create_optimizer
from utils.utils import *

import torch
from torch import nn
from torch.nn import functional as F

load_dotenv()
model_name = os.getenv("MODEL")
quantization_type = os.getenv("QUANTIZATION_TYPE")
dataset_name = os.getenv("DATASET")
image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
experiment_batches = int(os.getenv("EXPERIMENT_BATCHES"))
learning_rate = float(os.getenv("LEARNING_RATE"))
client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
iterations_per_epoch = int(os.getenv("ITERATIONS_PER_EPOCH"))
split_point = int(os.getenv("SPLIT_POINT"))
auto_save_models = int(os.getenv("AUTO_SAVE_MODELS"))
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
num_clients = len(os.getenv("CLIENT_ADDRESSES").split(","))
device_server = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_client = "cpu" 
results_inference_file_name = "./experiments/results_inference.csv"


loss_fn = nn.CrossEntropyLoss()
global_request_id = 1

##### CUSTOMIZE MODEL AND DATA #####
from model.models import ClientModel
from model.models import ServerModel
from dataset.datasets import get_num_classes
####################################

num_classes = get_num_classes(dataset_name)
server_model = ServerModel(model_name, num_classes=num_classes, quantization_type=quantization_type, split_point=split_point, device=device_server, input_shape=None)
optimizer, scheduler = create_optimizer(server_model.parameters(), learning_rate)

def server_forward(tensor_IR, labels):
    # update server model, returns grad of the input 
    # used to continue the backpropagation in client_model
    tensor_IR, labels = tensor_IR.to(device_server), labels.to(device_server)
    tensor_IR.requires_grad = True
    optimizer.zero_grad()
    outputs = server_model(tensor_IR)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    debug_print("updating server model")
    debug_print(torch.unique(labels, return_counts=True))
    debug_print("LR", scheduler.get_last_lr())
    return tensor_IR.grad.detach().to('cpu')

def server_test_inference(tensor_IR, labels):
    # update server model, returns grad of the input
    # used to continue the backpropagation in client_model
    correct = 0
    total = 0
    loss = 0
    debug_print(torch.unique(labels, return_counts=True))
    with torch.no_grad():
        tensor_IR = tensor_IR.to(device_server)
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

    def initialize(self, params_dict):
        query = pb2.Dictionary(dictionary= pickle.dumps(params_dict))
        return self.stub.Initialize(query)

    def forward(self, batch_size, request_id):
        query = pb2.Query(batch_size=batch_size, request_id=request_id, status=0)
        response = self.stub.Forward(query)
        tensor_IR = pickle.loads(response.tensor)
        labels = pickle.loads(response.label)
        request_id = response.request_id
        debug_print("IR", tensor_IR, labels)
        return tensor_IR, labels, request_id

    def backward(self, grad, lr, request_id): # TODO: implement request_id
        debug_print("GRAD", grad)
        grad = pickle.dumps(grad)
        tensor = pb2.Tensor(tensor=grad, label=None, request_id=request_id)
        tensor_with_lr = pb2.TensorWithLR(tensor=tensor, learning_rate=lr)
        return self.stub.Backward(tensor_with_lr)
    
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
    lr = scheduler.get_last_lr()[0]
    debug_print("Training models")

    async def call_client_forward_async(client, batch_size, request_id):
        return client.forward(batch_size=batch_size, request_id=request_id)

    print("Sending forward requests to clients")
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
    
    debug_print("Waiting for client forward tasks")
    for task in forward_tasks:
        tensor_IR, labels, request_id = await task
        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)
        clients_request_ids.append(request_id)

    debug_print("Concat IRs and labels")
    # Concat IR and labels to feed and train server model
    concat_IRs = torch.concatenate(clients_IRs).detach()
    concat_labels = torch.concatenate(clients_labels).detach()
    debug_print("Feed server model with IRs")
    concat_IRs_grad = server_forward(concat_IRs, concat_labels)
    debug_print(concat_IRs.sum())

    # Send IRs gradients back to the clients (train client model)
    async def call_client_backward_async(client, grad, lr, request_id):
        return client.backward(grad=grad, lr=lr, request_id=req_id)
    debug_print("Sending backward requests to clients")
    backward_tasks = []
    clients_IRs_grad = concat_IRs_grad.split(client_batch_size)
    for client, client_IR_grad, req_id in zip(clients, clients_IRs_grad, clients_request_ids):
        backward_tasks.append(
            asyncio.create_task(
                call_client_backward_async(client, client_IR_grad, lr, request_id=req_id))
        )
    debug_print("Waiting for client backward tasks")
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


async def initialize_clients(clients, params_dict=None):
    async def call_client_initialize_async(client):
        return client.initialize(params_dict)
    # initialize every client
    init_tasks = [asyncio.create_task(call_client_initialize_async(client)) for client in clients]
    await asyncio.gather(*init_tasks)

async def set_client_model_params(clients, model_state):
    async def call_client_set_model_async(client, new_state_dict):
        return client.set_model_state(new_state_dict)

    # set update every client model
    set_model_tasks = [asyncio.create_task(call_client_set_model_async(client, model_state)) for client in clients]
    await asyncio.gather(*set_model_tasks)

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
    await set_client_model_params(clients, new_state_dict)

def generate_quantized_models(clients):
    for client in clients:
        client.generate_quantized_model()


def collect_client_measures(clients, quantized=False):
    all_measures = []
    iterations = 0

    while True:
        client = clients[0]
        if quantized:
            tensor_IR, labels, measure = client.test_quantized_inference(batch_size=client_batch_size) #TODO fix batchsize
        else:
            tensor_IR, labels, measure = client.test_inference(batch_size=client_batch_size) #TODO fix batchsize
        iterations += 1
        all_measures.append(measure)
        if iterations >= experiment_batches:
            break
        
    if quantized:
        print("== Quantized Metrics ==")
    else:
        print("== Full Precision Metrics ==")
    agg_measures = aggregate_measures_mean(all_measures)
    print(f"Measure mean: ", agg_measures)
    print()
    return agg_measures, all_measures

def run_all_experiment_configs_in_client():
    all_configs = []
    with open("inference_configs.csv", "r") as file:
        header = file.readline()
        for line in file:
            values = line.split(",")
            print(values)
            all_configs.append({
                "MODEL_NAME": values[0].strip(),
                "QUANTIZATION_TYPE": values[1].strip(),
                "DATASET_NAME": values[2].strip(),
                "SPLIT_POINT": int(values[3]),
                "NUM_CLIENTS": int(values[4]),
                "IMAGE_SIZE": list(map(int, values[5].strip().split(" "))),
                "CLIENT_BATCH_SIZE": int(values[6]),
            })
    print("All configs: ", all_configs)

    for config in all_configs:
        print("Running experiment with config:", config)
        params_dict = {
            "model_name": config["MODEL_NAME"],
            "quantization_type": config["QUANTIZATION_TYPE"],
            "dataset_name": config["DATASET_NAME"],
            "image_size": config["IMAGE_SIZE"],
            "batch_size": config["CLIENT_BATCH_SIZE"],
            "split_point": config["SPLIT_POINT"],
            "learning_rate": learning_rate,
            "num_clients": config["NUM_CLIENTS"],
            "num_classes": num_classes,
        }
        # Initialize clients

        asyncio.run(initialize_clients(clients, params_dict))
        generate_quantized_models(clients)
        _, measures_full = collect_client_measures(clients, quantized=False)
        _, measures_quantized = collect_client_measures(clients, quantized=True)

        save_inference_measures_in_file(results_inference_file_name, generate_run_id(), model_name, "full", split_point, dataset_name, client_batch_size, measures_full)
        save_inference_measures_in_file(results_inference_file_name, generate_run_id(), model_name, quantization_type, split_point, dataset_name, client_batch_size, measures_quantized)

        print(f"Results for {config['MODEL_NAME']} with {config['QUANTIZATION_TYPE']} quantization on {config['DATASET_NAME']}:")
        print(f"Full Precision: {measures_full}")
        print(f"Quantized: {measures_quantized}")
        # TODO: save metrics in file


if __name__ == '__main__':
    message_max_size = int(os.getenv("MESSAGE_MAX_SIZE"))
    client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
    clients = [DistributedClient(address, message_max_size) for address in client_addresses]

    client_params_dict = {
        "model_name": model_name,
        "quantization_type": quantization_type,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "batch_size": client_batch_size,
        "split_point": split_point,
        "learning_rate": learning_rate,
        "num_clients": num_clients,
        "num_classes": num_classes,
    }
    asyncio.run(initialize_clients(clients, client_params_dict))
    # Initialize server and clients model params
    if auto_load_models:
        # server model
        load_model_if_exists(server_model, model_name, quantization_type, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
        print("Server model loaded")
        # client model
        client_model = ClientModel(model_name, num_classes=num_classes, quantization_type=quantization_type, split_point=split_point, device= device_client, input_shape=image_size)
        load_model_if_exists(server_model, model_name, quantization_type, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
        client_model_state_dict = client_model.state_dict()
        print("Client parameter SUM:", model_parameters_sum(client_model))

        asyncio.run(set_client_model_params(clients, client_model_state_dict))
        print("Client models loaded")

    op = ""
    generated_quantized = False
    while True:
        print("======== MENU ========")
        print("[1] - Train until accuracy reaches target")
        print("[2] - Quantize client model" + ((generated_quantized and ".") or (" (pendent)")))
        print("[3] - Show partial test dataset accuracy")
        print("[4] - Show full test dataset accuracy ")
        print("[5] - Run all experiment configs")
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
                
                if auto_save_models and i % iterations_per_epoch == 0:
                    scheduler.step()
                    print("Current learning rate", scheduler.get_last_lr())
                    if auto_save_models:
                        save_state_dict(server_model.state_dict(), model_name, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
                        save_state_dict(clients[0].get_model_state(), model_name, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
                    full_acc = print_test_accuracy(clients, num_instances=client_batch_size, quantized=False)
                    quant_acc = print_test_accuracy(clients, num_instances=client_batch_size, quantized=True)
                    stop_criteria = full_acc >= target_acc
                    if stop_criteria:
                        print(f"Accuracy {full_acc} reached")
                        break
        elif op == '2':
            generate_quantized_models(clients)
            generated_quantized = True
        elif op == '22':
            generated_quantized = True
        elif op == '3':
            if generated_quantized:
                print_test_accuracy(clients, num_instances=client_batch_size * experiment_batches)
                print_test_accuracy(clients, num_instances=client_batch_size * experiment_batches, quantized=True)
            else:
                print("Must quantize client model")
        elif op == '4':
            if generated_quantized:
                print_test_accuracy(clients, num_instances=10000) #TODO: adjust num_instances per dataset
                print_test_accuracy(clients, num_instances=10000, quantized=True)
            else:
                print("Must quantize client model")
        elif op == '5':
            run_all_experiment_configs_in_client()
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