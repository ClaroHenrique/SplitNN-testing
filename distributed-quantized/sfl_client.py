from dotenv import load_dotenv
import os
import argparse
from utils.utils import *
import itertools
import os
import sys
import tracemalloc
import torch.profiler
import gc
from memory_profiler import memory_usage



sys.path.append(os.path.abspath("proto"))
import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import pickle


import grpc
from concurrent import futures
import time
##### CUSTOMIZE MODEL AND DATA #####
from model.models import ClientModel
from model.models import ServerModel
from dataset.datasets import get_data_loaders #TODO use test dataset ??
####################################
from model.quantization import generate_quantized_model
from optimizer.adam import create_optimizer

parser = argparse.ArgumentParser(description="Cliente SFL")
parser.add_argument("--client_id", required=True, type=str, help="ID que identifica unicamente o cliente")
args = parser.parse_args()
client_id = int(args.client_id)

load_dotenv()

model_name = None # os.getenv("MODEL")
dataset_name = None # os.getenv("DATASET")
image_size = None # list(map(int, os.getenv("IMAGE_SIZE").split(",")))
batch_size = None # int(os.getenv("CLIENT_BATCH_SIZE"))
split_point = None # int(os.getenv("SPLIT_POINT"))
learning_rate = None # float(os.getenv("LEARNING_RATE"))
num_clients = None # len(os.getenv("CLIENT_ADDRESSES").split(","))

client_quantized_model = None
client_model = None

train_data_loader, test_data_loader = None, None
train_iter = None
test_iter = None

optimizer = None

def get_train_sample():
    # TODO: REMOVE intertools.cycle. it does not shuffle the data between epochs
    return next(train_iter)

def get_test_sample():
    return next(test_iter)

def inicialize(params_dict):
    print("inicializate params_dict:", params_dict)
    global model_name, dataset_name, image_size, batch_size, split_point, learning_rate, num_clients
    global client_model, client_quantized_model, train_data_loader, test_data_loader, optimizer
    global train_iter, test_iter

    model_name = params_dict["model_name"]
    dataset_name = params_dict["dataset_name"]
    image_size = params_dict["image_size"]
    batch_size = params_dict["batch_size"]
    split_point = params_dict["split_point"]
    learning_rate = params_dict["learning_rate"] # not really used (changes every backward)
    num_clients = params_dict["num_clients"]
    
    client_quantized_model = None
    client_model = ClientModel(model_name, split_point=split_point)

    train_data_loader, test_data_loader = get_data_loaders(dataset_name, batch_size=batch_size, client_id=client_id, num_clients=num_clients, image_size = image_size)
    train_iter = itertools.cycle(train_data_loader) # TODO: REMOVER itertools.cycle COLOCAR FUNCAO
    test_iter = itertools.cycle(test_data_loader)

    optimizer, _ = create_optimizer(client_model.parameters(), learning_rate)
    last_output = None
    last_request_id = None

def process_forward_query(batch_size, request_id): #TODO use batch_size
    global last_output

    optimizer.zero_grad()
    inputs, labels = get_train_sample()
    outputs = client_model(inputs)

    last_output = outputs
    return outputs, labels

def process_backward_query(grad, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    optimizer.zero_grad()
    last_output.backward(grad)
    optimizer.step()


def process_test_inference_query(model, batch_size, msg):
    print("process_test_inference_query -", msg)
    measure = {}
    outputs = None
    inputs, labels = get_test_sample()
    # warmup
    model.eval()
    model(inputs)
    gc.collect()

    # measure memory usage
    init_mem = memory_usage(-1, interval=.01, timeout=1, max_usage=True)
    mem_usage = memory_usage((model, (inputs,)), interval=0.01,
                             include_children=True,
                             max_usage=True,
                             #timestamps=True
                             )
    mem_first = init_mem
    mem_peak = mem_usage

    # measure time
    time_start = time.time()
    outputs = model(inputs)
    time_end = time.time()

    outputs, labels = pickle.dumps(outputs), pickle.dumps(labels)
    #measure["mem-peak-mb"] = mem_peak
    #measure["mem-first-mb"] = mem_first
    measure["mem-usage-mb"] = mem_peak - mem_first
    measure["time"] = time_end - time_start
    measure["bandwidth"] = len(outputs) + len(labels)
    measure["model-size"] = size_of_model(model)
    measure = pickle.dumps(measure)

    return outputs, labels, measure

class DistributedClientService(pb2_grpc.DistributedClientServicer):
    def __init__(self, *args, **kwargs):
        pass

    def Initialize(self, request, context):
        print("Initialize context:", context)
        params_dict = pickle.loads(request.dictionary)
        inicialize(params_dict)
        return pb2.Empty()

    def Forward(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id
        global last_request_id
        last_request_id = request_id

        tensor_IR, label = process_forward_query(batch_size, request_id)
        tensor_IR, label = pickle.dumps(tensor_IR), pickle.dumps(label)
        return pb2.Tensor(tensor=tensor_IR, label=label, request_id=request_id)

    def Backward(self, request, context):
        print("Backward context:", context)
        grad = pickle.loads(request.tensor.tensor)
        new_lr = request.learning_rate
        request_id = request.tensor.request_id

        process_backward_query(grad, new_lr)
        print("updating client model last_req", last_request_id, "current_req", request_id)
        return pb2.Query(batch_size=-1, request_id=request_id, status=1)

    def GetModelState(self, request, context):
        model_state = client_model.state_dict()
        model_state = pickle.dumps(model_state)
        return pb2.ModelState(state=model_state)

    def SetModelState(self, request, context):
        model_state = request.state # TODO fix?
        model_state = pickle.loads(model_state)
        client_model.load_state_dict(model_state)
        print("PARAMETER SUM:", model_parameters_sum(client_model))
        return pb2.Empty()

    def GenerateQuantizedModel(self, request, context):
        global client_quantized_model
        #TODO: generate decent calib data loader    
        client_quantized_model = generate_quantized_model(client_model, calib_dataloader=train_data_loader)
        return pb2.Empty()

    def TestInference(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id
        status = request.status
        print("context:", context)
        print("params:", {"batch_size": batch_size, "request_id": request_id, "status": status})
        tensor_IR, label, measure = process_test_inference_query(client_model, batch_size, "fully")
        pb2_tensor = pb2.Tensor(tensor=tensor_IR, label=label)
        pb2_measure = pb2.Measure(measure=measure)
        return pb2.TensorWithMeasure(tensor=pb2_tensor, measure=pb2_measure)

    def TestQuantizedInference(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id
        status = request.status

        print("context:", context)
        print("params:", {"batch_size": batch_size, "request_id": request_id, "status": status})
        tensor_IR, label, measure = process_test_inference_query(client_quantized_model, batch_size, "quantized")
        pb2_tensor = pb2.Tensor(tensor=tensor_IR, label=label)
        pb2_measure = pb2.Measure(measure=measure)
        return pb2.TensorWithMeasure(tensor=pb2_tensor, measure=pb2_measure)

def serve(client_id):
    message_max_size = int(os.getenv("MESSAGE_MAX_SIZE"))
    client_address = os.getenv(f"CLIENT_LOCALHOST").split(",")[client_id]
    client_port = client_address.split(":")[1]
    print("Client address", client_address)

    options=[
        ('grpc.max_send_message_length', message_max_size),
        ('grpc.max_receive_message_length', message_max_size),
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    pb2_grpc.add_DistributedClientServicer_to_server(DistributedClientService(), server)
    server.add_insecure_port(f"[::]:{client_port}")
    server.start()

    print("SFL Client started: waiting for queries")
    server.wait_for_termination()

if __name__ == '__main__':
    serve(client_id)
