from dotenv import load_dotenv
import os
import argparse
from utils.utils import *
import itertools
import os
import sys
import tracemalloc

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
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
print("auto_load_models", auto_load_models)


model_name = os.getenv("MODEL")
dataset_name = os.getenv("DATASET")
image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
split_point = int(os.getenv("SPLIT_POINT"))
learning_rate = float(os.getenv("LEARNING_RATE"))
print("LR", learning_rate)


client_quantized_model = None
client_model = ClientModel(model_name, split_point=split_point)
if auto_load_models:
    load_model_if_exists(client_model, model_name, is_client=True, dataset_name=dataset_name)

train_data_loader, test_data_loader = get_data_loaders(dataset_name, batch_size=batch_size, client_id=client_id, image_size = image_size)
train_iter = itertools.cycle(train_data_loader)
test_iter = itertools.cycle(test_data_loader)

def get_train_sample():
    return next(train_iter)

def get_test_sample():
    return next(test_iter)


loss_fn = nn.CrossEntropyLoss()
optimizer, scheduler = create_optimizer(client_model.parameters(), learning_rate)
last_output = None
last_request_id = None

def process_forward_query(batch_size, request_id): #TODO use batch_size
    global last_output

    optimizer.zero_grad()
    inputs, labels = get_train_sample()
    outputs = client_model(inputs)

    last_output = outputs
    return outputs, labels

def process_backward_query(grad):
    optimizer.zero_grad()
    last_output.backward(grad)
    optimizer.step()
    #scheduler.step()
    debug_print(scheduler.get_last_lr())


def process_test_inference_query(model, batch_size, msg):
    print("process_test_inference_query -", msg)
    tracemalloc.start()
    mem_first, _ = tracemalloc.get_traced_memory()
    time_start = time.time()
    measure = {}
    outputs = None
    
    with torch.no_grad():
        inputs, labels = get_test_sample()
        outputs = model(inputs)
    measure["time-to-output"] = time.time() - time_start
    _, mem_peak = tracemalloc.get_traced_memory()

    outputs, labels = pickle.dumps(outputs), pickle.dumps(labels)
    #measure["mem-peak"] = mem_peak
    #measure["mem-first"] = mem_first
    measure["mem-usage"] = mem_peak - mem_first
    measure["time"] = time.time() - time_start
    measure["bandwidth"] = len(outputs) + len(labels)
    measure["model-size"] = size_of_model(model)
    measure = pickle.dumps(measure)

    return outputs, labels, measure

class DistributedClientService(pb2_grpc.DistributedClientServicer):
    def __init__(self, *args, **kwargs):
        pass

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
        grad = pickle.loads(request.tensor)
        request_id = request.request_id
        process_backward_query(grad)
        print("updating client model last_req", last_request_id, "current_req", request.request_id)
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
