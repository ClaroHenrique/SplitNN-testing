from dotenv import load_dotenv
import os
import argparse
from utils.utils import *

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import pickle

import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2

import grpc
from concurrent import futures
import time
from model.vgg import ClientModel
from model.quantization import generate_quantized_model
from dataset.cifar10 import get_data_loaders #TODO use test dataset
from optimizer.adam import create_optimizer

load_dotenv()
auto_load_models = int(os.getenv("AUTO_LOAD_MODELS"))
print("auto_load_models", auto_load_models)

client_quantized_model = None
client_model, model_name = ClientModel()
if auto_load_models:
    load_model_if_exists(client_model, model_name)

image_size = list(map(int, os.getenv("IMAGE_SIZE").split(",")))
batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
print("LR", learning_rate)

train_data_loader, test_data_loader = get_data_loaders(batch_size=batch_size, client_id=1, image_size = image_size)
train_iter, test_iter = iter(train_data_loader), iter(test_data_loader)

def get_train_sample():
    global train_iter
    el = next(train_iter, None)
    if el == None:
        train_iter = iter(train_data_loader)
        el = next(train_iter, None)
    return el

def get_test_sample():
    global test_iter
    el = next(test_iter, None)
    if el == None:
        test_iter = iter(test_data_loader)
        el = next(test_iter, None)
    return el


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


def process_test_inference_query(model, batch_size):
    time_start = time.time()

    outputs = None
    with torch.no_grad():
        inputs, labels = get_test_sample()
        outputs = model(inputs)
    
    outputs, labels = pickle.dumps(outputs), pickle.dumps(labels)

    measure = {}
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
        tensor_IR, label, measure = process_test_inference_query(client_model, batch_size)
        pb2_tensor = pb2.Tensor(tensor=tensor_IR, label=label)
        pb2_measure = pb2.Measure(measure=measure)
        return pb2.TensorWithMeasure(tensor=pb2_tensor, measure=pb2_measure)
    
    def TestQuantizedInference(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id
        status = request.status

        print("context:", context)
        print("params:", {"batch_size": batch_size, "request_id": request_id, "status": status})
        tensor_IR, label, measure = process_test_inference_query(client_quantized_model, batch_size)
        pb2_tensor = pb2.Tensor(tensor=tensor_IR, label=label)
        pb2_measure = pb2.Measure(measure=measure)
        return pb2.TensorWithMeasure(tensor=pb2_tensor, measure=pb2_measure)

def serve(client_id):
    message_max_size = int(os.getenv("MESSAGE_MAX_SIZE"))
    client_address = os.getenv(f"CLIENT_ADDRESSES").split(",")[client_id]
    print("Client address", client_address)

    options=[
        ('grpc.max_send_message_length', message_max_size),
        ('grpc.max_receive_message_length', message_max_size),
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    pb2_grpc.add_DistributedClientServicer_to_server(DistributedClientService(), server)
    server.add_insecure_port(client_address)
    server.start()

    print("SFL Client started: waiting for queries")
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cliente SFL")
    parser.add_argument("--client_id", required=True, type=str, help="ID que identifica unicamente o cliente")
    args = parser.parse_args()

    client_id = int(args.client_id)
    serve(client_id)
