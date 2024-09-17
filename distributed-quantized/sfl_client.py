from dotenv import load_dotenv
import os
import argparse

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import pickle

import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2

import grpc
from concurrent import futures
import time
from model.mycnn import ClientModel
from dataset.cifar10 import get_data_loaders #TODO use test dataset
from optimizer.adam import create_optimizer

load_dotenv()
client_model = ClientModel()

def save_model():
    # initialize model #
    client_model.save("./model_state")
    if os.path.exists():
        client_model = torch.load("./model_state/client")

def load_model():
    global client_model
    if os.path.exists():
        client_model = torch.load("./model_state/client")

batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
print("LR", learning_rate)

train_data_loader, test_data_loader = get_data_loaders(batch_size=batch_size, client_id=1)
train_iter, test_iter = iter(train_data_loader), iter(test_data_loader)

def get_train_sample():
    global train_iter
    el = next(train_iter, None)
    if el == None:
        train_iter = iter(train_data_loader)
        el = next(train_iter, None)
    return el

loss_fn = nn.CrossEntropyLoss()
optimizer = create_optimizer(client_model.parameters(), learning_rate)
last_outputs = None

def process_forward_query(batch_size, request_id):
    global last_outputs

    optimizer.zero_grad()
    inputs, labels = get_train_sample()
    outputs = client_model(inputs)

    last_outputs = outputs
    return outputs, labels

def process_backward_query(grad):
    last_outputs.backward(grad)
    #optimizer.step()
    #optimizer.zero_grad()
    print("not updating client model")

def process_test_inference_query(batch_size):
    outputs = None
    with torch.no_grad():
        inputs, labels = next(iter(test_data_loader))
        outputs = client_model(inputs)
    return outputs, labels

class DistributedClientService(pb2_grpc.DistributedClientServicer):
    def __init__(self, *args, **kwargs):
        pass

    def Forward(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id

        tensor_IR, label = process_forward_query(batch_size, request_id)
        tensor_IR, label = pickle.dumps(tensor_IR), pickle.dumps(label)
        return pb2.Tensor(tensor=tensor_IR, label=label)

    def Backward(self, request, context):
        print("Backward context:", context)
        grad = pickle.loads(request.tensor)
        process_backward_query(grad)
        return pb2.Query(batch_size=-1, request_id=-1, status=1)

    def GetModelState(self, request, context):
        model_state = client_model.state_dict()
        model_state = pickle.dumps(model_state)
        return pb2.ModelState(state=model_state)

    def SetModelState(self, request, context):
        tensor = request.state
        model_state = pickle.loads(model_state)
        client_model.load_state_dict(model_state)
        return pb2.Empty()
    
    def TestInference(self, request, context):
        batch_size = request.batch_size
        request_id = request.request_id
        status = request.status

        print("context:", context)
        print("params:", {"batch_size": batch_size, "request_id": request_id, "status": status})
        tensor_IR, label = process_test_inference_query(batch_size)
        tensor_IR, label = pickle.dumps(tensor_IR), pickle.dumps(label)
        return pb2.Tensor(tensor=tensor_IR, label=label)

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
