from dotenv import load_dotenv
import os
import argparse

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import pickle

import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2

import grpc
from concurrent import futures
import time
from model.mycnn import ClientModel
from dataset.cifar10 import get_data_loaders

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


train_data_loader, test_data_loader = get_data_loaders(batch_size=32, client_id=1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
last_outputs = None


def process_forward_query(batch_size, request_id):
    global last_outputs

    optimizer.zero_grad()
    inputs, labels = next(iter(train_data_loader))
    outputs = client_model(inputs)

    last_outputs = outputs
    return outputs, labels

def process_backward_query(grad):
    last_outputs.backward(grad)
    optimizer.step()
    optimizer.zero_grad()
    print("updating client model")

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
        status = request.status

        print("context:", context)
        print("params:", {"batch_size": batch_size, "request_id": request_id, "status": status})
        tensor_IR, label = process_forward_query(batch_size, request_id)
        tensor_IR, label = pickle.dumps(tensor_IR), pickle.dumps(label)
        return pb2.Tensor(tensor=tensor_IR, label=label)

    def Backward(self, request, context):
        tensor = request.tensor
        grad = pickle.loads(tensor)
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
    client_address = os.getenv(f"CLIENT_ADDRESSES").split(",")[client_id]
    print("Client address", client_address)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
