from dotenv import load_dotenv
import os

import grpc
import distributed_learning_pb2_grpc as pb2_grpc
import distributed_learning_pb2 as pb2
import pickle

from model.mycnn import ServerModel
from optimizer.adam import create_optimizer
from utils.utils import debug_print

import torch
from torch import nn
from torch.nn import functional as F

load_dotenv()
server_model = ServerModel()

loss_fn = nn.CrossEntropyLoss()
learning_rate = float(os.getenv("LEARNING_RATE"))
optimizer = create_optimizer(server_model.parameters(), learning_rate)

def save_model():
    # initialize model #
    client_model.save("./model_state")
    if os.path.exists():
        client_model = torch.load("./model_state/client")

def load_model():
    global client_model
    if os.path.exists():
        client_model = torch.load("./model_state/client")

def server_forward(tensor_IR, labels):
    # update server model, returns grad of the input 
    # used to continue the backpropagation in client_model
    tensor_IR.requires_grad = True
    optimizer.zero_grad()
    outputs = server_model(tensor_IR)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    debug_print("updating server model")
    print(torch.unique(labels, return_counts=True))
    return tensor_IR.grad.detach()

def server_test_inference(tensor_IR, labels):
    # update server model, returns grad of the input
    # used to continue the backpropagation in client_model
    correct = 0
    total = 0
    loss = 0
    print(torch.unique(labels, return_counts=True))
    with torch.no_grad():
        tensor_IR.requires_grad = False
        outputs = server_model(tensor_IR)
        loss = loss_fn(outputs, labels)

        pred_class = torch.argmax(outputs, dim=1)
        correct += torch.eq(labels, pred_class).sum().item()
        total += len(labels)
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
        debug_print("IR", tensor_IR, labels)
        return tensor_IR, labels

    def backward(self, grad): # TODO: implement request_id
        debug_print("GRAD", grad)
        grad = pickle.dumps(grad)
        tensor = pb2.Tensor(tensor=grad, label=None)
        return self.stub.Backward(tensor)
    
    def get_model_state(self):
        query = pb2.Empty()
        response = self.stub.GetModelState(query)
        model_state = pickle.loads(response.state)
        return model_state

    def set_model_state(self, model_state): # TODO: implement request_id
        model_state = pickle.dumps(model_state)
        model_state = pb2.ModelState(model_state=model_state)
        return self.stub.SetModelState(model_state)
    
    def test_inference(self, batch_size): # TODO: ForwardTest ao inves de test_inference
        query = pb2.Query(batch_size=batch_size, request_id=-1, status=0)
        response = self.stub.TestInference(query)
        tensor_IR = pickle.loads(response.tensor)
        labels = pickle.loads(response.label)
        return tensor_IR, labels

def train_client_server_models(clients):
    clients_IRs = []
    clients_labels = []

    # Collect IR and Labels from the client
    for client in clients:
        tensor_IR, labels = client.forward(batch_size=32, request_id=1)
        clients_IRs.append(tensor_IR)
        clients_labels.append(labels)
    
    # Concat IR and labels to feed and train server model
    concat_IRs = torch.concatenate(clients_IRs).detach()
    concat_labels = torch.concatenate(clients_labels).detach()
    concat_IRs_grad = server_forward(concat_IRs, concat_labels)

    debug_print(concat_IRs.sum())

    # Send IRs gradients back to the clients (train client model)
    clients_IRs_grad = concat_IRs_grad.split(client_batch_size)
    for client, client_IR_grad in zip(clients, clients_IRs_grad):
        client.backward(grad=client_IR_grad)

def print_test_accuracy(clients):
    correct = 0
    total = 0
    loss = 0
    for client in clients:
        tensor_IR, labels = client.test_inference(batch_size=32) #TODO fix batchsize
        c_correct, c_total, c_loss = server_test_inference(tensor_IR, labels)
        correct += c_correct
        total += c_total
        loss += c_loss.item() / len(clients)
    print(f"Accuracy: {correct} / {total} = {correct / total}")
    print(f"Loss: ", loss)
    print()

def aggregate_client_model_params(clients):
    client_model_weights = []
    model_state_keys = None

    for client in clients:
        client_model_state = client.get_model_state()
        client_model_weights.append(list(client_model_state.values()))
        if model_state_keys == None:
            model_state_keys = client_model_state.keys()
    
    aggregated_params = []
    n_layers = len(model_state_keys)
    for l in range(n_layers):
        layer_params = []
        for client_w in client_model_weights:
            layer_params.append(client_w[l])
        layer_mean = torch.stack(layer_params, dim=0).mean(dim=0)
        aggregated_params.append(layer_mean)
    
    # get aggregated weights from server
    new_state_dict = dict(zip(model_state_keys, aggregated_params))
    return new_state_dict


if __name__ == '__main__':
    client_batch_size = int(os.getenv("CLIENT_BATCH_SIZE"))
    message_max_size = int(os.getenv("MESSAGE_MAX_SIZE"))
    client_addresses = os.getenv("CLIENT_ADDRESSES").split(",")
    clients = [DistributedClient(address, message_max_size) for address in client_addresses]

    num_iterations = 1000

    for i in range(num_iterations):
        if(i % 20 == 0):
            print_test_accuracy(clients)
        train_client_server_models(clients)
        aggregate_client_model_params(clients)
        # Estimate test dataset error

    print_test_accuracy(clients)



# # send client_model weights to the server for aggregation
# client_model_state_dict = client_model.state_dict()
# client_model_params = list(client_model_state_dict.values())

# # get aggregated weights from server
# new_state_dict = dict(zip(client_model_state_dict.keys(), aggregated_params))
# client_model.load_state_dict(new_state_dict)