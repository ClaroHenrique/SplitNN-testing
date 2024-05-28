import torch
from torch import nn
import torch.optim as optim

from split_learning.data_loader import load_cifar_10_fed_train_loaders
from split_learning.data_loader import load_cifar_10_test_loader
from split_learning.models.client_nn import ClientNN
from split_learning.models.server_nn import ServerNN

# global variables
n_clients = 4
learning_rate = 1e-3
batch_size = 32
epochs = 4
momentum = 0.9

# server initialization
server_model = ServerNN()
server_optimizer = optim.Adam(server_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# client initialization
client_models = [ClientNN() for _ in range(n_clients)]
client_optimizers = [optim.Adam(client_models[i].parameters(), lr=learning_rate) for i in range(n_clients)]
data_loaders = load_cifar_10_fed_train_loaders(n_clients, batch_size)

# test data
test_data_loader = load_cifar_10_test_loader(batch_size)
def test_loss(client_model, server_model, test_data_loader):
    loss_sum = 0.0
    correct = 0
    total = 0
    for (x, y) in test_data_loader:
        y_pred = server_model(client_model(x))
        loss_sum += loss_fn(y_pred, y)
        total += y.size(0)
        correct += sum(y_pred.argmax(dim=1) == y)
    print("--= TESTING =--")
    print("Loss:", loss_sum.item() / len(test_data_loader))
    print("Accuracy:", correct.item() / (len(test_data_loader) * batch_size))

test_loss(client_models[0], server_model, test_data_loader)

for ep in range(epochs):
    for client_id in range(n_clients):
        ### client side initialization ###
        client_model = client_models[client_id]
        data_loader = data_loaders[client_id]
        client_optimizer = client_optimizers[client_id]
        client_optimizer.zero_grad()
        
        ### server size initialization ###
        server_optimizer.zero_grad()
        
        ### client side ###
        x, y = next(iter(data_loader))
        client_output = client_model(x)
        
        ### server side ###
        # get input from client
        server_input = client_output.detach().requires_grad_(True)
        
        y_pred = server_model(server_input)
        loss = loss_fn(y_pred, y)
        print(loss)
        loss.backward()

        ### client side ###
        # get grad from server client #
        client_output_grad = server_input.grad.detach().requires_grad_(False)

        ### client side training ###
        client_output.backward(client_output_grad)
        client_optimizer.step()

        ### server side training ###
        server_optimizer.step()

        ### client side ###
        # send model to next client #
        next_client_model = client_models[(client_id+1) % n_clients]
        next_client_model.load_state_dict(client_model.state_dict())

        ### testing ###
        test_loss(client_model, server_model, test_data_loader)

test_loss(client_models[0], server_model, test_data_loader)































