import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mpi4py import MPI

from client_nn import ClientNN
from server_nn import ServerNN

# constants #

learning_rate = 1e-3
batch_size = 64
epochs = 10
loss_fn = nn.CrossEntropyLoss()

# import data #

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# create models #

client_model = ClientNN()
server_model = ServerNN()

def train_loop(dataloader, client_model, server_model, loss_fn, client_optimizer, server_optimizer):
    size = len(dataloader.dataset)
    client_model.train()
    server_model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        #   client
        client_output = client_model(X)
        server_input = client_output.detach().clone().requires_grad_(True)
        #   server
        pred = server_model(server_input)
        loss = loss_fn(pred, y)

        # Compute gradients with backpropagation
        #   server
        loss.backward()
        cut_layer_loss = server_input.grad.detach().clone()
        #   client
        client_output.backward(cut_layer_loss)

        # Update
        # client
        client_optimizer.step()
        client_optimizer.zero_grad()

        # server
        server_optimizer.step()
        server_optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, client_model, server_model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    client_model.eval()
    server_model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = server_model(client_model(X))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




## run training ##

client_optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate)
server_optimizer = torch.optim.SGD(server_model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, client_model, server_model, loss_fn, client_optimizer, server_optimizer)
    test_loop(test_dataloader, client_model, server_model, loss_fn)
print("Done!")


