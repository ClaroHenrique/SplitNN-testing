import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mpi4py import MPI

from client_nn import ClientNN
from server_nn import ServerNN

# MPI initialization
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
mpi_num_clients = comm.Get_size() - 1
is_server = (mpi_rank == 0)
is_client = (mpi_rank != 0)

# constants #
learning_rate = 1e-3 
batch_size = 32
epochs = 10
loss_fn = nn.CrossEntropyLoss()
N_train = (len(datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor())) // batch_size) * batch_size
N_test = (len(datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor())) // batch_size) * batch_size

# import data #

if is_client:
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
    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)


# create models #

if is_client:
    client_model = ClientNN()
    client_optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate)
else:
    server_model = ServerNN()
    # TODO: check if it is necessary to divide the learning rate  
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=learning_rate)


for ep in range(epochs):
    if is_server:
        print("epoch:", ep+1, "/", epochs)
    ## client train loop ##
    if is_client:
        client_model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            client_output = client_model(X)
            client_output_cp = client_output.detach().clone()
            # send output to server
            comm.gather(client_output_cp, root=0)
            # send y to server
            comm.gather(y, root=0)

            # receive grad from cut layer
            cut_layer_loss = comm.scatter(None, root=0)
            client_output.backward(cut_layer_loss)

            client_optimizer.step()
            client_optimizer.zero_grad()

        # send client_model weights to the server for aggregation
        client_model_state_dict = client_model.state_dict()
        client_model_params = list(client_model_state_dict.values())
        comm.gather(client_model_params, root=0)
        
        # get aggregated weights from server
        aggregated_params = comm.bcast(None, root=0)
        new_state_dict = dict(zip(client_model_state_dict.keys(), aggregated_params))
        client_model.load_state_dict(new_state_dict)

    else: # server train loop
        server_model.train()
        for i in range(N_train//batch_size):
            #receive client ouput
            client_outputs = comm.gather(None, root=0)
            #receive y
            y = comm.gather(None, root=0)
            y = torch.cat(y[1:], dim=0)

            server_input = torch.cat(client_outputs[1:], dim=0).requires_grad_(True)
            pred = server_model(server_input)
            loss = loss_fn(pred, y)
            loss.backward()
            cut_layer_loss = server_input.grad.detach().clone()

            loss_partition = [cut_layer_loss[i*batch_size: (i+1)*batch_size, :] for i in range(mpi_num_clients)]
            loss_partition = [None] + loss_partition

            # receive grad from cut layer
            comm.scatter(loss_partition, root=0)
        
            server_optimizer.step()
            server_optimizer.zero_grad()
    
        # agreggate client models
        all_client_model_params = comm.gather(None, root=0)[1:]
        n_layers = len(all_client_model_params[1])
        
        aggregated_params = []
        for l in range(n_layers):
            layer_params = []
            for c in range(mpi_num_clients):
                layer_params.append(all_client_model_params[c][l])
            p = torch.stack(layer_params, dim=0).mean(dim=0)
            aggregated_params.append(p)
        comm.bcast(aggregated_params, root=0)
        

    ## def test loop ##
    if is_client:
        client_model.eval()
        for batch, (X, y) in enumerate(test_dataloader):
            client_output = client_model(X)
            client_output_cp = client_output.detach().clone()

            # send client output
            comm.gather(client_output_cp, root=0)
            # send y
            comm.gather(y, root=0)
    else:
        server_model.eval()
        test_loss, correct, count = 0, 0, 0
        for i in range(N_test//batch_size):
            # receive client ouput
            client_outputs = comm.gather(None, root=0)
            server_input = torch.cat(client_outputs[1:], dim=0).requires_grad_(False)
            # receive y
            y = comm.gather(None, root=0)
            y = torch.cat(y[1:], dim=0)

            pred = server_model(server_input)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            count += len(y)
        #print("accuracy", correct, count, N_test, y.shape)
        accuracy = correct/count
        test_loss = test_loss / (N_test//batch_size)
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")



