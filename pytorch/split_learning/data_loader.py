from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_cifar_10_fed_train_loaders(n_clients, batch_size):
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    dt_size = len(train_data) // n_clients
    clients_data = random_split(train_data, [dt_size]*n_clients)
    train_loaders = [DataLoader(dataset=clients_data[i], batch_size=64, shuffle=True) for i in range(n_clients)]
    return train_loaders


def load_cifar_10_test_loader(batch_size=64):
    data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return DataLoader(dataset=data, batch_size=batch_size)








