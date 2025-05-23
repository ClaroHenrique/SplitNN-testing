from dataset.cifar10_non_iid import get_data_loaders as get_data_loaders_cifar10_non_iid
from dataset.cifar10_iid import get_data_loaders as get_data_loaders_cifar10_iid
from dataset.cifar10 import get_data_loaders as get_data_loaders_cifar10
from dataset.cifar100_non_iid import get_data_loaders as get_data_loaders_cifar100_non_iid
from dataset.cifar100_iid import get_data_loaders as get_data_loaders_cifar100_iid

def get_data_loaders(dataset, batch_size, client_id, num_clients, image_size):
    if dataset == "cifar10_non_iid":
        return get_data_loaders_cifar10_non_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "cifar10_iid":
        return get_data_loaders_cifar10_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "cifar10":
        return get_data_loaders_cifar10(batch_size, client_id, image_size)
    if dataset == "cifar100_non_iid":
        return get_data_loaders_cifar100_non_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "cifar100_iid":
        return get_data_loaders_cifar100_iid(batch_size, client_id, num_clients, image_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")    
