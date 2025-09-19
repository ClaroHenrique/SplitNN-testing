from dataset.cifar10_non_iid import get_data_loaders as get_data_loaders_cifar10_non_iid
from dataset.cifar10_iid import get_data_loaders as get_data_loaders_cifar10_iid
from dataset.cifar10 import get_data_loaders as get_data_loaders_cifar10
from dataset.cifar100_non_iid import get_data_loaders as get_data_loaders_cifar100_non_iid
from dataset.cifar100_iid import get_data_loaders as get_data_loaders_cifar100_iid
from dataset.xray_iid import get_data_loaders as get_data_loaders_xray_iid
from dataset.xray_non_iid import get_data_loaders as get_data_loaders_xray_non_iid

from dataset.imagenette import get_data_loaders as get_data_loaders_imagenette

def get_data_loaders(dataset, batch_size, client_id, num_clients, image_size):
    if dataset == "Cifar10_non_IID":
        return get_data_loaders_cifar10_non_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "Cifar10_IID":
        return get_data_loaders_cifar10_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "Cifar10":
        return get_data_loaders_cifar10(batch_size, client_id, image_size)
    if dataset == "Cifar100_non_IID":
        return get_data_loaders_cifar100_non_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "Cifar100_IID":
        return get_data_loaders_cifar100_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "ImageNette":
        return get_data_loaders_imagenette(batch_size, client_id, image_size)
    if dataset == "XRay_IID":
        return get_data_loaders_xray_iid(batch_size, client_id, num_clients, image_size)
    if dataset == "XRay_non_IID":
        return get_data_loaders_xray_non_iid(batch_size, client_id, num_clients, image_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")    


def get_num_classes(dataset):
    if dataset == "Cifar10_non_IID":
        return 10
    if dataset == "Cifar10_IID":
        return 10
    if dataset == "Cifar10":
        return 10
    if dataset == "Cifar100_non_IID":
        return 100
    if dataset == "Cifar100_IID":
        return 100
    if dataset == "ImageNette":
        return 10
    if dataset == "XRay_IID":
        return 2
    if dataset == "XRay_non_IID":
        return 2
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
