from dataset.cifar10_non_iid import get_data_loaders as get_data_loaders_cifar10

def get_data_loaders(dataset, batch_size, client_id, image_size):
    if dataset == "cifar10_non_iid":
        return get_data_loaders_cifar10(batch_size, client_id, image_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")    
