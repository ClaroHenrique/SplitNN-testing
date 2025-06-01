import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms
from datasets import Dataset as DT

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import numpy as np

class Cifar10_Train_IID_Dataset(Dataset):
    def __init__(self, client_id, num_clients, transform=None):
        # self.client_id = client_id
        # self.num_clients = num_clients
        # self.random_seed = random_seed
        self.transform = transform
        partitioner = IidPartitioner(
            num_partitions = num_clients,
        )
        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        dt_list = [{'img': f, 'label': l} for f,l in train_dataset]
        partitioner.dataset = DT.from_list(dt_list)
        
        self.partition = partitioner.load_partition(partition_id=client_id)

    def __getitem__(self, index):
        batch = self.partition[index]
        image, label = batch['img'], batch['label']
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.partition)



def get_data_loaders(batch_size, client_id, num_clients, image_size):
    # TODO: resize images
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(image_size), # Resize to 32x32
    ]) # 32x32

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(image_size), # Resize to 32x32
    ]) 
    
    #TODO: Implement shuffle in IID partitioner
    train_dataset = Cifar10_Train_IID_Dataset(client_id=client_id, num_clients=num_clients, transform=transform_train, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) #TODO: arbitrary test bath_size

    return train_dataloader, test_dataloader

