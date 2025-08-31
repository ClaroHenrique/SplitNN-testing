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

def calculate_mean_std():
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False)
    transform = transforms.Compose([transforms.ToTensor()])
    dataiter = iter(trainloader)
    images, _ = next(dataiter)
    mean = images.mean((0, 2, 3))
    std = images.std((0,2,3))
    return mean, std

def get_data_loaders(batch_size, client_id, num_clients, image_size):

    normalize_mean, normalize_std = calculate_mean_std()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.Resize(image_size),
    ]) # 32x32

    transform_calib = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.Resize(image_size),
    ]) 

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.Resize(image_size),
    ]) 
    
    train_dataset = Cifar10_Train_IID_Dataset(client_id=client_id, num_clients=num_clients, transform=transform_train)
    calib_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_calib) # TODO: (calib from partitioned training without crop) Cifar10_Train_IID_Dataset(client_id=client_id, num_clients=num_clients, transform=transform_calib)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    
    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    return train_dataloader, calib_dataset, test_dataloader

