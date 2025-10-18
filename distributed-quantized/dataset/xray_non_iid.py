import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import kagglehub
import os

# Download latest version

import torchvision
import torchvision.transforms as transforms
from datasets import Dataset as DT
import datasets

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import DirichletPartitioner

import numpy as np

class XRay_Train_non_IID_Dataset(Dataset):
    
    def __init__(self, client_id, num_clients, dirichlet_alpha = 0.1, random_state=42, transform=None, data_dir=None):
        # self.client_id = client_id
        # self.num_clients = num_clients
        # self.random_seed = random_seed
        self.transform = transform
        partitioner = DirichletPartitioner(
            num_partitions = num_clients,
            partition_by = "label",
            alpha = dirichlet_alpha,
            min_partition_size = 128,
            self_balancing = False,
            shuffle = True,
            seed = random_state,
        )

        self.train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'chest_xray', 'train'), transform=transform)
        #train_dataset = datasets.load_dataset("imagefolder", data_dir=os.path.join(data_dir, 'chest_xray'))["train"]
        dt_list = [{'real_index': i, 'label': l} for i, (_, l) in enumerate(self.train_dataset)]
        #print(train_dataset)

        partitioner.dataset = DT.from_list(dt_list)
        self.partition = partitioner.load_partition(partition_id=client_id)

    def __getitem__(self, index):
        batch = self.partition[index]
        real_index = batch["real_index"]
        label = batch["label"]
        #print("real index", real_index, index)

        image = self.train_dataset[real_index][0]
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.partition)


def get_data_loaders(batch_size, client_id, num_clients, image_size):
    #TODO: set dirichlet alpha as parameter
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) # 32x32

    transform_calib = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    train_dataset = XRay_Train_non_IID_Dataset(client_id=client_id, num_clients=num_clients, transform=transform_train, data_dir=data_dir)
    calib_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'chest_xray', 'train'), transform=transform_calib)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'chest_xray', 'test'), transform=transform_test)


    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    return train_dataloader, calib_dataset, test_dataloader