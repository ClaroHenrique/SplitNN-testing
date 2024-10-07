import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size, client_id, image_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]) # 32x32

    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(train_dataset, batch_size=9000, drop_last=True) #TODO: arbitrary bath_size

    return train_dataloader, test_dataloader


        



