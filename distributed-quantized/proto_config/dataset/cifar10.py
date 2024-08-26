import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size, client_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Resize((64, 64)),
    ]) # 32x32

    train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=512) #TODO: arbitrary bath_size

    return train_dataloader, test_dataloader


        


