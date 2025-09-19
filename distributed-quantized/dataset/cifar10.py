import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

def get_dataset_name():
    return "cifar10"

def get_data_loaders(batch_size, client_id, image_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(image_size),
    ]) # 32x32

    transform_calib = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(image_size),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(image_size),
    ]) 

    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    calib_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_calib)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False) #TODO: arbitrary test bath_size

    return train_dataloader, calib_dataloader, test_dataloader

