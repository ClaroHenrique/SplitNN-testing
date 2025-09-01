import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms


def get_data_loaders(batch_size, client_id, image_size):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    # ])

    transform_calib = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    # ])

    transform_test = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = torchvision.datasets.Imagenette(root='../data', split="train", download=True, transform=transform_train)
    calib_dataset = torchvision.datasets.Imagenette(root='../data', split="train", download=True, transform=transform_calib)
    test_dataset = torchvision.datasets.Imagenette(root='../data', split="val", download=True, transform=transform_test)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, drop_last=False) #TODO: arbitrary test bath_size

    return train_dataloader, calib_dataloader, test_dataloader

