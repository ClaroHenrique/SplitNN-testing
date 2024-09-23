import torch
from torch import nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# Dataset personalizado que gera imagens com pixels aleatórios
class RandomPixelDataset(Dataset):
    def __init__(self, num_images=256, img_size=(3, 32, 32)):
        self.num_images = num_images
        self.img_size = img_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img = torch.ones(self.img_size)
        label = 1
        return img, label



def get_data_loaders(batch_size, client_id):

    train_dataset = RandomPixelDataset(num_images=1024, img_size=(3, 32, 32))
    test_dataset = RandomPixelDataset(num_images=1024, img_size=(3, 32, 32))

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024, drop_last=True) #TODO: arbitrary bath_size

    return train_dataloader, test_dataloader


        



