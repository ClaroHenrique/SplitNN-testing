# Adapted from: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
# More appropriate ResNet for 32x32 images

import torch
import torch.nn as nn

__all__ = ['ResNet18_32x32', 'ResNet34_32x32', 'ResNet50_32x32', 'ResNet101_32x32', 'ResNet152_32x32']


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, is_client, split_point, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.split_point = split_point
        self.is_client = is_client

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        linear = nn.Linear(512*block.expansion, num_classes)

        if self.is_layer_in_current_model(split_index=0):
            self.conv1 = conv1
            self.bn1 = bn1
        if self.is_layer_in_current_model(split_index=1):
            self.layer1 = layer1
        if self.is_layer_in_current_model(split_index=2):
            self.layer2 = layer2
        if self.is_layer_in_current_model(split_index=3):
            self.layer3 = layer3
        if self.is_layer_in_current_model(split_index=4):
            self.layer4 = layer4
        if self.is_layer_in_current_model(split_index=5):
            self.linear = linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        if self.is_layer_in_current_model(split_index=0):
            out = F.relu(self.bn1(self.conv1(x)))
        if self.is_layer_in_current_model(split_index=1):
            out = self.layer1(out)
        if self.is_layer_in_current_model(split_index=2):
            out = self.layer2(out)
        if self.is_layer_in_current_model(split_index=3):
            out = self.layer3(out)
        if self.is_layer_in_current_model(split_index=4):
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        if self.is_layer_in_current_model(split_index=5):
            out = self.linear(out)
        return out
    
    def is_layer_in_current_model(self, split_index):
        return (self.is_client and split_index <= self.split_point) or ((not self.is_client) and (split_index > self.split_point))



def ResNet18_32x32(is_client, split_point=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], is_client=is_client, split_point=split_point,)


def ResNet34_32x32(is_client, split_point=0):
    return ResNet(BasicBlock, [3, 4, 6, 3], is_client=is_client, split_point=split_point,)


def ResNet50_32x32(is_client, split_point=0):
    return ResNet(Bottleneck, [3, 4, 6, 3], is_client=is_client, split_point=split_point,)


def ResNet101_32x32(is_client, split_point=0):
    return ResNet(Bottleneck, [3, 4, 23, 3], is_client=is_client, split_point=split_point,)


def ResNet152_32x32(is_client, split_point=0):
    return ResNet(Bottleneck, [3, 8, 36, 3], is_client=is_client, split_point=split_point,)

