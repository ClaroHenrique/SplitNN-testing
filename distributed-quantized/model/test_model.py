# Adapted from: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

__all__ = ['test_model']

# Define ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, is_client=None, split_point=None):
        super(ResNet, self).__init__()
        self.is_client = is_client
        self.split_point = split_point
        self.in_planes = 64
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        linear = nn.Linear(512 * block.expansion, num_classes)

        if self.is_layer_in_current_model(split_index=0):
            self.conv1 = conv1
            self.bn1 = nn.BatchNorm2d(64)
        if self.is_layer_in_current_model(split_index=1):
            self.layer1 = layer1
        if self.is_layer_in_current_model(split_index=2):
            self.layer2 = layer2
        if self.is_layer_in_current_model(split_index=3):
            self.layer3 = layer3
        if self.is_layer_in_current_model(split_index=4):
            self.layer4 = layer4
            self.linear = linear


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def is_layer_in_current_model(self, split_index):
        return (self.is_client and split_index <= self.split_point) or ((not self.is_client) and (split_index > self.split_point))

    def forward(self, x):
        out = x
        if self.is_layer_in_current_model(self, split_index=0):
            out = torch.relu(self.bn1(self.conv1(out)))
        if self.is_layer_in_current_model(self, split_index=1):
            out = self.layer1(out)
        if self.is_layer_in_current_model(self, split_index=2):
            out = self.layer2(out)
        if self.is_layer_in_current_model(self, split_index=3):
            out = self.layer3(out)
        if self.is_layer_in_current_model(self, split_index=4):
            out = self.layer4(out)
            out = torch.nn.functional.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(is_client, split_point):
    return ResNet(BasicBlock, [2, 2, 2, 2], is_client=is_client, split_point=split_point)


def test_model(is_client, split_point):
    model = ResNet18(is_client, split_point)
    return model
