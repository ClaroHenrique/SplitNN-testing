import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features, is_client, split_point, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.is_client = is_client
        self.split_point = split_point
        self.features = features

        if not self.is_client:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if not self.is_client:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def is_layer_in_current_model(current_block, is_client, split_point):
    return (is_client and current_block <= split_point) or ((not is_client) and current_block > split_point)

def make_layers(cfg, is_client, split_point, batch_norm=False):
    layers = []
    in_channels = 3
    current_block = 1
    for v in cfg:
        if is_layer_in_current_model(current_block, is_client, split_point):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
        if v == 'M':
            current_block += 1
        else:
            in_channels = v
            
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, is_client, split_point, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], is_client, split_point, batch_norm=batch_norm), is_client, split_point, **kwargs)
    return model


def vgg11(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg11'
    return arch, _vgg(arch, is_client, split_point, 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg11_bn'
    return arch, _vgg(arch, is_client, split_point, 'A', True, pretrained, progress, **kwargs)


def vgg13(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg13'
    return arch, _vgg(arch, is_client, split_point, 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg13_bn'
    return arch, _vgg(arch, is_client, split_point, 'B', True, pretrained, progress, **kwargs)


def vgg16(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg16'
    return arch, _vgg(arch, is_client, split_point, 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg16_bn'
    return arch, _vgg(arch, is_client, split_point, 'D', True, pretrained, progress, **kwargs)


def vgg19(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg19'
    return arch, _vgg(arch, is_client, split_point, 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(is_client, split_point, pretrained=False, progress=True, **kwargs):
    arch = 'vgg19_bn'
    return arch, _vgg(arch, is_client, split_point, 'E', True, pretrained, progress, **kwargs)

def ClientModel(split_point=1, num_classes=10):
    model_name, model = vgg19_bn(is_client=True, split_point=split_point, num_classes=num_classes)
    return model, model_name + "_client"

def ServerModel(split_point=1, num_classes=10):
    model_name, model = vgg19_bn(is_client=False, split_point=split_point, num_classes=num_classes)
    return model, model_name + "_server"



