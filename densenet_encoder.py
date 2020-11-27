from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class _DenseLayer(nn.Module):
    """
    Pytorch module for a dense layer. Taken from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient


class _DenseBlock(nn.ModuleDict):
    """
    Pytorch module for a dense block. Taken from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    """
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """
    Pytorch module for a transition layer. Taken from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    """

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNetMultiImageInput(models.DenseNet):
    """
    Constructs a DenseNet model with varying number of input images.
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, num_input_images=1):
        super(DenseNetMultiImageInput, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3 * num_input_images, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


def densenet_multiimage_input(num_layers: int, pretrained: bool = False, num_input_images: int = 1) -> DenseNetMultiImageInput:
    """
    Constructs a DenseNet model.
    :param num_layers: Number of DenseNet layers. Must be 121, 161, 169, or 201.
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param num_input_images: Number of frames stacked as input
    :return: A DenseNet model with the designated parameters
    """
    assert num_layers in [121, 169, 201, 161], "Can only run with 121, 161, 169, or 201 layer densenet"
    growth_rate = {121: 32, 161: 48, 169: 32, 201: 32}[num_layers]
    block_config = {121: (6, 12, 24, 16), 161: (6, 12, 36, 24), 169: (6, 12, 32, 32), 201: (6, 12, 48, 32)}[num_layers]
    num_init_features = {121: 64, 161: 96, 169: 64, 201: 64}[num_layers]
    model = DenseNetMultiImageInput(growth_rate=growth_rate, block_config=block_config,
                                    num_init_features=num_init_features, num_input_images=num_input_images)
    if pretrained:
        loaded = model_zoo.load_url(models.densenet.model_urls['densenet{}'.format(num_layers)])
        loaded['features[0].weight'] = torch.cat(
            [loaded['features[0].weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class DensenetEncoder(nn.Module):
    """
    Module for a densenet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, color="RGB"):
        super(DensenetEncoder, self).__init__()

        self.color = color

        self.num_ch_enc = np.array([64, 256, 512, 1024, 1024])
        densenets = {121: models.densenet121,
                     169: models.densenet169,
                     201: models.densenet201,
                     161: models.densenet161}

        if num_layers not in densenets:
            raise ValueError("{} is not a valid number of densenet layers".format(densenets))

        if num_input_images > 1:
            self.encoder = densenet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = densenets[num_layers](pretrained)

        if color == "HSV":
            self.encoder.features[0] = nn.Conv2d(4 * num_input_images, self.num_ch_enc[0], kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, input_image):
        features = []
        x = input_image
        if self.color == "RGB":
            x = (input_image - 0.45) / 0.225

        for i, layer in enumerate(self.encoder.features):
            x = layer(x)
            if i == 2:
                features.append(x)
            elif i > 3 and i % 2 == 0:
                features.append(x)
        return features
