import torch
import torch.nn as nn
import numpy as np


class FPN(nn.Module):
    def __init__(self, num_ch_enc):
        """
        Initializes a feature pyramid network based on the implementation described in the paper 
        Feature Pyramid Networks for Object Detection (https://arxiv.org/pdf/1612.03144.pdf)
        :param [np.array] num_ch_enc: [N], contains the number of channels of each layer for the output of the pyramid.
        """
        super(FPN, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.num_ch_pyramid = np.roll(num_ch_enc, -1)
        self.num_ch_pyramid[-1] = self.num_ch_pyramid[-2]
        self.convs = []
        for i in range(len(num_ch_enc) - 1):
            self.convs.append(nn.Conv2d(num_ch_enc[i], num_ch_enc[i+1], 1))
        self.fpn = nn.ModuleList(self.convs)

    def forward(self, input_features):
        """
        This function performs a forward pass in the FPN network.
        :param [list] input_features: List of an arbitrary number of feature maps.
        :return [list]: List of feature maps that is the same length as input_features where the number of channels of
        each feature map matches self.num_ch_pyramid.
        """
        pyramid = []
        for i in range(len(input_features) - 1):
            pyramid.append(self.upsampler(input_features[i+1]) + self.convs[i](input_features[i]))
        pyramid.append(input_features[-1])

        return pyramid
