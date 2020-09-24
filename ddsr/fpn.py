import torch
import torch.nn as nn
import numpy as np


class FPN(nn.Module):
    def __init__(self, num_ch_enc, num_ch_out):
        """
        Initializes a feature pyramid network based on the implementation described in the paper 
        Feature Pyramid Networks for Object Detection (https://arxiv.org/pdf/1612.03144.pdf)
        :param [np.array] num_ch_enc: [N], contains the number of channels of each layer for the output of the pyramid.
        """
        super(FPN, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_ch_pyramid = [num_ch_out] * len(num_ch_enc)
        self.layers = nn.ModuleDict()
        for i in range(len(num_ch_enc) - 1, -1, -1):
            self.layers[str(i) + "_1conv"] = nn.Sequential(nn.Conv2d(num_ch_enc[i], num_ch_out, 1, bias=False), nn.BatchNorm2d(num_ch_out))
            self.layers[str(i) + "_3conv"] = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(num_ch_out, num_ch_out, 3, bias=False), nn.BatchNorm2d(num_ch_out))

    def forward(self, input_features):
        """
        This function performs a forward pass in the FPN network.
        :param [list] input_features: List of an arbitrary number of feature maps.
        :return [list]: List of feature maps that is the same length as input_features where the number of channels of
        each feature map matches self.num_ch_pyramid.
        """
        pyramid = [None] * len(input_features)

        for i in range(len(input_features) - 1, -1, -1):
            curr_layer = self.layers[str(i) + "_1conv"](input_features[i])
            if i != len(input_features) - 1:
                print(pyramid[i+1].shape, self.upsampler(pyramid[i+1]).shape)
                curr_layer += self.upsampler(pyramid[i+1])
            pyramid[i] = self.layers[str(i) + "_3conv"](curr_layer)
        return pyramid
