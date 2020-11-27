import numpy as np
import torch
import torch.nn as nn


class FPN(nn.Module):
    def __init__(self, num_ch_enc: np.ndarray, num_ch_out: int):
        """
        Initializes a feature pyramid network based on the implementation described in the paper 
        Feature Pyramid Networks for Object Detection (https://arxiv.org/pdf/1612.03144.pdf)
        :param num_ch_enc: [N], contains the number of channels of each layer of the encoder output
        :param num_ch_out: The number of output channels for each layer of the FPN
        """
        super(FPN, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_ch_pyramid = [num_ch_out] * len(num_ch_enc)
        self.layers = nn.ModuleDict()
        for i in range(len(num_ch_enc) - 1, -1, -1):
            self.layers[str(i) + "_1conv"] = nn.Sequential(nn.Conv2d(num_ch_enc[i], num_ch_out, 1, bias=False), nn.BatchNorm2d(num_ch_out))
            self.layers[str(i) + "_3conv"] = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(num_ch_out, num_ch_out, 3, bias=False), nn.BatchNorm2d(num_ch_out))

    def forward(self, input_features: list) -> list:
        """
        This function performs a forward pass in the FPN network.
        :param input_features: List of an arbitrary number of feature maps.
        :return: List of feature maps that is the same length as input_features where the number of channels of
        each feature map matches self.num_ch_pyramid.
        """
        pyramid = [None] * len(input_features)

        for i in range(len(input_features) - 1, -1, -1):
            curr_layer = self.layers[str(i) + "_1conv"](input_features[i])
            if i != len(input_features) - 1:
                curr_layer += self.upsampler(pyramid[i+1])
            pyramid[i] = self.layers[str(i) + "_3conv"](curr_layer)
        return pyramid
