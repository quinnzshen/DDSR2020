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
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest', align_corners=False)
        self.num_ch_pyramid = [num_ch_out] * len(num_ch_enc)
        # self.convs = []
        # self.layers = nn.ModuleDict({str(len(num_ch_enc) - 1) + "_1x1conv": nn.Conv2d(num_ch_enc[len(num_ch_enc) - 1], num_ch_out, 1)})
        for i in range(len(num_ch_enc) - 1, -1, -1):
            self.layers[str(i) + "_1x1conv"] = nn.Conv2d(num_ch_enc[i], num_ch_out, 1)
            self.layers[str(i) + "_3x3conv"] = nn.Conv2d(num_ch_out, num_ch_out, 3)
        #     self.convs.append(nn.Conv2d(num_ch_enc[i], num_ch_enc[i+1], 1))
        # self.fpn = nn.ModuleList(self.convs)

    def forward(self, input_features):
        """
        This function performs a forward pass in the FPN network.
        :param [list] input_features: List of an arbitrary number of feature maps.
        :return [list]: List of feature maps that is the same length as input_features where the number of channels of
        each feature map matches self.num_ch_pyramid.
        """
        pyramid = [None] * len(input_features)
        # pyramid[-1] = self.layers[str(len(input_features) - 1) + "_1x1conv"](input_features[-1])
        for i in range(len(input_features) - 1, -1, -1):
            curr_layer = self.layers[str(i) + "_1x1conv"](input_features[i])
            if i != len(input_features) - 1:
                curr_layer += self.upsampler(pyramid[i+1])
            pyramid[i] = self.layers[str(i) + "_3x3conv"](curr_layer)
            # pyramid.append(self.upsampler(input_features[i+1]) + self.convs[i](input_features[i]))
        # pyramid.append(input_features[-1])
        return pyramid
