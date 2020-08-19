import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, num_ch_pyramid, device):
        """
        Initializes a feature pyramid network based on the implementation described in the paper 
        Feature Pyramid Networks for Object Detection (https://arxiv.org/pdf/1612.03144.pdf)
        :param [np.array] num_ch_pyramid: [N], contains the number of channels of each layer for the output of the pyramid.
        :param [torch.device] device: device (either cuda or cpu) being used. 
        """
        super(FPN, self).__init__()
        self.device = device
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampler.to(self.device)
        self.num_ch_pyramid = num_ch_pyramid
        
    
    def forward(self, input_features):
        """
        This function performs a forward pass in the FPN network.
        :param input_features [list]: List of an arbitrary number of feature maps.
        :return pyramid [list]: List of feature maps that is the same length as input_features where the number of channels of 
        each feature map matches self.num_ch_pyramid.
        """
        top_down = input_features[::-1]
        pyramid = [top_down[0]]
        input_features.reverse()
        
        for idx in range(1, len(input_features)):
            conv = nn.Conv2d(input_features[idx].shape[1], top_down[idx-1].shape[1], 1)
            conv.to(self.device)
            bottom_up_convoluted = conv(input_features[idx])
            level = self.upsampler(top_down[idx-1]) + bottom_up_convoluted
            pyramid.append(level)
        pyramid.reverse()
        return pyramid
