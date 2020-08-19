import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, num_ch_input, num_ch_pyramid, device):
        super(FPN, self).__init__()
        self.device = device
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampler.to(self.device)
        self.num_ch_input = num_ch_input
        self.num_ch_pyramid = num_ch_pyramid
        self.lateral_convs = []
        num_ch_top_down = num_ch_input[::-1]
        for idx in range(1, len(num_ch_input)):
            self.lateral_convs.append(nn.Conv2d(num_ch_input[idx], num_ch_to_down[idx-1], 1))
        for conv in self.lateral_convs:
            conv.to(self.device)

        
    
    def forward(self, input_features):
        top_down = input_features[::-1]
        pyramid = [top_down[0]]
        input_features.reverse()
        
        for idx in range(1, len(input_features)):
            bottom_up_convoluted = self.lateral_convs[idx](input_features[idx])
            level = self.upsampler(top_down[idx-1]) + bottom_up_convoluted
            pyramid.append(level)
        pyramid.reverse()
        return pyramid
