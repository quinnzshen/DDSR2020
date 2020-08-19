import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, num_ch_pyramid, device):
        super(FPN, self).__init__()
        self.device = device
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsampler.to(self.device)
        self.num_ch_pyramid = num_ch_pyramid
        
    
    def forward(self, input_features):
        top_down = input_features[::-1]
        pyramid = [top_down[0]]
        input_features.reverse()
        
        for idx in range(1, len(input_features)):
            conv = nn.Conv2d(input_features[idx].shape[1], top_down[idx-1].shape[1], 1)
            conv.to(self.device) # change to be adaptable
            tmp = conv(input_features[idx])
            level = self.upsampler(top_down[idx-1]) + tmp
            pyramid.append(level)
        pyramid.reverse()
        return pyramid
