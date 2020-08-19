from networks import ResnetEncoder
from networks import FPN
from networks import DepthDecoder
import torch
import torch.nn as nn
import numpy as np

class DenseNetwork(nn.Module):
    def __init__(self, options):
        """
        Initializes a dense network for predicting depth. This includes a ResNet Encoder, a Feature Pyramid Network if desired, and a Depth Decoder.
        :param [dict] options: Dictionary where the keys are the various options the user has and the values are their choices for each option.
        """
        super(DenseNetwork, self).__init__()

        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.models = {}
        self.parameters_to_train = []
        self.models["encoder"] = ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if(self.opt.use_fpn):
            self.models["fpn"] = FPN(np.append(self.models["encoder"].num_ch_enc[1:], self.models["encoder"].num_ch_enc[-1]), self.device)
            self.parameters_to_train += list(self.models["fpn"].parameters())
            self.models["depth"] = DepthDecoder(self.models["fpn"].num_ch_pyramid, self.opt.scales)

        else:
            self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

    def forward(self, input_image):
        """
        Performs a forward pass for the DenseNetwork.
        :param [torch.Tensor] input_image: [1, 3, H, W] image that disparity is being predicted for.
        :return [dict] ouputs: dictionary containg disparity maps for the given image at various different scales.
        """
        features = self.models["encoder"](input_image)

        if(self.opt.use_fpn):
            pyramid = self.models["fpn"](features)
            outputs = self.models["depth"](pyramid)
        else:
            outputs = self.models["depth"](features)

        return outputs