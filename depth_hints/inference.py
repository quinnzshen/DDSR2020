import torch
import os
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.fpn import FPN
import numpy as np


def run_inference(model_path, image, use_fpn=False):
    """
    This function does a forward pass on an image given a model and outputs a disparity map.
    :param [string] model_path: Path to a pretrained model that user wants to use to predict disparity.
    :param [torch.Tensor] image: [1, 3, H, W], Image that disparity is being predicted for.
    :return [torch.Tensor] disp: [1, 1, H, W], Disparity map for inputted image.
    """
    device = torch.device("cuda")
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    
    # LOADING PRETRAINED MODEL
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    image = torch.nn.functional.interpolate(image, (feed_height, feed_width), mode="bilinear", align_corners=False)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # Load FPN
    if(use_fpn):
        fpn = FPN(np.append(encoder.num_ch_enc[1:], encoder.num_ch_enc[-1]))
        fpn.to(device)
        # Load Depth Decoder
        depth_decoder = DepthDecoder(num_ch_enc=fpn.num_ch_pyramid, scales=range(4))
    else:
        # Load depth decoder
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    # Predict depth for single image
    image = image.to(device)
    features = encoder(image)

    if(use_fpn):
        pyramid = fpn(features)
        outputs = depth_decoder(pyramid)

    else:
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    return disp
