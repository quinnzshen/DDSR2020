import torch
import os
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.fpn import FPN
import numpy as np


def load_model(model_path, use_fpn=False):
    """
    Loads a pretrained model from a specified model path.
    :param model_path [string]: path to pretrained model's weights folder.
    :param use_fpn [boolean]: indicates whether an FPN was used when training the specified model.
    :return models [tuple]: tuple containing the loaded pretrained models -  (encoder, fpn, decoder)
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
    else:
        fpn = None
    # Load Depth Decoder
    depth_decoder = DepthDecoder(num_ch_enc=fpn.num_ch_pyramid, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return (encoder, fpn, depth_decoder)

def run_inference(models, image):
    """
    This function does a forward pass on an image given a model and outputs a disparity map.
    :param [tuple] models: tuple containing the loaded pretrained models - either (encoder, decoder) or (encoder, fpn, decoder)
    :param [torch.Tensor] image: [1, 3, H, W], Image that disparity is being predicted for.
    :return [torch.Tensor] disp: [1, 1, H, W], Disparity map for inputted image.
    """
    
    if(models[1] != None):
        encoder = models[0]
        fpn = models[1]
        decoder = models[2]
    else:
        encoder = models[0]
        decoder = models[1]

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
