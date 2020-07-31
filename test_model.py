import torch
import torch.nn.functional as F
import os
from dataloader import KittiDataset
from basic_trainer import disp_to_depth
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder

def test_depth_model(path_to_weights, test_config_path, encoder_layers, num_scales, cuda = False):

    # GPU/CPU
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Loading weights on encoder/decoder
    encoder_path = os.path.join(path_to_weights, "encoder.pth")
    decoder_path = os.path.join(path_to_weights, "encoder.pth")
    
    encoder = ResnetEncoder(encoder_layers, False)
    loaded_dict_enc = torch.load(encoder_path, map_location = device)
    
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(num_scales))
    loaded_dict = torch.load(decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()    
    
    test_dataset = KittiDataset.init_from_config(test_config_path)    
    
    # Predict depth across all test images
    inputs = torch.cat([F.interpolate((torch.tensor(test_dataset[i]["stereo_left_image"].transpose(2,0,1), device=device, dtype=torch.float32).unsqueeze(0)), [feed_height, feed_width], mode = "bilinear", align_corners = False) for i in range(0, len(test_dataset))])
    features = encoder(inputs)
    outputs = depth_decoder(features)
    disp = outputs[("disp", 0)]
    disp = F.interpolate(disp, [feed_height, feed_width], mode="bilinear", align_corners=False)
    _, depths = disp_to_depth(disp, 0.1, 100)
    
    # Compute errors