from __future__ import absolute_import, division, print_function
import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
import yaml
from torchvision import transforms

from color_utils import convert_rgb
from densenet_encoder import DensenetEncoder
from fpn import FPN
from third_party.monodepth2.DepthDecoder import DepthDecoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.layers import disp_to_depth


def run_inference(exp_dir: str, epoch: int, img_path: str, output_path: str):
    """
    Predicts disparity maps for a given set of images
    :param exp_dir: Path to an experiment directory
    :param epoch: epoch number correpsonding to a model within the experiment directory
    :param img_path: path to an input image OR a folder containing input images
    :param output_path: path to the directory where the output disparity maps will be saved
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = os.path.join(exp_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    depth_network_config = config["depth_network"]

    if depth_network_config.get("densenet"):
        models = {"depth_encoder": DensenetEncoder(depth_network_config["layers"], False, color=config["image"]["color"])}
    else:
        models = {"depth_encoder": ResnetEncoder(depth_network_config["layers"], False, color=config["image"]["color"])}
    decoder_num_ch = models["depth_encoder"].num_ch_enc
    
    if depth_network_config.get("fpn"):
        num_ch_fpn = depth_network_config.get("fpn_channels")
        if not num_ch_fpn:
            num_ch_fpn = 256
        models["fpn"] = FPN(decoder_num_ch, num_ch_fpn)
        decoder_num_ch = models["fpn"].num_ch_pyramid
    models["depth_decoder"] = DepthDecoder(decoder_num_ch)
    
    weights_folder = os.path.join(exp_dir, "models", f'weights_{epoch - 1}')
    print(f'-> Loading weights from {weights_folder}')

    for model_name in models:
        preset_path = os.path.join(weights_folder, f"{model_name}.pth")
        model_dict = models[model_name].state_dict()
        preset_dict = torch.load(preset_path)
        if model_name == "depth_encoder":
            dims = (preset_dict["height"], preset_dict["width"])
        model_dict.update({k: v for k, v in preset_dict.items() if k in model_dict})
        models[model_name].load_state_dict(model_dict)
        models[model_name].cuda()
        models[model_name].eval()
    
    print("-> Generating predictions with size {}x{}".format(
        dims[1], dims[0]))
    
    # Locating input images
    if os.path.isfile(img_path):
        # Only testing on a single image
        paths = [img_path]
        if output_path is None:
            output_directory = os.path.dirname(img_path)
        else:
            output_directory = output_path
    
    elif os.path.isdir(img_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(img_path, '*.*'))
        if output_path is None:
            output_directory = img_path
        else:
            output_directory = output_path
    else:
        raise Exception("Can not find image_path: {}".format(img_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Running inference on each input image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((dims[1], dims[0]), pil.LANCZOS)
            input_image = convert_rgb(transforms.ToTensor()(input_image).unsqueeze(0), color=config["image"]["color"])

            # Inference
            input_image = input_image.to(device)
            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](input_image)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](input_image))

            disp = output[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving NumPy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            
            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))

            np.save(name_dest_npy, scaled_disp.cpu().numpy())
            im.save(name_dest_im)
            print("Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qualitative options")
    parser.add_argument("--exp_dir",
                        type=str,
                        help="path to config in experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    parser.add_argument("--img_path",
                        type=str,
                        help="path to an input image OR a folder containing input images")
    parser.add_argument("--output_path",
                        type=int,
                        help="path to the directory where the output disparity maps will be saved",
                        default=None)
    opt = parser.parse_args()
    run_inference(opt.exp_dir, opt.epoch, opt.img_path, opt.output_path)
