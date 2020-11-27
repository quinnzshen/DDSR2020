import argparse
import os

import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from collate import Collator
from color_utils import convert_rgb
from densenet_encoder import DensenetEncoder
from fpn import FPN
from kitti_dataset import KittiDataset
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder


def generate_gif(exp_dir: str, exp_epoch: int, baseline_dir: str, baseline_epoch: int):
    """
    Generates a gif of input RGB images, corresponding disparity maps from a baseline model, and corresponding disparity
    maps from an experiment model
    :param exp_dir: Path to the directory of the experiment training job
    :param exp_epoch: Epoch number of the experiment model weights
    :param baseline_dir: Path to the directory of the baseline training job
    :param baseline_epoch: Epoch number of the baseline model weights
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from experiment config
    exp_config_path = os.path.join(exp_dir, "config.yml")
    with open(exp_config_path) as file:
        exp_config = yaml.load(file, Loader=yaml.Loader)

    dataset = KittiDataset.init_from_config(exp_config["dataset_config_paths"]["gif"])
    dataloader = DataLoader(dataset, exp_config["batch_size"], shuffle=False,
                            collate_fn=Collator(exp_config["image"]["height"], exp_config["image"]["width"]),
                            num_workers=exp_config["num_workers"], pin_memory=True)
   
    exp_depth_network_config = exp_config["depth_network"]
    
    # Setting up experiment model
    if exp_depth_network_config.get("densenet"):
        exp_models = {"depth_encoder": DensenetEncoder(exp_depth_network_config["layers"], False,
                                                       color=exp_config["image"]["color"])}
    else:
        exp_models = {"depth_encoder": ResnetEncoder(exp_depth_network_config["layers"], False,
                                                     color=exp_config["image"]["color"])}
    exp_decoder_num_ch = exp_models["depth_encoder"].num_ch_enc
    
    if exp_depth_network_config.get("fpn"):
        exp_num_ch_fpn = exp_depth_network_config.get("fpn_channels")
        if not exp_num_ch_fpn:
            exp_num_ch_fpn = 256
        exp_models["fpn"] = FPN(exp_decoder_num_ch, exp_num_ch_fpn)
        exp_decoder_num_ch = exp_models["fpn"].num_ch_pyramid
    exp_models["depth_decoder"] = DepthDecoder(exp_decoder_num_ch)
    
    exp_weights_folder = os.path.join(exp_dir, "models", f'weights_{exp_epoch - 1}')
    print(f'-> Loading weights from {exp_weights_folder}')

    # Load data from baseline config
    baseline_config_path = os.path.join(baseline_dir, "config.yml")
    with open(baseline_config_path) as file:
        baseline_config = yaml.load(file, Loader=yaml.Loader)
    baseline_depth_network_config = baseline_config["depth_network"]
    
    # Setting up baseline model
    if baseline_depth_network_config.get("densenet"):
        baseline_models = {"depth_encoder": DensenetEncoder(baseline_depth_network_config["layers"], False,
                                                            color=baseline_config["image"]["color"])}
    else:
        baseline_models = {"depth_encoder": ResnetEncoder(baseline_depth_network_config["layers"], False,
                                                          color=baseline_config["image"]["color"])}
    baseline_decoder_num_ch = baseline_models["depth_encoder"].num_ch_enc

    if baseline_depth_network_config.get("fpn"):
        baseline_num_ch_fpn = baseline_depth_network_config.get("fpn_channels")
        if not baseline_num_ch_fpn:
            baseline_num_ch_fpn = 256
        baseline_models["fpn"] = FPN(baseline_decoder_num_ch, baseline_num_ch_fpn)
        baseline_decoder_num_ch = baseline_models["fpn"].num_ch_pyramid
    baseline_models["depth_decoder"] = DepthDecoder(baseline_decoder_num_ch)
    
    baseline_weights_folder = os.path.join(baseline_dir, "models", f'weights_{baseline_epoch - 1}')
    print(f'-> Loading weights from {baseline_weights_folder}')
    
    # Loading weights from experiment model
    for model_name in exp_models:
        preset_path = os.path.join(exp_weights_folder, f"{model_name}.pth")
        model_dict = exp_models[model_name].state_dict()
        preset_dict = torch.load(preset_path)
        if model_name == "depth_encoder":
            dims = (preset_dict["height"], preset_dict["width"])
        model_dict.update({k: v for k, v in preset_dict.items() if k in model_dict})
        exp_models[model_name].load_state_dict(model_dict)
        exp_models[model_name].cuda()
        exp_models[model_name].eval()

    # Loading weights from baseline model
    for model_name in baseline_models:
        preset_path = os.path.join(baseline_weights_folder, f"{model_name}.pth")
        model_dict = baseline_models[model_name].state_dict()
        preset_dict = torch.load(preset_path)
        if model_name == "depth_encoder":
            dims = (preset_dict["height"], preset_dict["width"])
        model_dict.update({k: v for k, v in preset_dict.items() if k in model_dict})
        baseline_models[model_name].load_state_dict(model_dict)
        baseline_models[model_name].cuda()
        baseline_models[model_name].eval()

    exp_disp_maps = []
    baseline_disp_maps = []
    images = []
    
    print(f"-> Generating gif images with size {dims[1]}x{dims[0]}")
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device)
            images.append(inputs)

            if exp_config.get("use_fpn"):
                exp_output = exp_models["depth_decoder"](exp_models["fpn"](exp_models["depth_encoder"](
                    convert_rgb(inputs, exp_config["image"]["color"]))))
            else:
                exp_output = exp_models["depth_decoder"](exp_models["depth_encoder"](
                    convert_rgb(inputs, exp_config["image"]["color"])))
            
            if baseline_config.get("use_fpn"):
                baseline_output = baseline_models["depth_decoder"](baseline_models["fpn"](baseline_models["depth_encoder"](
                    convert_rgb(inputs, baseline_config["image"]["color"]))))
            else:
                baseline_output = baseline_models["depth_decoder"](baseline_models["depth_encoder"](
                    convert_rgb(inputs, baseline_config["image"]["color"])))
            
            exp_disp_maps.append(exp_output[("disp", 0)])
            baseline_disp_maps.append(baseline_output[("disp", 0)])

    images = torch.cat(images)
    exp_disp_maps = torch.cat(exp_disp_maps)
    baseline_disp_maps = torch.cat(baseline_disp_maps)
    disp_combined = (baseline_disp_maps, exp_disp_maps)
    disp_maps = torch.cat(disp_combined, 2)
    print("-> Creating gif")
    
    gif_images = []
    for i, disp in enumerate(disp_maps):
        disp_np = disp.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_disp = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
        
        image = images[i]
        img_np = image.squeeze().cpu().detach().numpy() * 255
        colormapped_img = img_np.astype(np.uint8).transpose(1, 2, 0)

        gif_image = np.vstack((colormapped_img, colormapped_disp))
        gif_images.append(gif_image)
    save_folder = os.path.join(exp_dir, "gifs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    imageio.mimsave(os.path.join(save_folder, "gif_epoch_{}.gif".format(exp_epoch)), gif_images)
    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qualitative options")
    parser.add_argument("--exp_dir",
                        type=str,
                        help="path to experiment directory")
    parser.add_argument("--exp_epoch",
                        type=int,
                        help="epoch number for experiment model")
    parser.add_argument("--baseline_dir",
                        type=str,
                        help="path to baseline directory")
    parser.add_argument("--baseline_epoch",
                        type=int,
                        help="epoch number for baseline_model")
    opt = parser.parse_args()

    generate_gif(opt.exp_dir, opt.exp_epoch, opt.baseline_dir, opt.baseline_epoch)
