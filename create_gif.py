import argparse
import os
import torch
from torch.utils.data import DataLoader
import yaml
from collate import Collator
from DensenetEncoder import DensenetEncoder
import imageio
from kitti_dataset import KittiDataset
import matplotlib as mpl
import matplotlib.cm as cm
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from fpn import FPN
import numpy as np


def generate_gif(experiment_dir: str, epoch: int):
    """
    Generates a gif based on a specified directory containing a config, an epoch number, and a split file.
    :param experiment_dir: Path to the config in the experiments directory that the model was trained on
    :param epoch: Epoch number corresponding to the model that metrics will be evaluated on
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = os.path.join(experiment_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    dataset = KittiDataset.init_from_config(config["dataset_config_paths"]["gif"], config["image"]["crop"], config["image"]["color"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=False,
                            collate_fn=Collator(config["image"]["height"], config["image"]["width"]), num_workers=config["num_workers"])
   
    depth_network_config = config["depth_network"]

    if depth_network_config.get("densenet"):
        models = {"depth_encoder": DensenetEncoder(depth_network_config["layers"], False)}
    else:
        models = {"depth_encoder": ResnetEncoder(depth_network_config["layers"], False)}
    decoder_num_ch = models["depth_encoder"].num_ch_enc
    
    if depth_network_config.get("fpn"):
        num_ch_fpn = depth_network_config.get("fpn_channels")
        if not num_ch_fpn:
            num_ch_fpn = 256
        models["fpn"] = FPN(decoder_num_ch, num_ch_fpn)
        decoder_num_ch = models["fpn"].num_ch_pyramid
    models["depth_decoder"] = DepthDecoder(decoder_num_ch)
    
    weights_folder = os.path.join(experiment_dir, "models", f'weights_{epoch - 1}')
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

    disp_maps = []
    images = []
    
    print(f"-> Generating gif images with size {dims[1]}x{dims[0]}")
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device).float()
            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](inputs)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](inputs))

            disp_maps.append(output[("disp", 0)])
            images.append(inputs)
    images = torch.cat(images)
    disp_maps = torch.cat(disp_maps)
    
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
    save_folder = os.path.join(experiment_dir, "gifs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imageio.mimsave(os.path.join(save_folder, f"gif_epoch_{epoch-1}.gif"), gif_images)
    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qualitative options")
    parser.add_argument("--experiment_dir",
                        type=str,
                        help="path to experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    opt = parser.parse_args()
    generate_gif(opt.experiment_dir, opt.epoch)
