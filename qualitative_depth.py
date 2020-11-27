import argparse
import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from collate import Collator
from color_utils import convert_rgb
from densenet_encoder import DensenetEncoder
from fpn import FPN
from kitti_dataset import KittiDataset
from third_party.monodepth2.DepthDecoder import DepthDecoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder


def generate_qualitative(exp_dir: str, epoch: int) -> torch.Tensor:
    """
    Generates qualitative images based on a specified directory containing a config and an epoch number.
    :param exp_dir: Path to an experiment directory
    :param epoch: Epoch number corresponding to a model within the experiment directory
    :return: Tensor representing the generated qualitative depth maps in dimension [B, 1, H, W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = os.path.join(exp_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    dataset = KittiDataset.init_from_config(config["dataset_config_paths"]["qual"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=False,
                            collate_fn=Collator(config["image"]["height"], config["image"]["width"]),
                            num_workers=config["num_workers"], pin_memory=True)
   
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

    disp_maps = []

    print(f"-> Generating qualitative predictions with size {dims[1]}x{dims[0]}")

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device)
            inputs = convert_rgb(inputs, config["image"]["color"])

            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](inputs)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](inputs))

            disp_maps.append(output[("disp", 0)])
    
    disp_maps = torch.cat(disp_maps)
    
    print("-> Saving images")
    
    outputs = []
    for i, disp in enumerate(disp_maps):
        disp_np = disp.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_disp = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
        final_disp = transforms.ToTensor()(colormapped_disp)
        outputs.append(final_disp)

        save_folder = os.path.join(exp_dir, "qual_images", "qual_images_epoch_{}".format(epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        path = os.path.join(save_folder, "img_" + str(i)+ ".jpeg")
        save_image(final_disp, path)
   
    print("\n-> Done!")

    return disp_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qualitative options")
    parser.add_argument("--exp_dir",
                        type=str,
                        help="path to config in experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    opt = parser.parse_args()
    generate_qualitative(opt.exp_dir, opt.epoch)
