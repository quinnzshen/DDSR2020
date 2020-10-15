import argparse
import os
import torch
from torch.utils.data import DataLoader
import yaml
from collate import Collator
from kitti_dataset import KittiDataset
from third_party.DensenetEncoder import DensenetEncoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from fpn import FPN


def generate_qualitative(log_dir, epoch):
    """
    Generates metrics based on a specified directory containing a config and an epoch number.
    :param [str] log_dir: Path to the config in the experiments directory that the model was trained on
    :param [int] epoch: Epoch number corresponding to the model that metrics will be evaluated on
    :return [torch.Tensor]: Tensor representing the generated qualitative depth maps in dimension [B, 1, H, W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = os.path.join(log_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    dataset = KittiDataset.init_from_config(config["dataset_config_paths"]["qual"], config["image"]["crop"], config["image"]["color"])
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
    
    weights_folder = os.path.join(log_dir, "models", f'weights_{epoch - 1}')
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

    outputs = []

    print("-> Generating qualitative predictions with size {}x{}".format(
        dims[1], dims[0]))

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device).float()
            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](inputs)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](inputs))

            outputs.append(output[("disp", 0)])
    outputs = torch.cat(outputs)

    print("\n-> Done!")

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qualitative options")
    parser.add_argument("--log_dir",
                        type=str,
                        help="path to config in experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    opt = parser.parse_args()
    generate_qualitative(opt.log_dir, opt.epoch)
