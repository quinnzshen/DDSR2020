import argparse
from collate import Collator
from kitti_dataset import KittiDataset
import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader
import yaml

cv2.setNumThreads(0)

STEREO_SCALE_FACTOR = 5.4
# Sets N bins for metrics, from 0 -> BINS[0], BINS[1] -> BINS[2}, etc.
BINS = [25, 50, 75, 100]


def run_metrics(log_dir):
    """Computes metrics based on a specified directory containing a config and an epoch number. Adapted from Monodepth2
    :param [String] config_path: Path to the config directory that the model was trained on
    :param [int] epoch: Epoch number corresponding to the model that metrics will be evaluated on
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = log_dir
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    dataset = KittiDataset.init_from_config(config["test_config_path"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=False,
                            collate_fn=Collator(config["height"], config["width"]), num_workers=config["num_workers"])
    pred_disps = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device).float()
            pred_disp = inputs.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join(config["gt_path"], "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    image_len = pred_disps.shape[0]
    avg = 0
    for i in range(image_len):

        gt_depth = gt_depths[i]
        print(gt_depth.max())
        avg += gt_depth.max()

    print("\n-> Done!")
    print("avg max:", avg / image_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metrics options")
    parser.add_argument("--log_dir",
                        type=str,
                        help="path to experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    opt = parser.parse_args()
    run_metrics(opt.log_dir)
