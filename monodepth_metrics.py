import argparse
from collate import Collator
from kitti_dataset import KittiDataset
import numpy as np
import os
import csv
import cv2
import torch
import time
from torch.utils.data import DataLoader
import yaml
from densenet_encoder import DensenetEncoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from third_party.monodepth2.layers import disp_to_depth
from fpn import FPN
from color_utils import convert_rgb

cv2.setNumThreads(0)

# Sets N bins for metrics, from 0 -> BINS[0], BINS[1] -> BINS[2}, etc.
BINS = [25, 50, 75, 100]

STEREO_SCALE_FACTOR = 5.4

MIN_DEPTH = 0.001
MAX_DEPTH = 80


def get_labels() -> list:
    """
    Gets the lables for the metrics
    :return: List of strings representing the respective metrics
    """
    labels = ["metric_time", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    for i in BINS:
        labels.extend(["abs_rel_" + str(i), "a1_" + str(i)])
    return labels


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Computation of error metrics between predicted and ground truth depths.
    Adapted from https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
    :param gt: Ground truth depth maps
    :param pred: Predicted depth maps
    :return: NumPy array with respective metrics
    """
    metrics = np.empty(7 + len(BINS) * 2, dtype=np.float64)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    metrics[:7] = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

    prev_depth = 0
    for i in range(len(BINS)):
        mask_indices = np.logical_and(prev_depth <= gt, gt < BINS[i])
        filt_gt = gt[mask_indices]
        filt_pred = pred[mask_indices]
        metrics[7 + i * 2] = np.mean(np.abs(filt_gt - filt_pred) / filt_gt)
        metrics[8 + i * 2] = (np.maximum((filt_gt / filt_pred), (filt_pred / filt_gt)) < 1.25).mean()
        prev_depth = BINS[i]

    return metrics


def run_metrics(exp_dir: str, epoch: int, lidar: bool) -> tuple:
    """
    Computes metrics for a single epoch.
    Adapted from https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
    :param exp_dir: Path to the experiment directory
    :param epoch: Epoch number corresponding to the model that metrics will be evaluated on
    :param lidar: Setting to True --> Lidar data (eigen), False --> improved GT depth maps (eigen_benchmark)
    :return: Returns mean metrics and their labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from config
    config_path = os.path.join(exp_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if lidar:
        dataset = KittiDataset.init_from_config(config["dataset_config_paths"]["test_lidar"], config["image"]["crop"])
    else:
        dataset = KittiDataset.init_from_config(config["dataset_config_paths"]["test_gt_map"], config["image"]["crop"])
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

    pred_disps = []

    print(f"-> Computing predictions with size {dims[1]}x{dims[0]}")
    start_metric_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device)
            inputs = convert_rgb(inputs, config["image"]["color"])

            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](inputs)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](inputs))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], config["min_depth"], config["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    if lidar:
        gt_path = os.path.join(config["gt_dir"], "gt_lidar.npz")
        print("-> Evaluating from LiDAR data")
    else:
        gt_path = os.path.join(config["gt_dir"], "gt_depthmaps.npz")
        print("-> Evaluating from KITTI ground truth depth maps")

    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    labels = get_labels()
    image_len = pred_disps.shape[0]

    ratios = np.empty(image_len, dtype=np.float32)
    errors = np.empty((image_len, 7 + len(BINS) * 2), dtype=np.float64)

    for i in range(image_len):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if lidar:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if config["use_stereo"]:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios[i] = ratio
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors[i] = compute_errors(gt_depth, pred_depth)

    total_metric_time = time.time() - start_metric_time

    if config["use_stereo"]:
        print(f"   Stereo evaluation - disabling median scaling, scaling by {STEREO_SCALE_FACTOR}")
    else:
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:.3f} | std: {np.std(ratios / med):.3f}")

    mean_errors = np.nanmean(errors, 0).tolist()
    mean_errors.insert(0, total_metric_time)

    print("\n  " + ("{:>11} | " * len(labels)).format(*labels))
    print(("&{: 11.3f}  " * len(labels)).format(*mean_errors) + "\\\\")
    print("\n-> Done!")
    return mean_errors, labels


def run_metrics_all_epochs(exp_dir: str, lidar: bool):
    """
    Computes metrics for ALL epochs
    :param exp_dir: Path to the experiment directory
    :param lidar: Setting to True --> Lidar data (eigen), False --> improved GT depth maps (eigen_benchmark)
    """
    if lidar:
        metrics_file = open(os.path.join(exp_dir, "lidar_metrics.csv"), "a", newline='')
    else:
        metrics_file = open(os.path.join(exp_dir, "kitti_gt_maps_metrics.csv"), "a", newline='')

    metrics_writer = csv.writer(metrics_file, delimiter=',')
    metrics_list = ["epoch"]
    metrics_list.extend(get_labels())
    metrics_writer.writerow(metrics_list)
    weights_folder = os.path.join(exp_dir, "models")
    num_epochs = len(next(os.walk(weights_folder))[1])

    for i in range(num_epochs):
        metrics, metric_labels = run_metrics(exp_dir, i + 1, lidar)
        metrics = [round(num, 3) for num in metrics]
        metrics.insert(0, i + 1)
        metrics_writer.writerow(metrics)

    metrics_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metrics options")
    parser.add_argument("--exp_dir",
                        type=str,
                        help="path to experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    parser.add_argument("--all_epochs",
                        action='store_true',
                        help="Activating this flag runs metrics for all epochs and stores the results in a csv.")
    parser.add_argument("--lidar",
                        action='store_true',
                        help="Activating this flag uses lidar instead of ground truth KITTI depth maps")
    opt = parser.parse_args()

    if opt.all_epochs:
        run_metrics_all_epochs(opt.exp_dir, opt.lidar)
    else:
        run_metrics(opt.exp_dir, opt.epoch, opt.lidar)
