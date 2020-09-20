import argparse
from collate import Collator
from kitti_dataset import KittiDataset
import numpy as np
import os
import cv2
import torch
import time
from torch.utils.data import DataLoader
import yaml
from third_party.DensenetEncoder import DensenetEncoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from third_party.monodepth2.layers import disp_to_depth
from fpn import FPN

cv2.setNumThreads(0)

STEREO_SCALE_FACTOR = 5.4
# Sets N bins for metrics, from 0 -> BINS[0], BINS[1] -> BINS[2}, etc.
BINS = [25, 50, 75, 100]


def get_labels():
    """
    Gets the lables for the metrics
    :return [list]: List of strings representing the respective metrics
    """
    labels = ["metric_time", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    for i in BINS:
        labels.extend(["abs_rel_" + str(i), "a1_" + str(i)])
    return labels


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths. Taken from Monodepth2
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


def run_metrics(log_dir, epoch, eigen):
    """Computes metrics based on a specified directory containing a config and an epoch number. Adapted from Monodepth2
    :param [str] log_dir: Path to the config directory that the model was trained on
    :param [int] epoch: Epoch number corresponding to the model that metrics will be evaluated on
    :param [bool] eigen: Setting to True --> eigen (Lidar data), False --> improved GT maps
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIN_DEPTH = 0.001
    MAX_DEPTH = 80

    # Load data from config
    config_path = os.path.join(log_dir, "config.yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)
    
    if eigen:
        dataset = KittiDataset.init_from_config(config["lidar_test_config_path"])
    else:    
        dataset = KittiDataset.init_from_config(config["gt_depthmap_test_config_path"])
        
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=False, collate_fn=Collator(config["height"], config["width"]), num_workers=config["num_workers"])
    
    if config.get("use_densenet"):
        models = {"depth_encoder": DensenetEncoder(config["densenet_layers"], False)}
    else:
        models = {"depth_encoder": ResnetEncoder(config["resnet_layers"], False)}
    decoder_num_ch = models["depth_encoder"].num_ch_enc
    
    if config.get("use_fpn"):
        models["fpn"] = FPN(decoder_num_ch)
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

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        dims[1], dims[0]))
    start_metric_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device).float()
            if config.get("use_fpn"):
                output = models["depth_decoder"](models["fpn"](models["depth_encoder"](inputs)))
            else:
                output = models["depth_decoder"](models["depth_encoder"](inputs))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], config["min_depth"], config["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    
    if eigen == True:
            gt_path = os.path.join(config["gt_path"], "gt_eigen_lidar.npz")
    else:
            gt_path = os.path.join(config["gt_path"], "gt_depths.npz")

    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

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

        if eigen == True:
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
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.nanmean(errors, 0).tolist()
    mean_errors.insert(0, total_metric_time)

    print("\n  " + ("{:>11} | " * len(labels)).format(*labels))
    print(("&{: 11.3f}  " * len(labels)).format(*mean_errors) + "\\\\")
    print("\n-> Done!")
    return mean_errors, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metrics options")
    parser.add_argument("--log_dir",
                        type=str,
                        help="path to experiment directory")
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch number")
    parser.add_argument("--eigen",
                        type=bool,
                        help="determines whether to use eigen lidar or kitti maps",
                        default=False)
    opt = parser.parse_args()
    run_metrics(opt.log_dir, opt.epoch, opt.eigen)