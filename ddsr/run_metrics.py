from torch.utils.data import DataLoader
from kitti_dataset import KittiDataset
from collate import Collator
import numpy as np
import os
import cv2
import torch
import yaml
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from third_party.monodepth2.layers import disp_to_depth

cv2.setNumThreads(0)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def run_metrics(config_path, epoch):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIN_DEPTH = 0.1
    MAX_DEPTH = 100
    
    # Load data from config
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)
            
    weights_folder = os.path.join("models", config["log_path"], f'weights_{epoch-1}')
    print("-> Loading weights from {weights_folder}")
    encoder_path = os.path.join(weights_folder, "resnet_encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth_decoder.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = KittiDataset.init_from_config(config["test_config_path"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=False, collate_fn=Collator(config["height"], config["width"]), num_workers=config["num_workers"])

    encoder = ResnetEncoder(config["encoder_layers"], False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["stereo_left_image"].to(device).float()

            output = depth_decoder(encoder(inputs))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join(config["gt_path"], "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio
            
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

run_metrics("configs/basic_model.yml", 5)