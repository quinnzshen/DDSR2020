import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loss import GenerateReprojections
from kitti_dataset import KittiDataset
import numpy as np
from depth_completion_utils import create_depth_map_from_nearest_lidar_point
import os
from compute_photometric_error_utils import *
from kitti_utils import *
import PIL.Image as pil
import cv2

device = "cpu"
processd = GenerateReprojections(384, 1280, 1)
calibration_dir = 'data/kitti_example/2011_09_26'
SCENE_PATH = r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync"
TEST_CONFIG_PATH = "configs/datasets/kitti_dataset.yml"
dataset = KittiDataset.init_from_config(TEST_CONFIG_PATH)
h = 384
w = 1280
gt_depth_path = "data/gt_example/2011_09_26_drive_0048_sync/proj_depth/groundtruth/image_02/0000000008.png"
gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
gt_depth = cv2.resize(gt_depth, (h, w))
gt_depth = torch.from_numpy(gt_depth)

target = F.interpolate(dataset[6]["stereo_left_image"].permute(2, 0, 1).unsqueeze(0).float(), [h, w], mode="bilinear", align_corners=False)
stereo = dataset[6]["stereo_right_image"].permute(2, 0, 1).unsqueeze(0).float().to(device)

sources=[]
sources.append(F.interpolate(stereo, [h, w], mode="bilinear", align_corners=False))
sources = torch.stack(sources, dim=0)

rel_pose_stereo = dataset[6]["rel_pose_stereo"].unsqueeze(0).to(device)
rel_pose_stereo = torch.eye(4, dtype=torch.float).unsqueeze(0)
rel_pose_stereo[0, 0, 3] = -0.54

poses = []
poses.append(rel_pose_stereo)
poses = torch.stack(poses, dim=0)

left_intrinsic = dataset[6]["intrinsics"]["stereo_left"].unsqueeze(0).to(device)
right_intrinsic = dataset[6]["intrinsics"]["stereo_right"].unsqueeze(0).to(device)
shape = dataset[6]["stereo_left_image"].shape

# Remember to scale intrinsic matrices by the image dimension (by default they are calibrated to 1242x375)
left_intrinsic[:, 0] = left_intrinsic[:, 0] * (1280 / shape[1])
left_intrinsic[:, 1] = left_intrinsic[:, 1] * (384 / shape[0])
right_intrinsic[:, 0] = right_intrinsic[:, 0] * (1280 / shape[1])
right_intrinsic[:, 1] = right_intrinsic[:, 1] * (384 / shape[0])

src_intrinsics = []
src_intrinsics.append(right_intrinsic)
src_intrinsics = torch.stack(src_intrinsics)

plt.imshow(gt_depth)
plt.figure()
plt.show()

out_imgs = processd(sources, gt_depth, poses, left_intrinsic, src_intrinsics, 1)

plt.imshow(target[0].permute(1, 2, 0) / 255)
plt.show()
plt.figure()
plt.imshow(out_imgs[0][0].permute(1, 2, 0) / 255)