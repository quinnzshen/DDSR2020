from loss import process_depth
from dataloader import KittiDataset
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from kitti_utils import get_relative_pose

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
height = 384
width = 1280

disp = np.load('data/0000000002_disp.npy')
disp = torch.from_numpy(disp)
disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
disp = 31.257 / disp

_, depths = disp_to_depth(disp, 0.1, 100)
depths=disp
dataset = KittiDataset.init_from_config("configs/kitti_dataset.yml")

target_image =  torch.cat([F.interpolate((torch.tensor(
dataset[2]["stereo_left_image"].transpose(2, 0, 1),
dtype=torch.float32).unsqueeze(0)), [height, width], mode="bilinear", align_corners=False)])

stereo_images = torch.cat([F.interpolate((torch.tensor(
dataset[2]["stereo_right_image"].transpose(2, 0, 1),
dtype=torch.float32).unsqueeze(0)), [height, width], mode="bilinear", align_corners=False)])

temporal_forward_images = torch.cat([F.interpolate((torch.tensor(
dataset[2]["nearby_frames"][1]["camera_data"]["stereo_left_image"].transpose(2, 0, 1),
dtype=torch.float32).unsqueeze(0)), [height, width], mode="bilinear",
                                               align_corners=False)])

sources = torch.stack((stereo_images, temporal_forward_images))

# Poses
tgt_poses = torch.cat(
[torch.tensor(dataset[2]["pose"], dtype=torch.float32).unsqueeze(0)])
temporal_forward_poses = torch.cat([torch.tensor(dataset[2]["nearby_frames"][1]["pose"],
                                              dtype=torch.float32).unsqueeze(0)])
# Relative Poses
rel_pose_stereo = torch.cat(
[torch.tensor(dataset[2]["rel_pose_stereo"], dtype=torch.float32).unsqueeze(0)])
rel_pose_forward = torch.matmul(torch.inverse(tgt_poses), temporal_forward_poses)

rel_pose_forward2 = torch.tensor(get_relative_pose('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/', 2, 1)).unsqueeze(0)

poses = torch.stack((rel_pose_stereo, rel_pose_forward))
# Intrinsics
tgt_intrinsics = torch.cat([torch.tensor(dataset[2]["intrinsics"]["stereo_left"],
                                     dtype=torch.float32).unsqueeze(0)])
src_intrinsics_stereo = torch.cat([torch.tensor(dataset[2]["intrinsics"]["stereo_right"],
                                            dtype=torch.float32).unsqueeze(0)])

# Adjust intrinsics based on input size
for i in range(0, 1):
    tgt_intrinsics[i][0] = tgt_intrinsics[i][0] * (width / 1242)
    tgt_intrinsics[i][1] = tgt_intrinsics[i][1] * (height / 375)
    src_intrinsics_stereo[i][0] = src_intrinsics_stereo[i][0] * (width / 1242)
    src_intrinsics_stereo[i][1] = src_intrinsics_stereo[i][1] * (height / 375)
src_intrinsics = torch.stack((src_intrinsics_stereo, tgt_intrinsics))

reprojected, mask = process_depth(sources, depths, poses, tgt_intrinsics, src_intrinsics,
                              (height, width))
plt.imshow(target_image[0].permute(1,2,0)/255)
plt.figure()
plt.imshow(reprojected[0,0].permute(1,2,0)/255)
plt.figure()
plt.imshow(reprojected[1,0].permute(1,2,0)/255)

