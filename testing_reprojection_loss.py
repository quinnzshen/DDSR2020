import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loss import process_depth
from kitti_dataset import KittiDataset
import numpy as np
from depth_completion_utils import create_depth_map_from_nearest_lidar_point

def disp_to_depth(disp, min_depth, max_depth):
    """
    Converts network's sigmoid output into depth prediction (from monodepth 2 repo)
    The formula for this conversion is given in the 'additional considerations'
    section of the paper
    :param [tensor] disp: The disparity map outputted by the network
    :param [int] min_depth: The minimum depth value
    :param [int] max_depth: The maximum depth value
    """

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
device = "cpu"

calibration_dir = 'data/kitti_example/2011_09_26'
SCENE_PATH = r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync"
TEST_CONFIG_PATH = "configs/kitti_dataset.yml"
dataset = KittiDataset.init_from_config(TEST_CONFIG_PATH)

h = 384
w = 1280

disp = np.load("data/disp_example/0000000002_disp.npy")
disp = torch.from_numpy(disp)
disp = F.interpolate(disp, [h, w], mode="bilinear", align_corners=False)

target = F.interpolate(dataset[1]["stereo_left_image"].permute(2, 0, 1).unsqueeze(0).float(), [h, w], mode="bilinear", align_corners=False)

tmp_forward = dataset[1]["nearby_frames"][1]["camera_data"]["stereo_left_image"].permute(2, 0, 1).unsqueeze(0).float().to(device)
tmp_backward = dataset[1]["nearby_frames"][-1]["camera_data"]["stereo_left_image"].permute(2, 0, 1).unsqueeze(0).float().to(device)
stereo = dataset[1]["stereo_right_image"].permute(2, 0, 1).unsqueeze(0).float().to(device)

sources = []
sources.append(F.interpolate(stereo, [h, w], mode="bilinear", align_corners=False))
sources.append(F.interpolate(tmp_forward, [h, w], mode="bilinear", align_corners=False))
sources.append(F.interpolate(tmp_backward, [h, w], mode="bilinear", align_corners=False))
sources = torch.stack(sources, dim=0)

rel_pose_stereo = dataset[1]["rel_pose_stereo"].unsqueeze(0).to(device)
rel_pose_forward = dataset[1]["nearby_frames"][1]["pose"].unsqueeze(0).to(device)
rel_pose_backward = dataset[1]["nearby_frames"][-1]["pose"].unsqueeze(0).to(device)

poses = []
poses.append(rel_pose_stereo)
poses.append(rel_pose_forward)
poses.append(rel_pose_backward)
poses = torch.stack(poses, dim=0)

depth = torch.from_numpy(create_depth_map_from_nearest_lidar_point(dataset[1]["lidar_point_coord_velodyne"].numpy(), 384, 1280)).unsqueeze(0).unsqueeze(0).float().to(device)
#_, depth = disp_to_depth(disp, 0.1, 100)
#depth = 31 * depth

left_intrinsic = dataset[1]["intrinsics"]["stereo_left"].unsqueeze(0).to(device)
right_intrinsic = dataset[1]["intrinsics"]["stereo_right"].unsqueeze(0).to(device)

shape = dataset[1]["stereo_left_image"].shape

# Remember to scale intrinsic matrices by the image dimension (by default they are calibrated to 1242x375)
left_intrinsic[:, 0] = left_intrinsic[:, 0] * (1280 / shape[1])
left_intrinsic[:, 1] = left_intrinsic[:, 1] * (384 / shape[0])
right_intrinsic[:, 0] = right_intrinsic[:, 0] * (1280 / shape[1])
right_intrinsic[:, 1] = right_intrinsic[:, 1] * (384 / shape[0])

src_intrinsics = []
src_intrinsics.append(right_intrinsic)
src_intrinsics.append(left_intrinsic)
src_intrinsics.append(left_intrinsic)
src_intrinsics = torch.stack(src_intrinsics)

plt.imshow(depth[0][0])
plt.figure()
plt.show()
out_imgs = process_depth(sources, depth, poses, left_intrinsic, src_intrinsics, (384, 1280))
plt.imshow(target[0].permute(1, 2, 0) / 255)
plt.show()

#plt.imshow(sources[0][0].permute(1, 2, 0) / 255)
plt.figure()
plt.imshow(out_imgs[0, 0].permute(1, 2, 0) / 255)
plt.show()

#plt.imshow(sources[1][0].permute(1, 2, 0) / 255)
plt.figure()
plt.imshow(out_imgs[1, 0].permute(1, 2, 0) / 255)
plt.show()

#plt.imshow(sources[2][0].permute(1, 2, 0) / 255)
plt.figure()
plt.imshow(out_imgs[2, 0].permute(1, 2, 0) / 255)
plt.show()
