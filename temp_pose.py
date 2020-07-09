from __future__ import division
import os
from use_existingmodels import test_pose_model, test_depth_model
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch

from kitti_utils import (
    compute_image_from_velodyne_matrices, load_lidar_points,
    get_camera_intrinsic_dict, get_relative_pose
)

from overlay_lidar_utils import (
     generate_lidar_point_coord_camera_image, plot_lidar_on_image, plot_point_hue_on_image
)
from compute_photometric_error_utils import (
    color_target_points_with_source, project_points_on_image, filter_to_plane, filter_to_fov,
    reproject_source_to_target, plot_sparse_img_and_surrounding_lidar,
    calc_photo_error_velo
)

test_depth_model('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000011.png', 'mono_640x192', output_path='data', display_result = True, no_filesave = False)
# Path names
calib_path = "data/kitti_example/2011_09_26"
scene_path = os.path.join(calib_path, "2011_09_26_drive_0048_sync")

target = 11
source = 21

img_tgt = np.array(Image.open(os.path.join(scene_path, f"image_02/data/{target:010}.png")))
img_src = np.array(Image.open(os.path.join(scene_path, f"image_02/data/{source:010}.png")))
intrinsics = get_camera_intrinsic_dict(calib_path)["stereo_left"]

pose = get_relative_pose(scene_path, target, source)

velo2cam = compute_image_from_velodyne_matrices(calib_path)["stereo_left"]
tgt_intrinsic = get_camera_intrinsic_dict(calib_path)["stereo_left"]

target_velodyne = load_lidar_points(os.path.join(scene_path, f"velodyne_points/data/{target:010}.bin"))
orig_colors = np.copy(target_velodyne[:, 3])
target_velodyne[:, 3] = 1

tgt_coord = filter_to_fov(filter_to_plane(project_points_on_image(target_velodyne, velo2cam)), img_tgt.shape)

pose_mat = test_pose_model(os.path.join(scene_path, f"image_02/data/{target:010}.png"),os.path.join(scene_path, f"image_03/data/{source:010}.png"),  'mono_640x192')

pose_mat = np.matmul(intrinsics, pose_mat[:3,:])
 
fig = plt.figure(figsize=(32, 9))

plot_lidar_on_image(img_tgt, tgt_coord, fig, 1)


tgt_coord = tgt_coord[:,:3]
tgt_cam = np.dot(np.linalg.inv(tgt_intrinsic), tgt_coord.T).T

rot, tr = pose_mat[:,:3], pose_mat[:3,-1:]
pcoords = np.dot(rot, tgt_cam.T).T[:,:3]
p_trans=np.zeros(np.shape(pcoords))
for i in range(np.shape(pcoords)[0]):
    p_trans[i] = pcoords[i]+tr[0]
print(p_trans)
p_trans = filter_to_fov(p_trans, img_tgt.shape)
print(tgt_coord)
print(p_trans)
plot_lidar_on_image(img_src, p_trans, fig, 2)
