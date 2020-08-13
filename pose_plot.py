import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch

import os

from kitti_utils import (
    compute_image_from_velodyne_matrices, load_lidar_points,
    get_camera_intrinsic_dict, get_relative_pose_between_consecutive_frames, get_pose
)
from overlay_lidar_utils import (
     generate_lidar_point_coord_camera_image, plot_lidar_on_image, plot_point_hue_on_image
)
from compute_photometric_error_utils import (
    color_target_points_with_source, project_points_on_image, filter_to_plane, filter_to_fov,
    reproject_source_to_target, plot_sparse_img_and_surrounding_lidar,
    calc_photo_error_velo
)

start = 0
end = 100
calib_path = "data/kitti_example/2011_09_28"
scene_path = os.path.join(calib_path, "2011_09_28_drive_0001_sync")
abs_coord = np.empty((end-start, 3))
rel_coord = np.empty((end-start, 3))
for i in range(start, end):
    rel_pose = np.zeros((4, 4))
    for idx in range(0, i):
        rel_pose += get_relative_pose_between_consecutive_frames(scene_path, idx, idx + 1).numpy()
    abs_pose = get_pose(scene_path, i)

    abs_coord[i] = abs_pose[:,3][:3]
    rel_coord[i] = rel_pose[:,3][:3]
    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(abs_coord[:,0], abs_coord[:,1], abs_coord[:,2])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(rel_coord[:,0], rel_coord[:,1], rel_coord[:,2])