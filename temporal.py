import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

import os

from kitti_utils import compute_image_from_velodyne_matrices, read_calibration_file, load_lidar_points, \
    get_camera_intrinsic_dict, get_relative_pose
from overlay_lidar_utils import generate_lidar_point_coord_camera_image, plot_lidar_on_image, plot_lidar_3d
from compute_photometric_error_utils import plot_sparse_img_and_surrounding_lidar, plot_source_in_target, calc_transformation_matrix


if __name__ == "__main__":
    path = r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync"

    rot = np.deg2rad([45, 0, 0])
    # print(rot)
    t_mat = calc_transformation_matrix(rot, [0, 3, 0])
    # print(t_mat)
    inarr = np.array([
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [3, 3, 3, 1]
    ])
    inarr = np.transpose(inarr)

    # test_transform(inarr, t_mat)

    d = compute_image_from_velodyne_matrices(r"data\kitti_example\2011_09_26")

    # print(d["cam02"], "\n")
    # print(d["cam02"] @ inarr)

    cam2cam = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_cam_to_cam.txt'))

    velo2cam = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3, 1)))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    print(velo2cam)

    imu2velo = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_imu_to_velo.txt'))
    imu2velo = np.hstack((imu2velo['R'].reshape(3, 3), imu2velo['T'].reshape(3, 1)))
    imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1.0])))

    lidar_point_coord_velodyne = load_lidar_points(
        'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000011.bin')

    print(lidar_point_coord_velodyne.shape)

    orig_colors = np.copy(lidar_point_coord_velodyne[:, 3])
    lidar_point_coord_velodyne[:, 3] = 1
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    print(lidar_point_coord_velodyne)

    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(coord2image)
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    rel_pose = get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", 11, 21)
    # rel_pose = imu2velo @ get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", 1, 0)
    intr = np.eye(4)
    intr[:3, :3] = get_camera_intrinsic_dict(r"data\kitti_example\2011_09_26").get("stereo_left")
    print(intr)
    lidar_point_coord_velodyne = lidar_point_coord_velodyne @ rel_pose.T
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    lidar_point_coord_velodyne = lidar_point_coord_velodyne @ velo2cam.T
    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(coord2image)
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    print(lidar_point_coord_velodyne)
    print(orig_colors)

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_02"].reshape(3, 3)
    P_rect = cam2cam["P_rect_02"].reshape(3, 4)
    camera_image_from_velodyne = np.dot(P_rect, R_cam2rect)
    camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
    # intrinsic_matrix = cam2cam["K_02"].reshape(3, 3)
    print(camera_image_from_velodyne)

    # lidar_point_coord_camera_image, filt = generate_lidar_point_coord_camera_image(
    #     lidar_point_coord_velodyne, camera_image_from_velodyne, 1242, 375)

    # Remove points behind velodyne sensor.
    # lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >=0, :]

    # Project points to image plane.
    lidar_point_coord_camera_image = lidar_point_coord_velodyne @ camera_image_from_velodyne.T
    lidar_point_coord_camera_image = lidar_point_coord_camera_image[lidar_point_coord_camera_image[:, 2] > 0]
    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / \
                                            lidar_point_coord_camera_image[:, 2][..., np.newaxis]

    # Round X and Y pixel coordinates to int.
    lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)

    # Create filtered index only inlude those in image field of view.
    filt = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
           (lidar_point_coord_camera_image[:, 0] < 1242) & (lidar_point_coord_camera_image[:, 1] < 375)

    # Load image file.
    src_image = np.array(Image.open('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000021.png'))

    filtered = lidar_point_coord_camera_image[filt, :3]
    colors = np.zeros(filtered.shape, dtype=np.uint8)
    for i in range(colors.shape[0]):
        colors[i] = src_image[filtered[i, 1], filtered[i, 0]]
        # print(image[filtered[i, 0], filtered[i, 1]])
    fig = plt.figure(figsize=(32, 9))

    plot_lidar_on_image(src_image, lidar_point_coord_camera_image[filt], fig, 1)

    # plot_sparse_img_and_surrounding_lidar(lidar_point_coord_camera_image, filtered, colors)

    orig_points = load_lidar_points(
        'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000011.bin')

    camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
    # lidar_point_coord_camera_image = generate_lidar_point_coord_camera_image(
    #     orig_points, camera_image_from_velodyne, 1242, 375)[:, :3]

    lidar_point_coord_camera_image = orig_points @ camera_image_from_velodyne.T
    lidar_point_coord_camera_image = lidar_point_coord_camera_image[lidar_point_coord_camera_image[:, 2] > 0]
    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / \
                                            lidar_point_coord_camera_image[:, 2][..., np.newaxis]

    # Round X and Y pixel coordinates to int.
    lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)

    # Create filtered index only inlude those in image field of view.
    filt = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
           (lidar_point_coord_camera_image[:, 0] < 1242) & (lidar_point_coord_camera_image[:, 1] < 375)

    tgt_image = np.array(Image.open('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000011.png'))

    plot_lidar_on_image(tgt_image, lidar_point_coord_camera_image[filt], fig, 2)
    plt.show()
    print(rel_pose)
    print("hi")

    print(camera_image_from_velodyne.shape)
    plot_source_in_target(orig_points, src_image, camera_image_from_velodyne, rel_pose)

# def calc_transformation_mat(sample_path, idx):
#     """
#     Given the file path to the scene and the frame number within that scene, returns a 4x4 NumPy array containing the
#     translation matrix to convert the LiDAR point coordinates (relative to the sensor) into global coordinates
#     (relative to the starting point), where +x is East, +y is North, and +z is up.
#     :param sample_path: A file_path to a scene within the dataset
#     :param idx: The frame number within the scene
#     :return: 4x4 homogenous translation matrix to convert relative coordinates into continuous coordinates
#     """
#     with open(os.path.join(sample_path, "oxts/data/") + f"{0:010}.txt") as f:
#         line = f.readline().split()
#         orig_coords = np.array(line[:3], dtype=np.float64)
#         if idx == 0:
#             new_coords = np.array(line[:6], dtype=np.float64)
#         else:
#             with open(os.path.join(sample_path, "oxts/data/") + f"{idx:010}.txt") as fi:
#                 new_coords = np.array(fi.readline().split(), dtype=np.float64)
#
#     latlon_orig = np.deg2rad(orig_coords[:2])
#     latlon_new = np.deg2rad(new_coords[:2])
#     sin_rpy = np.sin(new_coords[3:])
#     cos_rpy = np.cos(new_coords[3:])
#
#     # translation matrix
#     return np.array([
#         [
#             cos_rpy[2] * cos_rpy[1],
#             cos_rpy[2] * sin_rpy[1] * sin_rpy[0] - sin_rpy[2] * cos_rpy[0],
#             cos_rpy[2] * sin_rpy[1] * cos_rpy[0] + sin_rpy[2] * sin_rpy[0],
#             calc_lon_dist(latlon_orig[0], latlon_new[0], latlon_orig[1], latlon_new[1])
#         ],
#         [
#             sin_rpy[2] * cos_rpy[1],
#             sin_rpy[2] * sin_rpy[1] * sin_rpy[0] + cos_rpy[2] * cos_rpy[0],
#             sin_rpy[2] * sin_rpy[1] * sin_rpy[0] - cos_rpy[2] * sin_rpy[0],
#             EARTH_RADIUS * (latlon_new[0] - latlon_orig[0])
#         ],
#         [
#             -1 * sin_rpy[1],
#             cos_rpy[1] * sin_rpy[0],
#             cos_rpy[1] * cos_rpy[0],
#             new_coords[2] - orig_coords[2]
#         ],
#         [0, 0, 0, 1],
#     ], dtype=np.float64)
