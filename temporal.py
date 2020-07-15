import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

import os

# from kitti_utils import compute_image_from_velodyne_matrices, read_calibration_file, load_lidar_points, \
#     get_camera_intrinsic_dict, get_relative_pose
from overlay_lidar_utils import generate_lidar_point_coord_camera_image, plot_lidar_on_image
# from compute_photometric_error_utils import plot_sparse_img_and_surrounding_lidar, color_target_points_with_source, calc_photo_error_velo
import plotly_utils


NORMALIZE_DIST = np.sqrt(3) * 255

def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since the midnight of that day
    :param time_string: The string to be converted into nanoseconds
    :return: The number of nanoseconds since midnight
    """
    total = 0
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total


def get_relative_poseO(scene_path, target, source):
    # source frame to target frame
    with open(os.path.join(scene_path, f"oxts/data/{target:010}.txt")) as ft:
        # start index 8
        # end index 10
        datat = ft.readline().split()
        with open(os.path.join(scene_path, f"oxts/data/{source:010}.txt")) as fs:
            datas = fs.readline().split()
            rot = np.array(datas[3:6], dtype=np.float) - np.array(datat[3:6], dtype=np.float)
            velo = (np.array(datas[8:11], dtype=np.float) + np.array(datat[8:11], dtype=np.float)) / 2
    with open(os.path.join(scene_path, "oxts/timestamps.txt")) as time:
        i = 0
        target_time = 0
        source_time = 0
        for line in time:
            if i == target:
                target_time = iso_string_to_nanoseconds(line)
                if source_time:
                    break
            elif i == source:
                source_time = iso_string_to_nanoseconds(line)
                if target_time:
                    break
            i += 1
        delta_time = source_time - target_time

    pos = velo * delta_time / 1E9
    # print(delta_time / 1e9)
    return calc_transformation_matrix(rot, pos)


def calc_transformation_matrix(rotation, translation):
    sin_rot = np.sin(rotation)
    cos_rot = np.cos(rotation)
    return np.array([
        [
            cos_rot[2] * cos_rot[1],
            cos_rot[2] * sin_rot[1] * sin_rot[0] - sin_rot[2] * cos_rot[0],
            cos_rot[2] * sin_rot[1] * cos_rot[0] + sin_rot[2] * sin_rot[0],
            translation[0]
        ],
        [
            sin_rot[2] * cos_rot[1],
            sin_rot[2] * sin_rot[1] * sin_rot[0] + cos_rot[2] * cos_rot[0],
            sin_rot[2] * sin_rot[1] * sin_rot[0] - cos_rot[2] * sin_rot[0],
            translation[1]
        ],
        [
            -1 * sin_rot[1],
            cos_rot[1] * sin_rot[0],
            cos_rot[1] * cos_rot[0],
            translation[2]
        ],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def t_transform(inarr, t_mat):
    outarr = t_mat @ inarr
    # print(outarr)
    # print(np.array([0, 1, 0, 1]) @ np.transpose(t_mat))

    fig = go.Figure(layout=go.Layout(
        scene=dict(camera=dict(eye=dict(x=1.14, y=1.14, z=1.14)),  # the default values are 1.25, 1.25, 1.25
                   xaxis=dict(),
                   yaxis=dict(),
                   zaxis=dict(),
                   aspectmode="cube",  # this string can be 'data', 'cube', 'auto', 'manual'
                   # a custom aspectratio is defined as follows:
                   aspectratio=dict(x=1, y=1, z=1)
                   )))
    fig.add_trace(go.Scatter3d(x=inarr[0],
                               y=inarr[1],
                               z=inarr[2],
                               mode='markers',
                               marker=dict(size=3, color=1, colorscale='Viridis'),
                               name='lidar')
                  )
    fig.add_trace(go.Scatter3d(x=outarr[0],
                               y=outarr[1],
                               z=outarr[2],
                               mode='markers',
                               marker=dict(size=3, color=1, colorscale='Viridis'),
                               name='lidar')
                  )

    # fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
    #                     shared_xaxes=True, shared_yaxes=True)
    # fig.add_trace(go.Scatter3d(x=inarr[0],
    #                            y=inarr[1],
    #                            z=inarr[2],
    #                            mode='markers',
    #                            marker=dict(size=1, color=1, colorscale='Viridis'),
    #                            name='lidar'), row=1, col=1
    #               )
    # fig.add_trace(go.Scatter3d(x=outarr[0],
    #                     y=outarr[1],
    #                     z=outarr[2],
    #                     mode='markers',
    #                     marker=dict(size=1, color=1, colorscale='Viridis'),
    #                     name='lidar'), row=1, col=2
    #               )

    # plotly_utils.setup_layout(fig)
    fig.show()


def t_transform_n(inarr, colors):
    # print(outarr)
    # print(np.array([0, 1, 0, 1]) @ np.transpose(t_mat))

    fig = go.Figure(layout=go.Layout(
        scene=dict(camera=dict(eye=dict(x=1.14, y=1.14, z=1.14)),  # the default values are 1.25, 1.25, 1.25
                   xaxis=dict(),
                   yaxis=dict(),
                   zaxis=dict(),
                   aspectmode="cube",  # this string can be 'data', 'cube', 'auto', 'manual'
                   # a custom aspectratio is defined as follows:
                   aspectratio=dict(x=1, y=1, z=1)
                   )))
    fig.add_trace(go.Scatter3d(x=inarr[0],
                               y=inarr[1],
                               z=inarr[2],
                               mode='markers',
                               marker=dict(size=1.6, color=colors, colorscale='Inferno_r'),
                               name='lidar')
                  )
    fig.show()


if __name__ == "__main__":
    path = r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync"

    rot = np.deg2rad([45, 0, 0])
    tran = np.array([2, 0, 0])
    # print(rot)

    t_mat = calc_transformation_matrix(rot, tran)
    # print(t_mat)
    inarr = np.array([
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ])
    inarr_alt = inarr.copy()
    inarr_alt[:, 0] = -inarr[:, 1]
    inarr_alt[:, 1] = -inarr[:, 2]
    inarr_alt[:, 2] = inarr[:, 0]
    print(inarr_alt)

    inarr = np.transpose(inarr)
    inarr_alt = inarr_alt.T

    t_transform(inarr, t_mat)

    altmat = np.copy(t_mat)
    altmat[:, 3] = np.array([-altmat[1, 3], -altmat[2, 3], altmat[0, 3], 1])

    altmat = calc_transformation_matrix(np.array([-rot[1], -rot[2], rot[0]]), np.array([-tran[1], -tran[2], tran[0]]))

    t_transform(inarr_alt, altmat)



    # cam2cam = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_cam_to_cam.txt'))
    #
    # velo2cam = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_velo_to_cam.txt'))
    # velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3, 1)))
    # velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    #
    # imu2velo = read_calibration_file(os.path.join(r"data\kitti_example\2011_09_26", 'calib_imu_to_velo.txt'))
    # imu2velo = np.hstack((imu2velo['R'].reshape(3, 3), imu2velo['T'].reshape(3, 1)))
    # imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1.0])))
    #
    # lidar_point_coord_velodyne = load_lidar_points(
    #     'data/kitti_example/2011_09_26/2011_09_26_drive_0060_sync/velodyne_points/data/0000000011.bin')
    #
    # orig_colors = np.copy(lidar_point_coord_velodyne[:, 3])
    # lidar_point_coord_velodyne[:, 3] = 1
    #
    # # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(coord2image)
    # # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)
    #
    # rel_pose = get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0060_sync", 11, 21)
    # # rel_pose = imu2velo @ get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0060_sync", 11, 21)
    #
    # bruhp = np.copy(rel_pose)
    # bruhp[:, 3] = np.array([-bruhp[1, 3], -bruhp[2, 3], bruhp[0, 3], 1])
    # print(bruhp)
    # print(velo2cam @ rel_pose, "awefawefawefpoij")
    # print(rel_pose)
    #
    #
    # intr = np.eye(4)
    # intr[:3, :3] = get_camera_intrinsic_dict(r"data\kitti_example\2011_09_26").get("stereo_left")
    #
    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ rel_pose.T
    # # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)
    #
    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ velo2cam.T
    # # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(rel_pose)
    # # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)
    #
    # R_cam2rect = np.eye(4)
    # R_cam2rect[:3, :3] = cam2cam["R_rect_02"].reshape(3, 3)
    # P_rect = cam2cam["P_rect_02"].reshape(3, 4)
    # camera_image_from_velodyne = np.dot(P_rect, R_cam2rect)
    # camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
    # # intrinsic_matrix = cam2cam["K_02"].reshape(3, 3)
    #
    # # lidar_point_coord_camera_image, filt = generate_lidar_point_coord_camera_image(
    # #     lidar_point_coord_velodyne, camera_image_from_velodyne, 1242, 375)
    #
    # # Remove points behind velodyne sensor.
    # # lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >=0, :]
    #
    # # Project points to image plane.
    # lidar_point_coord_camera_image = lidar_point_coord_velodyne @ camera_image_from_velodyne.T
    # lidar_point_coord_camera_image = lidar_point_coord_camera_image[lidar_point_coord_camera_image[:, 2] > 0]
    # lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / \
    #                                         lidar_point_coord_camera_image[:, 2][..., np.newaxis]
    #
    # # Round X and Y pixel coordinates to int.
    # lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)
    #
    # # Create filtered index only inlude those in image field of view.
    # filt = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
    #        (lidar_point_coord_camera_image[:, 0] < 1242) & (lidar_point_coord_camera_image[:, 1] < 375)
    #
    # # Load image file.
    # src_image = np.array(Image.open('data/kitti_example/2011_09_26/2011_09_26_drive_0060_sync/image_02/data/0000000021.png'))
    #
    # filtered = lidar_point_coord_camera_image[filt, :3]
    # colors = np.zeros(filtered.shape, dtype=np.uint8)
    # for i in range(colors.shape[0]):
    #     colors[i] = src_image[filtered[i, 1], filtered[i, 0]]
    #     # print(image[filtered[i, 0], filtered[i, 1]])
    # fig = plt.figure(figsize=(32, 9))
    #
    # plot_lidar_on_image(src_image, lidar_point_coord_camera_image[filt], fig, 1)
    #
    # # plot_sparse_img_and_surrounding_lidar(lidar_point_coord_camera_image, filtered, colors)
    #
    # orig_points = load_lidar_points(
    #     'data/kitti_example/2011_09_26/2011_09_26_drive_0060_sync/velodyne_points/data/0000000011.bin')
    # orig_points[:, 3] = 1
    #
    # camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    # camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
    # # lidar_point_coord_camera_image = generate_lidar_point_coord_camera_image(
    # #     orig_points, camera_image_from_velodyne, 1242, 375)[:, :3]
    #
    # lidar_point_coord_camera_image = orig_points @ camera_image_from_velodyne.T
    # lidar_point_coord_camera_image = lidar_point_coord_camera_image[lidar_point_coord_camera_image[:, 2] > 0]
    # lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / \
    #                                         lidar_point_coord_camera_image[:, 2][..., np.newaxis]
    #
    # # Round X and Y pixel coordinates to int.
    # lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)
    #
    # # Create filtered index only inlude those in image field of view.
    # filt = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
    #        (lidar_point_coord_camera_image[:, 0] < 1242) & (lidar_point_coord_camera_image[:, 1] < 375)
    #
    # tgt_image = np.array(Image.open('data/kitti_example/2011_09_26/2011_09_26_drive_0060_sync/image_02/data/0000000011.png'))
    #
    # plot_lidar_on_image(tgt_image, lidar_point_coord_camera_image[filt], fig, 2)
    # plt.show()
    #
    #
    # color_points, _ = color_target_points_with_source(orig_points, src_image, camera_image_from_velodyne, rel_pose)
    # errors = calc_photo_error_velo(tgt_image, color_points) / NORMALIZE_DIST
    # colors = np.ones((errors.shape[0], 3), dtype=np.float32)
    # colors[:, 0] = errors
    # fig = plt.figure(figsize=(32, 9))
    # colors = mcolor.hsv_to_rgb(colors)
    #
    # # Show grayscale image.
    # fig.add_subplot(1, 1, 1)
    # plt.imshow(tgt_image, cmap='Greys_r')
    #
    # # Plot lidar points.
    # plt.scatter(color_points[:, 0], color_points[:, 1], c=colors, s=5)
    # plt.show()


# APJOFWIEJOIPWAFEPJOIWAEFIOPAWEJFAEW




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
