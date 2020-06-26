import numpy as np
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os

from kitti_utils import compute_image_from_velodyne_matrices, read_calibration_file, load_lidar_points, \
    get_camera_intrinsic_dict
from overlay_lidar_utils import generate_lidar_point_coord_camera_image, plot_lidar_on_image
from compute_photometric_error_utils import plot_sparse_img_and_surrounding_lidar
import plotly_utils


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


def get_relative_pose(scene_path, target, source):
    # target coord to source coord
    with open(os.path.join(scene_path, f"oxts/data/{target:010}.txt")) as ft:
        # start index 8
        # end index 10
        datat = ft.readline().split()
        with open(os.path.join(scene_path, f"oxts/data/{source:010}.txt")) as fs:
            datas = fs.readline().split()
            rot = np.array(datas[3:6], dtype=np.float) - np.array(datat[3:6], dtype=np.float)
            velo = (np.array(datat[8:11], dtype=np.float) + np.array(datas[8:11], dtype=np.float)) / 2
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


def test_transform(inarr, t_mat):
    outarr = t_mat @ inarr
    print(outarr)
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


def get_associated_colors(points_on_image, src_image):
    colors = np.zeros(points_on_image.shape, dtype=np.uint8)
    # iterate through points_on_image and store the associated color in colors
    return colors


def color_image(shape, positions, colors):
    img = np.zeros(shape, dtype=np.uint8)
    img.fill(255)
    # iterate through positions and assign respective colors
    return img


def get_transform_velo2cam():
    pass


def get_transform_coord2image():
    pass


def project_points_on_image(velo_points, d):
    coord2image = get_transform_coord2image()
    return velo_points


def plot_source_in_target(velo_points, src_image, pose_mat):
    # transform velo_points into
    velo2cam = get_transform_velo2cam()
    velo_points = velo_points @ velo2cam.T @ pose_mat.T
    colors = get_associated_colors(project_points_on_image(velo_points, "source"), src_image)

    pixel_positions = project_points_on_image(velo_points, "target")

    out_image = color_image(src_image.shape, pixel_positions, colors)

    # Do something with the image

    pass


def plot_lidar_3d(lidar, orig):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=lidar[:, 0],
                               y=lidar[:, 1],
                               z=lidar[:, 2],
                               mode='markers',
                               marker=dict(size=1, color=orig, colorscale='Viridis'),
                               name='lidar')
                  )

    plotly_utils.setup_layout(fig)
    fig.show()


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
        'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000001.bin')

    print(lidar_point_coord_velodyne.shape)

    orig_colors = np.copy(lidar_point_coord_velodyne[:, 3])
    lidar_point_coord_velodyne[:, 3] = 1
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    print(lidar_point_coord_velodyne)

    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(velo2cam)
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    rel_pose = get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", 1, 0)
    # rel_pose = imu2velo @ get_relative_pose(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", 1, 0)

    lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(rel_pose)
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    lidar_point_coord_velodyne = lidar_point_coord_velodyne @ velo2cam.T
    # lidar_point_coord_velodyne = lidar_point_coord_velodyne @ np.transpose(velo2cam)
    # plot_lidar_3d(lidar_point_coord_velodyne, orig_colors)

    print(lidar_point_coord_velodyne)
    print(orig_colors)

    intr = get_camera_intrinsic_dict(r"data\kitti_example\2011_09_26").get("stereo_left")

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
    image = mpimg.imread('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000000.png')

    filtered = lidar_point_coord_camera_image[filt, :3]
    colors = np.zeros(filtered.shape, dtype=np.float32)
    for i in range(colors.shape[0]):
        colors[i] = image[filtered[i, 1], filtered[i, 0]]
        # print(image[filtered[i, 0], filtered[i, 1]])

    plot_lidar_on_image(image, lidar_point_coord_camera_image[filt])

    plot_sparse_img_and_surrounding_lidar(lidar_point_coord_camera_image, filtered, colors)

    orig_points = load_lidar_points(
        'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000001.bin')

    camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    lidar_point_coord_camera_image = generate_lidar_point_coord_camera_image(
        orig_points, camera_image_from_velodyne, 1242, 375)[:, :3]
    image = mpimg.imread('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000000.png')

    plot_lidar_on_image(image, lidar_point_coord_camera_image)
    print(rel_pose)
    print("hi")

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
