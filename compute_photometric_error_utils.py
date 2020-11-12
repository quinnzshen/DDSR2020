import numpy as np
from matplotlib import pyplot as plt
import torch


SURROUNDING_LIDAR_COLOR = np.array([[.75, .75, .75]])


def rel_pose_from_rotation_matrix_translation_vector(relative_translation: torch.Tensor, relative_rotation: torch.Tensor) -> torch.Tensor:
    """
    This function computes the relative pose matrix that relates the positions of the target and source cameras.
    :param relative_translation: [3, 1] vector representing the relative translation between the camera that
    captured the source image and the camera that captured the target image.
    :param relative_rotation: [3, 3] matrix representing the relative rotation between the camera that
    captured the source image and the camera that captured the target image.
    :return: Tensor of shape [4, 4] that relates the positions of the target and source cameras.
    """
    pose = torch.eye(4)
    pose[:3, :3] = relative_rotation
    pose[:3, 3] = relative_translation
    return pose


def reproject_source_to_target(tgt_intrinsic: np.ndarray, src_intrinsic: np.ndarray,
                               lidar_point_coord_camera_image_tgt: np.ndarray, relative_pose: np.ndarray) -> tuple:
    """
    This function computes which pixels in the source image a given set of target pixels map to.
    :param tgt_intrinsic: [3, 3] intrinsic matrix for camera capturing target image.
    :param src_intrinsic: [3, 3] intrinsic matrix for camera capturing source image.
    :param lidar_point_coord_camera_image_tgt: [N, 3] contains target lidar points on target image plane, each row is format [X, Y, depth]
    :param relative_pose: [4, 4] relates the positions of the target and source cameras.
    :return: Array of shape [N, 2] containing coordinates for pixels in the target image and array of shape [N, 2]
    containing coordinates for pixels in the source image that those target pixels map to.
    """
    # Get depth values for image frame target lidar points.
    depth = lidar_point_coord_camera_image_tgt[:, 2].reshape(-1, 1)
    # Homogenize image frame target lidar point coordinates
    lidar_point_coord_camera_image_tgt = np.hstack((lidar_point_coord_camera_image_tgt[:, :2], np.ones((len(lidar_point_coord_camera_image_tgt), 1))))
    # Project target lidar points from image frame into world frame.
    lidar_point_coord_world = lidar_point_coord_camera_image_tgt @ np.linalg.inv(tgt_intrinsic).T * depth
    # Homogenize world frame lidar point coordinates.
    lidar_point_coord_world = np.hstack((lidar_point_coord_world, np.ones((len(lidar_point_coord_world), 1))))
    # Project world frame lidar points into source image frame
    lidar_point_coord_camera_image_src = (lidar_point_coord_world @ relative_pose.T)[:, :3] @ src_intrinsic.T
    lidar_point_coord_camera_image_src = lidar_point_coord_camera_image_src[lidar_point_coord_camera_image_src[:, 2] > 0]
    lidar_point_coord_camera_image_src = lidar_point_coord_camera_image_src[:, :2] / lidar_point_coord_camera_image_src[:, 2].reshape(-1, 1)
    # Round image frame target lidar points and image frame source lidar points.
    lidar_point_coord_camera_image_src = np.round(lidar_point_coord_camera_image_src, decimals=0).astype(int)
    lidar_point_coord_camera_image_tgt = lidar_point_coord_camera_image_tgt.astype(int)

    return lidar_point_coord_camera_image_tgt, lidar_point_coord_camera_image_src


def plot_sparse_img_and_surrounding_lidar(front_lidar_points_image_plane: np.ndarray, pixel_coords: np.ndarray, colors: np.ndarray):
    """
    This function sparsely plots an image and the lidar points surrounding it.
    :param front_lidar_points_image_plane: [N, 3] contains the pixel coordinates of all of the lidar points that
    are in front of the velodyne sensor.
    :param pixel_coords: [N, 2] contains the coordinates of the pixels of the image that are to be plotted, each row
    is format [x, y]
    :param colors: [N, 3] each row contains the RGB values for the color of each image pixel that is to be plotted.
    :return: None. Plots image and surrounding lidar points.
    """
    plt.figure(figsize=(40, 7.5))
    # Plot surrounding lidar points.
    plt.scatter(front_lidar_points_image_plane[:, 0], front_lidar_points_image_plane[:, 1], c=SURROUNDING_LIDAR_COLOR, s=7, marker='s')
    # Plot sparse image on top of lidar points.
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=colors, s=7, marker='s')
    plt.axis([-500, 1750, 375, 100])
    plt.show()


def calc_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Calculates the homogeneous transformation matrix given relative rotation and translation
    :param rotation: Shape of [3] containing the relative roll, pitch, and yaw (in radians)
    :param translation: Shape of [3] containing the relative XYZ displacement
    :return: 4x4 matrix that transforms given the relative rotation and translation
    """
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
    ], dtype=np.float32)
