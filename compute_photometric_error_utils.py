import numpy as np
from matplotlib import pyplot as plt

from overlay_lidar_utils import get_associated_colors, color_image


def compute_relative_pose_matrix(relative_translation, relative_rotation):
    """
    This function computes the relative pose matrix that relates the positions of the target and source cameras.
    :param [numpy.array] relative_translation: [3, 1] vector representing the relative translation between the camera that 
    captured the source image and the camera that captured the target image.
    :param [numpy.array] relative_rotation: [3, 3] matrix representing the relative rotation between the camera that 
    captured the source image and the camera that captured the target image.
    :return: numpy.array of shape [4, 4] that relates the positions of the target and source cameras.
    """
    relative_pose = np.hstack((relative_rotation, relative_translation))
    relative_pose = np.vstack((relative_pose, np.array([[0., 0., 0., 1.]])))
    return relative_pose


def reproject_source_to_target(tgt_intrinsic, src_intrinsic, lidar_point_coord_camera_image_tgt, relative_pose):
    """
    This function computes which pixels in the source image a given set of target pixels map to.
    :param [numpy.array] tgt_intrinsic: [3, 3] intrinsic matrix for camera capturing target image.
    :param [numpy.array] src_intrinsic: [3, 3] intrinsic matrix for camera capturing source image.
    :param [numpy.array] lidar_point_coord_camera_image_tgt: [N, 3] contains target lidar points on target image plane, each row is format [X, Y, depth]
    :param [numpy.array] relative_pose: [4, 4] relates the positions of the target and source cameras.
    :return: numpy.array of shape [N, 2] containing coordinates for pixels in the target image and numpy.array of shape [N, 2] 
    containing coordinates for pixels in the source image that those target pixels map to.
    """
    # Get depth values for image frame target lidar points.
    depth = lidar_point_coord_camera_image_tgt[:, 2].reshape(-1, 1)
    # Homogenize image frame target lidar point coordinates
    lidar_point_coord_camera_image_tgt = np.hstack((lidar_point_coord_camera_image_tgt[:, :2], np.ones((len(lidar_point_coord_camera_image_tgt), 1))))
    # Project target lidar points from image frame into world frame.
    lidar_point_coord_world = (np.linalg.inv(src_intrinsic) @ lidar_point_coord_camera_image_tgt.T).T * depth
    # Homogenize world frame lidar point coordinates.
    lidar_point_coord_world = np.hstack((lidar_point_coord_world, np.ones((len(lidar_point_coord_world), 1))))
    # Convert target intrinsic matrix into 3x4.
    tgt_intrinsic = np.hstack((tgt_intrinsic, np.array([[0.], [0.], [0.]])))
    # Project world frame lidar points into source image frame
    lidar_point_coord_camera_image_src = (tgt_intrinsic @ relative_pose @ lidar_point_coord_world.T).T

    lidar_point_coord_camera_image_src = lidar_point_coord_camera_image_src[:, :2] / lidar_point_coord_camera_image_src[:, 2].reshape(-1, 1)
    # Round image frame target lidar points and image frame source lidar points.
    lidar_point_coord_camera_image_src = np.round(lidar_point_coord_camera_image_src, decimals=0).astype(int)
    lidar_point_coord_camera_image_tgt = lidar_point_coord_camera_image_tgt.astype(int)

    return lidar_point_coord_camera_image_tgt, lidar_point_coord_camera_image_src


def plot_sparse_img_and_surrounding_lidar(front_lidar_points_image_plane, pixel_coords, colors):
    """
    This function sparsely plots and image and the lidar points surrounding it.
    :param [numpy.array] front_lidar_points_image_plane: [N, 3] contains the pixel coordinates of all of the lidar points that 
    are in front of the velodyne sensor.
    :param [numpy.array] pixel_coords: [N, 2] contains the coordinates of the pixels of the image that are to be plotted, each row 
    is format [x, y]
    :param [numpy.array] colors: [N, 3] each row contains the RGB values for the color of each image pixel that is to be plotted.
    :return: None. Plots image and surrounding lidar points.
    """
    plt.figure(figsize=(40, 7.5))
    # Plot surrounding lidar points.
    plt.scatter(front_lidar_points_image_plane[:, 0], front_lidar_points_image_plane[:, 1], c=np.array([[.75, .75, .75]]), s=7, marker='s')
    # Plot sparse image on top of lidar points.
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=colors, s=7, marker='s')
    plt.axis([-500, 1750, 375, 100])
    plt.show()


def project_points_on_image(velo_points, coord2image):
    """
    Projects lidar points onto image plane based on given matrix
    :param [np.ndarray] velo_points: Array with at least 4 dimensions in 3D coordinates to be transformed
    :param [np.ndarray] coord2image: Shape of [4, 4] that transforms 3D velodyne coordinates into the camera plane
    :return [np.ndarray]: The same array transformed into camera plane coordinates
    """
    point_on_image = np.copy(velo_points)
    point_on_image[:, :4] = point_on_image[:, :4] @ coord2image.T
    return point_on_image


def filter_to_plane(positions):
    """
    Filters out all points that have a non-positive depth (behind camera) and puts coordinates onto actual camera coordinates
    :param [np.ndarray] positions: Array representing the point positions in the camera plane
    :return [np.ndarray]: Int array of the filtered points on the camera plane
    """
    positions = positions[positions[:, 2] > 0]
    positions[:, :2] = positions[:, :2] / positions[:, 2].reshape(-1, 1)
    return np.around(positions).astype(int)


def filter_to_fov(positions, shape):
    """
    Filters out all points not in the camera FOV
    :param [np.ndarray] positions: Points coordinates in camera plane frame
    :param [tuple] shape: The shape of the camera array (H, W, 3)
    :return [np.ndarray]: Filtered version of the points that lie within the camera FOV
    """
    return positions[
        (positions[:, 1] >= 0) & (positions[:, 1] < shape[0]) & (positions[:, 0] >= 0) & (positions[:, 0] < shape[1])]


def plot_source_in_target(velo_points_tgt, src_image, coord2image, rel_pose_mat):
    """
    Plots the source image into target frame given target velodyne points, relative pose, and a matrix to convert
    velodyne points into camera plane.
    :param [np.ndarray] velo_points_tgt: Shape of [N, 4] which contains the homogenized velodyne coordinates in 3D
    :param [np.ndarray] src_image: Shape of [H, W, 3] which contains the RGB data of the source image (255 scale)
    :param [np.ndarray] coord2image: 4x4 matrix transforming velodyne coordinates into camera plane coordinates
    :param [np.ndarray] rel_pose_mat: 4x4 matrix containing pose information to transform target to source frame
    :return: Nothing, just plots the source image projected into the target frame
    """
    # Expands the given points by one column to fit the point index tracker
    tracked_points = np.empty((velo_points_tgt.shape[0], 5), dtype=velo_points_tgt.dtype)
    tracked_points[:, 4] = np.arange(tracked_points.shape[0])
    # Transforms points into source frame
    tracked_points[:, :4] = velo_points_tgt @ rel_pose_mat.T

    # Projects source frame velodyne points into source image plane and grabs their associated colors
    velo_colors = get_associated_colors(filter_to_fov(filter_to_plane(project_points_on_image(tracked_points, coord2image)), src_image.shape), src_image)

    # Projects the original points into the target camera
    tgt_points_image = project_points_on_image(velo_points_tgt, coord2image)
    tgt_points_color = np.empty((velo_points_tgt.shape[0], 7), dtype=velo_points_tgt.dtype)
    tgt_points_color.fill(np.nan)
    tgt_points_color[:, :4] = tgt_points_image

    # Transfers the colors that points were associated with in the source frame
    tgt_points_color[velo_colors[:, 3], 4:] = velo_colors[:, :3]
    # Filters out points that didn't recieve a color
    tgt_points_color = tgt_points_color[~np.isnan(tgt_points_color[:, 4])]
    # Filters out points not in the camera image
    tgt_points_color = filter_to_fov(filter_to_plane(tgt_points_color), src_image.shape)
    # tgt_points_color = filter_to_plane(tgt_points_color)

    # out_image = color_image(tgt_points_color, src_image.shape)
    # Plots points
    plot_sparse_img_and_surrounding_lidar(filter_to_plane(tgt_points_image), tgt_points_color[:, :4], tgt_points_color[:, 4:] / 255)


def calc_transformation_matrix(rotation, translation):
    """
    Calculates the homogeneous transformation matrix given relative rotation and translation
    :param [np.ndarray] rotation: Shape of [3] containing the relative roll, pitch, and yaw (in radians)
    :param [np.ndarray] translation: Shape of [3] containing the relative XYZ displacement
    :return [np.ndarray]: 4x4 matrix that transforms given the relative rotation and translation
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
