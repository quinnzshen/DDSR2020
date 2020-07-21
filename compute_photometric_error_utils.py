import numpy as np
from matplotlib import pyplot as plt


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


def calc_photo_error(tgt_image, color_points):
    """
    Calculates photometric error given the target image and coordinates of the projected points and their colors
    :param [np.ndarray] tgt_image: Shape of [H, W, 3], the target image to be compared to
    :param [np.ndarray] color_points: The projected point information, in format [x, y, depth, 1, R, G, B]
    :return [np.ndarray]: Float array of shape [H, W], where each value is the photometric error at that position if
    there is was a color point there. If not, that position is just 0
    """
    pixel_error = np.zeros((tgt_image.shape[0], tgt_image.shape[1]), dtype=np.float32)
    pixel_error[color_points[:, 1], color_points[:, 0]] = np.sqrt(np.sum(np.square(color_points[:, 4:] - tgt_image[color_points[:, 1], color_points[:, 0]]), axis=1))
    return pixel_error


def calc_photo_error_velo(tgt_image, color_points):
    """
    Calculates photometric error given the target image and coordinates of the projected points and their colors
    :param [np.ndarray] tgt_image: Shape of [H, W, 3], the target image to be compared to
    :param [np.ndarray] color_points: The projected point information, in format [x, y, depth, 1, R, G, B]
    :return [np.ndarray]: Float array of shape [N], where N is the number of color_points. Each value represents the
    photometric error of the associated point in color_points when compared to the image
    """
    velo_error = np.zeros(color_points.shape[0], dtype=np.float32)
    velo_error[:] = np.sqrt(np.sum(np.square(color_points[:, 4:] - tgt_image[color_points[:, 1], color_points[:, 0]]), axis=1))
    return velo_error
