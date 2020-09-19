import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolor
from plotly import graph_objects as go

import plotly_utils

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


def get_associated_colors(points_on_image, src_image):
    """
    Given point coordinates on the image plane, associates those points with the color of its position on src_image
    :param [np.ndarray] points_on_image: Shape of [N, 5], where [:, 0 & 1] are the xy coordinates, [:, 2] is the depth, [:, 3] is 1, and [:, 4] is the point number
    :param [np.ndarray] src_image: Shape of [H, W, 3] representing the image in RGB format
    :return [np.ndarray]: Shape of [N, 4], where [:, 0:3] are the color values and [:, 3] is the associated point number
    """
    src_colors = np.zeros((points_on_image.shape[0], 4), dtype=points_on_image.dtype)
    src_colors[:, :3] = src_image[points_on_image[:, 1], points_on_image[:, 0]]
    # Copies over point indices
    src_colors[:, 3] = points_on_image[:, 4]
    return src_colors


def color_image(color_points, shape):
    """
    Colors a blank image with given positions and colors
    :param [np.ndarray] color_points: Shape of [N, 7] representing point coordinates on the image and their colors. [:, :2] is the xy coordinates and [:, 4:] are the color values
    :param [tuple] shape: The shape of the image to be created
    :return [np.ndarray]: The new blank image with pixels painted in based on color_points
    """
    img = np.zeros(shape, dtype=np.uint8)
    # Makes it all white
    img.fill(255)

    img[color_points[:, 1], color_points[:, 0]] = color_points[:, 4:]
    return img


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


def color_target_points_with_source(velo_points_tgt, src_image, coord2image, rel_pose_mat):
    """
    Plots the source image into target frame given target velodyne points, relative pose, and a matrix to convert
    velodyne points into camera plane.
    :param [np.ndarray] velo_points_tgt: Shape of [N, 4] which contains the homogenized velodyne coordinates in 3D
    :param [np.ndarray] src_image: Shape of [H, W, 3] which contains the RGB data of the source image (255 scale)
    :param [np.ndarray] coord2image: 4x4 matrix transforming velodyne coordinates into camera plane coordinates
    :param [np.ndarray] rel_pose_mat: 4x4 matrix containing pose information to transform target to source frame
    :return [tuple]: A tuple of 2 elements containing the target lidar points with color information in camera image,
    and all target lidar points with a positive depth in camera image, respectively
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

    return tgt_points_color, filter_to_plane(tgt_points_image)

    # Plots points
    # plot_sparse_img_and_surrounding_lidar(filter_to_plane(tgt_points_image), tgt_points_color[:, :4], tgt_points_color[:, 4:] / 255)


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

def generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne,
                                            im_height, im_width):
    """
    This function removes the lidar points that are not in the image plane, rounds x/y pixel values for lidar points, 
    and projects the lidar points onto the image plane
    :param [numpy.array] lidar_point_coord_velodyne: [N, 3], matrix of lidar points, each row is format [X, Y, Z]
    :param [numpy.array] camera_image_from_velodyne: [4, 4], converts 3D lidar points to 2D image plane
    :param [int] im_width: width of image in pixels
    :param [int] im_height: height of image in pixels
    :return: numpy.array of shape [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    """
    # Based on code from monodepth2 repo.

    # Add right column of ones to lidar_point_coord_velodyne.
    lidar_point_coord_velodyne = np.hstack((lidar_point_coord_velodyne, np.ones((len(lidar_point_coord_velodyne), 1))))
    # Remove points behind velodyne sensor.
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >= 0, :]

    # Project points to image plane.
    lidar_point_coord_camera_image = lidar_point_coord_velodyne @ camera_image_from_velodyne.T

    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / \
                                            lidar_point_coord_camera_image[:, 2][..., np.newaxis]

    # Round X and Y pixel coordinates to int.
    lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)

    # Create filtered index only inlude those in image field of view.
    filtered_index = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
                     (lidar_point_coord_camera_image[:, 0] < im_width) & (
                             lidar_point_coord_camera_image[:, 1] < im_height)

    return lidar_point_coord_camera_image, filtered_index


def plot_lidar_on_image(image, lidar_point_coord_camera_image, fig, ind):
    """
    This function plots lidar points on the image with colors corresponding to their depth(higher hsv hue val = further away) 
    :param [numpy.array] image: [H, W], contains image data
    :param [numpy.array] lidar_point_coord_camera_image: [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    :param [plt.Figure] fig: The figure for the image to be plotted on
    :param [int] ind: The plot number in the figure
    :return: None. Plots image w/ lidar overlay.
    """
    # Normalize depth values.
    norm_depth = normalize_depth(lidar_point_coord_camera_image[:, :3])
    plot_point_hue_on_image(image, norm_depth[:, :2], norm_depth[:, 2], fig, ind)


def normalize_depth(lidar_point_coord_camera_image):
    """
    This function normalizes the depth values in lidar_point_coord_camera_image so they are between 0 and 1, inclusive 
    :param [numpy.array] lidar_point_coord_camera_image: [N, 3] contains lidar points on image plane, each row is format [X, Y, depth]
    :return: numpy.array of shape [N, 3] containing the normalized lidar points on image plane, each row is format [X, Y, normalized depth] 
            (normalized depth is between 0 and 1, inclusive)
    """
    lidar_point_coord_camera_image = lidar_point_coord_camera_image.astype('float32')
    max = lidar_point_coord_camera_image[:, 2].max()
    min = lidar_point_coord_camera_image[:, 2].min()
    lidar_point_coord_camera_image[:, 2] = (lidar_point_coord_camera_image[:, 2] - min) / (max - min)
    return lidar_point_coord_camera_image


def plot_lidar_3d(lidar, colors):
    """
    Function to help visualize lidar points by plotting them in a point cloud
    :param [np.ndarray] lidar: The XYZ of the lidar points
    :param [np.ndarray] colors: Shape of [N], usually the reflectivity
    :return: No return, only plots the points
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=lidar[:, 0],
                               y=lidar[:, 1],
                               z=lidar[:, 2],
                               mode='markers',
                               marker=dict(size=1, color=colors, colorscale='Viridis'),
                               name='lidar')
                  )

    plotly_utils.setup_layout(fig)
    fig.show()


def plot_point_hue_on_image(img, positions, scale, fig, ind):
    """
    Plots points on an image with a chosen color based on the corresponding scale value. 0 < scale < 1, and if scale
    is 0, then the point is red, if scale is 1, it is blue, and so forth.
    :param [np.ndarray] img: Shape of [H, W, 3], the image background the points are plotted on
    :param [np.ndarray] positions: Shape of [N, 2] or more, representing the positions of the points
    :param [np.ndarray] scale: Shape of [N], the hue of each point
    :param [plt.Figure] fig: The plt figure
    :param [int] ind: Which subplot it is a part of
    :return: Nothing, just creates an additional plt figure
    """

    colors = np.ones((scale.shape[0], 3), dtype=np.float32)
    colors[:, 0] = scale * 240 / 360
    colors = mcolor.hsv_to_rgb(colors)

    # Show grayscale image.
    fig.add_subplot(2, 1, ind)
    plt.imshow(img, cmap='Greys_r')

    # Plot lidar points.
    plt.scatter(positions[:, 0], positions[:, 1], c=colors, s=5)