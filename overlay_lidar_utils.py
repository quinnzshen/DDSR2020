import numpy as np
from matplotlib import pyplot as plt
import colorsys
from plotly import graph_objects as go

import plotly_utils


def generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne, im_width, im_height):
    """
    This function removes the lidar pointts that are not in the image plane, rounds x/y pixel values for lidar points, 
    and projects the lidar points onto the image plane
    :param [numpy.array] lidar_point_coord_velodyne: [N, 4], matrix of lidar points, each row is format [X, Y, Z, reflectivity]
    :param [numpy.array] camera_image_from_velodyne: [4, 4], converts 3D lidar points to 2D image plane
    :param [int] im_width: width of image in pixels
    :param [int] im_height: height of image in pixels
    :return: numpy.array of shape [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    """
    # Based on code from monodepth2 repo.

    # Add right column of ones to lidar_point_coord_velodyne.
    lidar_point_coord_velodyne[:, 3] = np.ones((len(lidar_point_coord_velodyne)))
    
    # Remove points behind velodyne sensor.
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >=0, :]
    
    # Project points to image plane.
    lidar_point_coord_camera_image = np.dot(camera_image_from_velodyne, lidar_point_coord_velodyne.T).T
    # lidar_point_coord_camera_image = lidar_point_coord_camera_image[lidar_point_coord_camera_image[:, 2] > 0]
    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / lidar_point_coord_camera_image[:, 2][..., np.newaxis]
    
    # Round X and Y pixel coordinates to int.
    lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)

    # Create filtered index only inlude those in image field of view.
    filtered_index = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
                    (lidar_point_coord_camera_image[:, 0] < im_width) & (lidar_point_coord_camera_image[:, 1] < im_height)
    
    return lidar_point_coord_camera_image, filtered_index


def plot_lidar_on_image(image, lidar_point_coord_camera_image, fig, ind):
    """
    This function plots lidar points on the image with colors corresponding to their depth(higher hsv hue val = further away) 
    :param [numpy.array] image: [H, W], contains image data
    :param [numpy.array] lidar_point_coord_camera_image: [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    :return: None. Plots image w/ lidar overlay.
    """
    # Normalize depth values.
    lidar_point_coord_camera_image = normalize_depth(lidar_point_coord_camera_image[:, :3])
    
    # Make array of colors (row number is equal to row number containing corresponding x/y point in lidar_point_coord_camera_image)
    colors = np.zeros(lidar_point_coord_camera_image.shape)
    for idx in range(len(colors)):
        colors[idx] = np.asarray(colorsys.hsv_to_rgb(lidar_point_coord_camera_image[idx][2] * (240/360), 1.0, 1.0))
    
    # Show grayscale image.
    fig.add_subplot(2, 1, ind)
    plt.imshow(image, cmap='Greys_r')
    
    # Plot lidar points.
    plt.scatter(lidar_point_coord_camera_image[:, 0], lidar_point_coord_camera_image[:, 1], c = colors, s = 5)
    # plt.show()


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
    lidar_point_coord_camera_image[:, 2] = (lidar_point_coord_camera_image[:, 2] - min)/(max-min)
    return lidar_point_coord_camera_image


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
