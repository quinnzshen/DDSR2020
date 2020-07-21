import numpy as np
from matplotlib import pyplot as plt
import colorsys
import matplotlib.colors as mcolor
from plotly import graph_objects as go

import plotly_utils


def generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne, im_width, im_height):
    """
    This function removes the lidar pointts that are not in the image plane, rounds x/y pixel values for lidar points, 
    and projects the lidar points onto the image plane
    :param [numpy.array] lidar_point_coord_velodyne: [N, 3], matrix of lidar points, each row is format [X, Y, Z]
    :param [numpy.array] camera_image_from_velodyne: [4, 4], converts 3D lidar points to 2D image plane
    :param [int] im_width: width of image in pixels
    :param [int] im_height: height of image in pixels
    :return: numpy.array of shape [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    """
    # Based on code from monodepth2 repo.

    # Add right column of ones to lidar_point_coord_velodyne.
    lidar_point_coord_velodyne = np.hstack((lidar_point_coord_velodyne, np.ones((len(lidar_point_coord_velodyne))).reshape(-1, 1)))
    # Remove points behind velodyne sensor.
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >=0, :]
    
    # Project points to image plane.
    lidar_point_coord_camera_image = np.dot(camera_image_from_velodyne, lidar_point_coord_velodyne.T).T
    
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
    lidar_point_coord_camera_image[:, 2] = (lidar_point_coord_camera_image[:, 2]-min) / (max-min)
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
