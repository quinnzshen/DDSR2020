import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import colorsys
import kitti_utils

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
    lidar_point_coord_velodyne = np.hstack((lidar_point_coord_velodyne, np.ones((np.size(lidar_point_coord_velodyne, 0)))[:, None]))
    
    # Remove points behind velodyne sensor.
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:, 0] >=0, :]
    
    # Project points to image plane.
    lidar_point_coord_camera_image = np.dot(camera_image_from_velodyne, lidar_point_coord_velodyne.T).T
    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / lidar_point_coord_camera_image[:, 2][..., np.newaxis]
    
    # Round X and Y pixel coordinates to int.
    lidar_point_coord_camera_image = np.around(lidar_point_coord_camera_image).astype(int)

    # Filtering points to only inlude those in image field of view.
    filtered_index = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0) & \
                    (lidar_point_coord_camera_image[:, 0] < im_width) & (lidar_point_coord_camera_image[:, 1] < im_height)
    
    return lidar_point_coord_camera_image[filtered_index, :]

def plot_lidar_on_image(image, lidar_point_coord_camera_image):
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
    plt.imshow(image, cmap='Greys_r')
    
    # Plot lidar points.
    plt.scatter(lidar_point_coord_camera_image[:, 0], lidar_point_coord_camera_image[:, 1], c = colors, s = 5)
    plt.show()

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
