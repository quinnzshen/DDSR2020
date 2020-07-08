import numpy as np
import overlay_lidar_utils as olu
from matplotlib import pyplot as plt

def create_depth_map_from_nearest_lidar_point(lidar_point_coord_camera_image, img_height, img_width):
    """
    This function creates an array the size of the image where each location contains the normalized depth value of the closest lidar point.
    :param [numpy.array] lidar_point_coord_camera_image: [[N, 3], contains lidar points on image plane, each row is format [X, Y, depth]].
    :param [int] img_height: Height of the image that corresponds to the lidar sweep in pixels.
    :param [int] img_width: Height of the image that corresponds to the lidar sweep in pixels.
    :return: numpy.array of shape [img_height, img_width] where each location contains the normalized depth value of the closest lidar point.
    """
    # Normalize depth values.
    lidar_point_coord_camera_image_norm = olu.normalize_depth(lidar_point_coord_camera_image)
    # Find height of the lidar point furthest from the bottom of the image.
    max_lidar_height = int(min(lidar_point_coord_camera_image_norm[:, 1]))
    # Create dense_depth_map and fill locations with corresponding lidar points with the lidar points' normalized depth value.
    dense_depth_map = np.zeros((img_height, img_width))
    for x, y, depth in lidar_point_coord_camera_image_norm[:, :3]:
        dense_depth_map[int(y)][int(x)] = depth
    # Fill locations without corresponding lidar points with the normalized depth value of the closest lidar point.
    for r in range(max_lidar_height, img_height):
        for c in range(img_width):
            if dense_depth_map[r][c] == 0:
                distances_to_lidar_points = np.sqrt((lidar_point_coord_camera_image_norm[:, 0] - c)**2 + (lidar_point_coord_camera_image_norm[:, 1] - r)**2)
                idx = np.argmin(distances_to_lidar_points)
                dense_depth_map[r][c] = lidar_point_coord_camera_image_norm[idx][2]
    return dense_depth_map

def create_depth_map_from_lidar_smoothing(lidar_point_coord_camera_image, img_height, img_width, kernel_size):
    """
    This function creates an array the size of the image where each kernel_size x kernel_size kernel is filled with the average depth value of all of the lidar points contained in that kernel.
    :param [numpy.array] lidar_point_coord_camera_image: [[N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]].
    :param [int] img_height: Height of the image that corresponds to the lidar sweep in pixels.
    :param [int] img_width: Height of the image that corresponds to the lidar sweep in pixels.
    :param [int] kernel_size: Size in pixels of the kernels that the depth values are being averaged within.
    :return: numpy.array of shape [img_height, img_width] where each kernel_size x kernel_size kernel is filled with the average depth value of all of the lidar points contained in that kernel.
    """
     # Normalize depth values.
    lidar_point_coord_camera_image_norm = olu.normalize_depth(lidar_point_coord_camera_image)
    # Find height of the lidar point furthest from the bottom of the image.
    max_lidar_height = int(min(lidar_point_coord_camera_image_norm[:, 1]))
    # Create dense_depth_map and fill locations with corresponding lidar points with the lidar points' normalized depth value
    dense_depth_map = np.zeros((img_height, img_width))
    for x, y, depth in lidar_point_coord_camera_image_norm[:, :3]:
        dense_depth_map[int(y)][int(x)] = depth
    # Fill each kernel_size x kernel_size kernel of the image with the average depth value of all of the lidar points contained in that kernel.
    for r in range(max_lidar_height, img_height, kernel_size):
        for c in range(0, img_width, kernel_size):
            kernel = dense_depth_map[r:r+kernel_size, c:c+kernel_size]
            depth_vals = kernel[kernel>0]
            if(depth_vals.size != 0):
                avg_depth = np.average(depth_vals)
            else:
                avg_depth = None
            dense_depth_map[r:r+kernel_size, c:c+kernel_size].fill(avg_depth)
    return dense_depth_map

def plot_depth_completion_image(dense_depth_map, max_lidar_height):
    """
    This function plots a given dense depth map as an image.
    :param [numpy.array] dense_depth_map: [H, W], each location contains the normalized depth value of the closest lidar point.
    :param [int] max_lidar_height: height of the lidar point furthest from the bottom of the image.
    :return: None, plots dense_depth_map as an image.
    """
    # Set figure size.
    plt.figure(figsize=(20, 20))
    # Show image with reversed magma colormap.
    plt.imshow(dense_depth_map, cmap = 'magma_r')
    # Set y-axis limits to exclude area where there are no lidar points.
    plt.gca().set_ylim([plt.gca().get_ylim()[0], max_lidar_height])
    plt.show()
