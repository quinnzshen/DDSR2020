import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import colorsys
import kitti_utils

def generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne, im_width, im_height):
    """
    This function removes the lidar pts that are not in the image plane, rounds x/y pixel vals for lidar pts, and projects the lidar pts onto the image plane
    :param [numpy.array] lidar_point_coord_velodyne: [N,3], matrix of lidar pts, each row is format [X, Y, Z]
    :param [numpy.array] camera_image_from_velodyne: [4,4], converts 3D lidar pts to 2D image plane
    :param [int] im_width: width of image in pixels
    :param [int] im_height: height of image in pixels
    :return: numpy.array of shape [N,4], contains lidar pts on image plane, each row is format [X, Y, depth, 1]
    """
    #Based on code from monodepth2 repo.

    #Add bottom row of ones to lidar_point_coord_velodyne
    lidar_point_coord_velodyne = np.hstack((lidar_point_coord_velodyne, np.ones((np.size(lidar_point_coord_velodyne,0)))[:, None]))
    
    #Remove points behind camera.
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[lidar_point_coord_velodyne[:,0] >=0, :]
    
    #Project points to image plane.
    lidar_point_coord_camera_image = np.dot(camera_image_from_velodyne, lidar_point_coord_velodyne.T).T
    lidar_point_coord_camera_image[:, :2] = lidar_point_coord_camera_image[:, :2] / lidar_point_coord_camera_image[:, 2][..., np.newaxis]
  
    #Check if in image FOV.
    lidar_point_coord_camera_image[:, 0] = lidar_point_coord_camera_image[:, 0].astype(int) - 1
    lidar_point_coord_camera_image[:, 1] = lidar_point_coord_camera_image[:, 1].astype(int) - 1
    val_inds = (lidar_point_coord_camera_image[:, 0] >= 0) & (lidar_point_coord_camera_image[:, 1] >= 0)
    val_inds = val_inds & (lidar_point_coord_camera_image[:, 0] < im_width) & (lidar_point_coord_camera_image[:, 1] < im_height)
    lidar_point_coord_camera_image = lidar_point_coord_camera_image[val_inds, :]
    
    return lidar_point_coord_camera_image

def render_lidar_on_image(image, lidar_point_coord_camera_image):
    """
    This function plots lidar points on the image with colors corresponding to their depth(higher hsv hue val = further away) 
    :param [numpy.array] image: [H, W], contains image data
    :param [numpy.array] lidar_point_coord_camera_image: [N,4], contains lidar pts on image plane, each row is format [X, Y, depth, 1]
    :return: no return val, shows image w/ lidar overlay
    """
    #Normalize depth values.
    lidar_point_depth = lidar_point_coord_camera_image[:, 2].reshape(-1,1)
    lidar_point_depth_normalized = normalize_depth(lidar_point_depth)
   
    #Show grayscale image.
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    plt.imshow(image, cmap='Greys_r')
    
    #Plot lidar points.
    for idx in range(len(lidar_point_coord_camera_image)):
        normalized_depth = np.asscalar(lidar_point_depth_normalized[idx][0])
        col = colorsys.hsv_to_rgb(normalized_depth * (240/360), 1.0, 1.0)
        circ = patches.Circle((lidar_point_coord_camera_image[idx][0],lidar_point_coord_camera_image[idx][1]),2)
        circ.set_facecolor(col) 
        ax.add_patch(circ)
    
    plt.show()

def normalize_depth(lidar_point_depth):
    """
    This function normalizes the depth values that are passed so they are between 0 and 1, inclusive 
    :param [numpy.array] lidar_point_depth: [N, 1] contains all of the depth values for points that are in the image FOV
    :return: numpy.array of shape [N,1] containing the depth values normalized so they are between 0 and 1, inclusive
    """
    max = lidar_point_depth.max()
    min = lidar_point_depth.min()
    lidar_point_depth_normalized = (lidar_point_depth - min)/(max-min)
    return lidar_point_depth_normalized