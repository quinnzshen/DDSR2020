import numpy as np
import kitti_utils as ku
import os
from matplotlib import pyplot as plt


def plot_sparse_image(lidar_point_coord_camera_image, image):
    """
    This function plots only the pixels of an image that have a corresponding lidar depth value.
    :param: [numpy.array] lidar_point_coord_camera_image: [N, 4], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    :param: [numpy.array] image: [H, W, 3], contains R, G, and B values for every pixel of the image that the user wants to plot.
    :return: None, plots a sparse image where only the pixels woth corresponding depth values are shown (pixels are dilated).
    """
    # Create array of colors for each pixel.
    colors = np.zeros(lidar_point_coord_camera_image[:, :3].shape)
    for idx in range(len(colors)):
        y = lidar_point_coord_camera_image[idx][0]
        x = lidar_point_coord_camera_image[idx][1]
        colors[idx] = image[x][y]
    
    # Plot pixels that have a corresponding depth value.
    plt.figure(figsize=(image.shape[1]/50, image.shape[0]/50))
    plt.gca().invert_yaxis()
    plt.scatter(lidar_point_coord_camera_image[:, 0], lidar_point_coord_camera_image[:, 1], c = colors, s = 7, marker = 's')
    plt.show()

def get_depth_map_for_target_image(lidar_point_coord_camera_image, img_height, img_width):
    """
    This function creates a depth map the size of the image where each location contains the depth at that pixel, 
    or zero if there is no ground-truth depth for that pixel.
    :param: [numpy.array] lidar_point_coord_camera_image: [N, 4], contains lidar depth values for points on image plane, each row is format [X, Y, depth, 1].
    :param: [int] im_height: Height of the image that a depth map is being produced for in pixels.
    :param: [int] im_width: Width of the image that a depth map is being produced for in pixels.
    :return: numpy.array of shape [img_height, img_width] that contains the lidar depth value for ech pixel location in the image, 
    or zero if there is no depth value for a point.
    """
    # Create array of zeros that is the same shape as the image.
    full_image_depth_map = np.zeros((img_height, img_width))
    # Add depth values at pixel points that have a corresponding lidar point.
    for row in lidar_point_coord_camera_image:
        full_image_depth_map[row[1]][row[0]] = row[2]
    
    return full_image_depth_map

def get_meshgrid(img_height, img_width):
    """
    This function creates a meshgrid, or an array where each location contains an array containing the homogeneous coordinates 
    of that location, that is the same shape as the image.
    :param: [int] im_height: Height of the image that a meshgrid is being produced for in pixels.
    :param: [int] im_width: Width of the image that a meshgrid is being produced for in pixels.
    :return: numpy.array of shape [im_height, im_width, 3] where each location in the array contains the coordinates of that 
    location in the format [x, y, 1].
    """
    # Create array of zeros that is the same shape as the image with a 3x1 array at each of each locations.
    meshgrid = np.zeros((img_height, img_width, 3))
    # Set the value at each location in the array to the homogeneous coordinates of that location
    for row in range(len(meshgrid)):
        for col in range (len(meshgrid[0])):
            meshgrid[row][col] = np.array([[row, col, 1]])
    return meshgrid

def convert_source_pixel_coords_to_camera_frame(pixel_coordinates, full_image_depth_map, source_intrinsic, img_height, img_width):
    """
    This function projects the pixel coordinates of the source image into camera_frame.
    :param: [numpy.array] pixel_coordinates: [im_height, im_width, 3], array that is the same shape as the source image, where 
    each location contains an array of the homogenous pixel coordinates of that location, in format [x, y, 1]
    :param: [numpy.array] full_image_depth_map: [im_height, im_width],  contains the lidar depth value for each pixel location in the image, 
    or zero if there is no depth value for a point.
    :param: [numpy_array] source_intrinsic: [3, 3], intrinsic matrix for the camera that captured the source image
    :param: [int] im_height: Height of the source image in pixels.
    :param: [int] im_width: Width of the source image  in pixels.
    :return: numpy.array of shape [4, im_height, im_width] containing the camera coordinates for each pixel location in the 
    source image (only includes camera coordinates for pixels that have corresponding ground truth depth values)
    """
    # Reshape depth and pixel coordinate arrays for multiplication.
    depth = np.reshape(full_image_depth_map, (1,-1))
    pixel_coordinates = pixel_coordinates.flatten().reshape((3, -1), order = 'F')
    # Convert source pixel coordinates to camera coordinates.
    camera_coordinates = (np.linalg.inv(source_intrinsic) @ pixel_coordinates) * depth
    # Add a bottom row of ones to make camera coordinates homogeneous.
    ones = np.ones((1, img_height * img_width))
    camera_coordinates = np.vstack((camera_coordinates, ones))
    # Reshape camera coordinates back to the shape of the source image.
    camera_coordinates = np.reshape(camera_coordinates, (-1, img_height, img_width))
    
    return camera_coordinates

def get_target_pixel_frame_from_source_camera_frame_matrix(rotation_source_to_target, translation_source_to_target, target_intrinsic):
    """
    This function computes the matrix needed to convert source camera coordinates to target pixel coordinates.
    :param: [numpy.array] rotation_source_to_target: [3, 3] matrix representing the relative rotation between the camera that 
    captured the source image and the camera that captured the target image.
    :param: [numpy.array] translation_source_to_target: [3, 1] vector representing the relative translation between the camera that 
    captured the source image and the camera that captured the target image.
    :param: [numpy.array] target_intrinsic: [3, 3] intrinsic matrix for the camera that captured the target image
    :return: numpy.array of shape [4, 4] that contains matrix needed to convert source camera coordinates to target pixel coordinates.
    """
    # Overlay intrinsic matrix on 4x4 identity matrix.
    target_intrinsic_4x4 = np.eye(4)
    target_intrinsic_4x4[:3, :3] = target_intrinsic
    # Compute relative pose matrix from relative rotation and relative translation.
    relative_pose_matrix = np.hstack((rotation_source_to_target, translation_source_to_target))
    relative_pose_matrix = np.vstack((relative_pose_matrix, np.array([[0, 0, 0, 1]])))
    # Multiply target intrinsic matrix by relative pose matrix.
    target_camera_frame_from_source_camera_frame = target_intrinsic_4x4 @ relative_pose_matrix
    return target_camera_frame_from_source_camera_frame

def compute_relative_rotation_stereo(calibration_dir):
    """
    This function computes the relative rotation matrix between stereo cameras for KITTI.
    :param: [string] calibration dir: directory where KITTI calibration files are located.
    :return: numpy.array of shape [3, 3], matrix representing the relative rotation between the camera that captured the source 
    image and the camera that captured the target image.
    """
    # Read calibration file.
    cam2cam = ku.read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    # Compute relative rotation matrix.
    rotation_target = cam2cam['R_rect_02'].reshape(3, 3)
    rotation_source = cam2cam['R_rect_03'].reshape(3,3)
    rotation_source_to_target = rotation_source @ np.linalg.inv(rotation_target)

    return rotation_source_to_target

def compute_relative_translation_stereo(calibration_dir):
    """
    This function computes the relative translation vector between stereo cameras for KITTI.
    :param: [string] calibration dir: directory where KITTI calibration files are located.
    :return: numpy.array of shape [3, 1], vector representing the relative translation between the camera that captured the source 
    image and the camera that captured the target image.
    """
     # Read calibration file.
    cam2cam = ku.read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    # Compute relative translation vector.
    translation_target = cam2cam['T_02'].reshape(3, 1)
    translation_source = cam2cam['T_03'].reshape(3, 1)
    rotation_source = cam2cam['R_03'].reshape(3,3)
    translation_source_to_target = (np.linalg.inv(rotation_source) @ (translation_target - translation_source))

    
    temp = translation_source_to_target[0][0]
    translation_source_to_target[0] = translation_source_to_target[1]
    translation_source_to_target[1][0] = temp

    return translation_source_to_target
    

def project_source_camera_to_target_pixel_frame(camera_coordinates, target_camera_frame_from_source_pixel_frame, img_height, img_width):
    """
    This function projects the source camera coordinates into the target pixel frames.
    :param: [numpy.array] camera_coordinates: [4, im_height, im_width] contains the camera coordinates for each pixel location in the 
    source image (only includes camera coordinates for pixels that have corresponding ground truth depth values)
    :param: [numpy.array] target_camera_frame_from_source_pixel_frame: [4, 4], contains matrix needed to convert source camera coordinates to target pixel coordinates.
    :param: [int] im_height: Height of the source image in pixels.
    :param: [int] im_width: Width of the source image  in pixels.
    :return: numpy.array of hape [2, img_height, img_width] where each location is a source image pixel and the element in each 
    location is a 2x1 array containing the target pixel coordinates that correspond to that source image pixel 
    (only contains projected target pixel coordinates for source pixels that have a corresponding lidar point).
    """
    camera_coordinates = np.reshape(camera_coordinates, (4, -1))
    target_pixel_coordinates = target_camera_frame_from_source_pixel_frame @ camera_coordinates
    target_pixel_coord_norm_x = target_pixel_coordinates[0, :] / target_pixel_coordinates[2, :]
    target_pixel_coord_norm_y = target_pixel_coordinates[1, :] / target_pixel_coordinates[2, :]
    target_pixel_coord_norm = np.vstack((target_pixel_coord_norm_x, target_pixel_coord_norm_y)).astype(int)
    target_pixel_coord_norm = np.reshape(target_pixel_coord_norm, (2, img_height, img_width)) 
    
    return target_pixel_coord_norm



