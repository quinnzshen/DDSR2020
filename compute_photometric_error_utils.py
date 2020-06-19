import numpy as np
import kitti_utils as ku
import os
from matplotlib import pyplot as plt


def get_camera_intrinsic_dict(calibration_dir): # Might be better off in kitti_utils.py
    """
    This function gets the intrinsic matrix for each camera from the KITTI claibation file
    :param: [string] calibration_dir: directory where the KITTI calbration files are located
    :return: dictionary of length 4 containing the 3x3 intrinsic matrix for each camera (keys: cam00, cam01, cam02, cam03)
    """
    # Load cam_to_cam calib file.
    cam2cam = ku.read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    camera_intrinsic_dict = {}
    KITTI_CAMERA_NAMES = ['00', '01', '02', '03']

    for camera_name in KITTI_CAMERA_NAMES:
        intrinsic_matrix = cam2cam[f"K_{camera_name}"].reshape(3,3)
        camera_intrinsic_dict.update({f"cam{camera_name}" : intrinsic_matrix})
    
    return camera_intrinsic_dict

def plot_sparse_image(lidar_point_coord_camera_image, image):
    """
    This function plots only the pixels of an image that have a corresponding lidar depth value.
    :param: [numpy.array] lidar_point_coord_camera_image: [N, 3], contains lidar points on image plane, each row is format [X, Y, depth, 1]
    :param: [numpy.array] image: [H, W, 3], contains R, G, and B values for every pixel of the image that the user wants to plot.
    :return: None, plots a sparse image where only the pixels woth corresponding depth values are shown (pixels are dilated).
    """
    # Create array of colors for each pixel.
    colors = np.zeros(lidar_point_coord_camera_image.shape)
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
    # Create full-image depth map
    full_image_depth_map = np.zeros((img_height, img_width))
    for row in lidar_point_coord_camera_image:
        full_image_depth_map[row[1]][row[0]] = row[2]
    return full_image_depth_map

def get_meshgrid(img_height, img_width):
    # Create meshgrid.
    meshgrid = np.zeros((img_height, img_width, 3))
    for row in range(len(meshgrid)):
        for col in range (len(meshgrid[0])):
            meshgrid[row][col] = np.array([[row, col, 1]])
    return meshgrid

def convert_source_pixel_coords_to_camera_frame(pixel_coordinates, full_image_depth_map, source_intrinsic, img_height, img_width):
    depth = np.reshape(full_image_depth_map, (1,-1))
    pixel_coordinates = pixel_coordinates.flatten().reshape((3, -1), order = 'F')
    camera_coordinates = (np.linalg.inv(source_intrinsic) @ pixel_coordinates) * depth
    
    ones = np.ones((1, img_height * img_width))
    camera_coordinates = np.vstack((camera_coordinates, ones))
    camera_coordinates = np.reshape(camera_coordinates, (-1, img_height, img_width))
    return camera_coordinates

def get_target_camera_frame_from_source_pixel_frame_matrix(rotation_source_to_target, translation_source_to_target, target_intrinsic):
    target_intrinsic_4x4 = np.eye(4)
    target_intrinsic_4x4[:3, :3] = target_intrinsic
    relative_pose_matrix = np.hstack((rotation_source_to_target, translation_source_to_target))
    relative_pose_matrix = np.vstack((relative_pose_matrix, np.array([[0, 0, 0, 1]])))
    target_camera_frame_from_source_pixel_frame = target_intrinsic_4x4 @ relative_pose_matrix
    return target_camera_frame_from_source_pixel_frame

def compute_relative_rotation_stereo(calibration_dir):
    # Read calibration file.
    cam2cam = ku.read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    # Compute relative rotation matrix.
    rotation_target = cam2cam['R_rect_02'].reshape(3, 3)
    rotation_source = cam2cam['R_rect_03'].reshape(3,3)
    rotation_source_to_target = np.linalg.inv(rotation_source) @ rotation_target
    return rotation_source_to_target

def compute_relative_translation_stereo(calibration_dir):
     # Read calibration file.
    cam2cam = ku.read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    # Compute relative translation vector.
    translation_target = cam2cam['T_02'].reshape(3, 1)
    translation_source = cam2cam['T_03'].reshape(3, 1)
    rotation_source = cam2cam['R_rect_03'].reshape(3,3)
    translation_source_to_target = np.linalg.inv(rotation_source) @ (translation_target - translation_source)
    return translation_source_to_target

def project_source_camera_to_target_pixel_frame(camera_coordinates, target_camera_frame_from_source_pixel_frame, img_height, img_width):
    camera_coordinates = np.reshape(camera_coordinates, (4, -1))
    target_pixel_coordinates = target_camera_frame_from_source_pixel_frame @ camera_coordinates
    target_pixel_coord_norm_x = target_pixel_coordinates[0, :] / target_pixel_coordinates[2, :]
    target_pixel_coord_norm_y = target_pixel_coordinates[1, :] / target_pixel_coordinates[2, :]
    target_pixel_coord_norm = np.vstack((target_pixel_coord_norm_x, target_pixel_coord_norm_y)).astype(int)
    target_pixel_coord_norm = np.reshape(target_pixel_coord_norm, (2, img_height, img_width)) 
    
    return target_pixel_coord_norm



