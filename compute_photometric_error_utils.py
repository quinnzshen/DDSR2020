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
    plt.figure(figsize=(1242/50, 375/50))
    plt.gca().invert_yaxis()
    plt.scatter(lidar_point_coord_camera_image[:, 0], lidar_point_coord_camera_image[:, 1], c = colors, s = 7, marker = 's')
    plt.show()
