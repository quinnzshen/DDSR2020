from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pandas as pd

def load_lidar_points(filename):
    """
    This function loads 3D point cloud from KITTI file format.
    :param [string] filename: File path for 3d point cloud data.
    :return [np.array]: [N, 4] N lidar points represented as (X, Y, Z, reflectivity) points.
    """
    return np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

def read_calibration_file(path):
    """
    This function reads the KITTI calibration file.
    (from https://github.com/hunse/kitti)
    :param [string] path: File path for KITTI calbration file.
    :return [dictionary] data: Dictionary containing the camera intrinsic and extrinsic matrices.
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # Try to cast to float array.
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # Casting error: data[key] already eq. value, so pass.
                    pass

    return data

def compute_image_from_velodyne_matrices(calibration_dir):
    """
    This function computes the transformation matrix to project 3D lidar points into the 2D image plane.
    :param [String] calibration_dir: Directory to folder containing camera/lidar calibration files
    :return:  dictionary of numpy.arrays of shape [4, 4] that converts 3D lidar points to 2D image plane for each camera
    (keys: cam00, cam01, cam02, cam03)
    """
    # Based on code from monodepth2 repo.
    
    # Load cam_to_cam calib file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    # Load velo_to_cam file.
    velo2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))

    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3, 1)))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    camera_image_from_velodyne_dict = {}

    KITTI_CAMERA_NAMES = ['00', '01', '02', '03']
    for camera_name in KITTI_CAMERA_NAMES:
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam[f"R_rect_{camera_name}"].reshape(3, 3) 
        P_rect = cam2cam[f"P_rect_{camera_name}"].reshape(3, 4)
        camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
        camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
        camera_image_from_velodyne_dict.update({f"cam{camera_name}" : camera_image_from_velodyne})
    
    return camera_image_from_velodyne_dict


def get_imu_data(scene_path, idx):
    """
    Get Intertial Measurement Unit (IMU) data. 
    :param [string] scene_path: A file path to a scene within the KITTI dataset.
    :param [int] idx: The frame number in the scene. 
    :return [dict]: Return a dictionary of imu data (key: field name, value: field value).
    """
    imu_data_path = os.path.join(scene_path, f"oxts/data/{idx:010}.txt")
    imu_format_path = os.path.join(scene_path, "oxts/dataformat.txt")
    
    with open(imu_format_path) as f:
        # The data is formatted as "name: description". We only care about the name here.
        imu_keys = [line.split(':')[0] for line in f.readlines()]

    with open(imu_data_path) as f:
        imu_values = f.read().split()

    return dict(zip(imu_keys, imu_values))


def get_imu_dataframe(scene_path):
    """
    Get Intertial Measurement Unit (IMU) data for an entire scene.
    :param [string] scene_path: A file path to a scene within the KITTI dataset.
    :return [pd.DataFrame]: A dataframe with the entire scenes IMU data.
    """
    num_frames = len(os.listdir(os.path.join(scene_path, 'oxts/data')))

    imu_values = []
    for idx in range(num_frames):
        imu_data = get_imu_data(scene_path, idx)
        imu_values.append(list(imu_data.values()))

    return pd.DataFrame(imu_values, columns=list(imu_data.keys()))