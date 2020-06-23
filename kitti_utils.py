from __future__ import absolute_import, division, print_function

import numpy as np
from PIL import Image

import os
import pandas as pd
from enum import Enum


class KITTICameraNames(str, Enum):
    stereo_left = "stereo_left"
    stereo_right = "stereo_right"

CAMERA_NAME_TO_PATH_MAPPING = {
    KITTICameraNames.stereo_left: "image_02",
    KITTICameraNames.stereo_right: "image_03"
}


KITTI_TIMESTAMPS = ["/timestamps.txt", "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]
EPOCH = np.datetime64("1970-01-01")


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


def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since EPOCH
    :param [str] time_string: The string to be converted into nanoseconds
    :return [np.int64]: The number of nanoseconds since epoch (Jan 1st 1970 UTC)
    """
    return (np.datetime64(time_string) - EPOCH).astype(np.int64)


def get_timestamp_nsec(sample_path, idx):
    """
    Given the file path to the scene and the frame number within that scene, returns an integer containing the
    time (nanoseconds) retrieved from the idx'th line in the path given.
    :param [str] sample_path: A file_path to a timestamps file
    :param [int] idx: The frame number within the scene
    :return [np.int64]: Integer containing the nanosecond taken of the given frame
    """
    with open(sample_path) as f:
        count = 0
        for line in f:
            if count == idx:
                return iso_string_to_nanoseconds(line)
            count += 1

def get_nearby_frames_data(path_name, idx, delta):
        """
        Given a specific index, return a dictionary containing information about the frame n frames before and after the target index
        in the dataset.
        :param dataset_index [pd.DataFrame]: The dataframe containing the paths and indices of the data
        :param [int] delta: Number of frames before and after target frame to retrieve.
        :return [dict]: Dictionary containing camera data of nearby frames, the key is the relative index and the value is the data.
                        (e.g. -1 would be the previous image, 2 would be the next-next image).
        """
        nearby_frames = {}
        for nearby_idx in range(idx - delta, idx + delta + 1):
            # We do not want to include the current frame in the nearby frames data.
            if nearby_idx == 0:
                continue

            print("idx", nearby_idx)
            nearby_frames[nearby_idx] = get_camera_data(path_name, nearby_idx)
        return nearby_frames


def get_camera_data(path_name, idx):
    """
    Gets the basic camera information given the path name to the scene, the camera name, and the frame number within
    that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing camera data. If the camera data cannot be found, return an empty dictionary.
    """
    camera_data = dict()

    for camera_name in KITTICameraNames:
        camera_path = CAMERA_NAME_TO_PATH_MAPPING[camera_name]
        
        # Check if required paths exist.
        camera_image_path = os.path.join(path_name, f"{camera_path}/data/{idx:010}.png")
        timestamp_path = os.path.join(path_name, f"{camera_path}/timestamps.txt")

        if os.path.exists(camera_image_path) and os.path.exists(timestamp_path):
            # The f-string is following the format of KITTI, padding the frame number with 10 zeros.
            camera_image = np.asarray(Image.open(camera_image_path))
            timestamp = get_timestamp_nsec(timestamp_path, idx)
            camera_data[f"{camera_name}_image"] = camera_image
            camera_data[f"{camera_name}_shape"] = camera_image.shape
            camera_data[f"{camera_name}_capture_time_nsec"] = timestamp

    return camera_data


def get_lidar_data(path_name, idx):
    """
    Gets the basic LiDAR information given the path name to the scene and the frame number within that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing the points, reflectivity, start, and end times of the LiDAR scan.
    """
    lidar_points = load_lidar_points(os.path.join(path_name, f"velodyne_points/data/{idx:010}.bin"))
    start_time = get_timestamp_nsec(os.path.join(path_name, "velodyne_points/timestamps_start.txt"), idx)
    end_time = get_timestamp_nsec(os.path.join(path_name, "velodyne_points/timestamps_end.txt"), idx)
    return {
        "lidar_point_coord_velodyne": lidar_points[:, :3],
        "lidar_point_reflectivity": lidar_points[:, 3],
        "lidar_start_capture_time_nsec": start_time,
        "lidar_end_capture_time_nsec": end_time
    }
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
