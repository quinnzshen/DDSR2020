import numpy as np
from PIL import Image

import os
import enum

from kitti_utils import load_velodyne_points


class KITTICameraNames(enum.Enum):
    stereo_left = "image_02"
    stereo_right = "image_03"


KITTI_TIMESTAMPS = ["/timestamps.txt", "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]


def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since the midnight of that day
    :param [str] time_string: The string to be converted into nanoseconds
    :return [np.int64]: The number of nanoseconds since midnight
    """
    total = np.int64(0)
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total


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


def get_camera_data(path_name, camera_name, idx):
    """
    Gets the basic camera information given the path name to the scene, the camera name, and the frame number within
    that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [str] camera_name: A camera name as defined in CAMERAS
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing the image (in a NumPy array), the shape of that array, and time taken
    """

    # The f-string is following the format of KITTI, padding the frame number with 10 zeros.
    img_arr = np.asarray(Image.open(os.path.join(path_name, KITTICameraNames[camera_name].value + f"/data/{idx:010}.png")))
    timestamp = get_timestamp_nsec(os.path.join(path_name, KITTICameraNames[camera_name].value + KITTI_TIMESTAMPS[0]), idx)
    return {
        f"{camera_name}_image": img_arr,
        f"{camera_name}_shape": img_arr.shape,
        f"{camera_name}_capture_time_nsec": timestamp
    }


def get_lidar_data(path_name, idx):
    """
    Gets the basic LiDAR information given the path name to the scene and the frame number within that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing the points, reflectivity, start, and end times of the LiDAR scan.
    """
    lidar_points = load_velodyne_points(os.path.join(path_name, f"velodyne_points/data/{idx:010}.bin"))
    start_time = get_timestamp_nsec(os.path.join(path_name, KITTI_TIMESTAMPS[1]), idx)
    end_time = get_timestamp_nsec(os.path.join(path_name, KITTI_TIMESTAMPS[2]), idx)
    return {
        "lidar_point_coord_velodyne": lidar_points[:, :3],
        "lidar_point_reflectivity": lidar_points[:, 3],
        "lidar_start_capture_time_nsec": start_time,
        "lidar_end_capture_time_nsec": end_time
    }
