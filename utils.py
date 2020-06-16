import numpy as np
from PIL import Image

import random
import os
from glob import glob

from kitti_utils import load_velodyne_points

CAMERAS = {"stereo_left": "image_02", "stereo_right": "image_03"}
KITTI_TIMESTAMPS = ["/timestamps.txt", "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]
SPLIT_NAMES = ["dataloader_tests.txt", "validate.txt"]
CAMERA_FIELD_NAMES = [
    "_image",
    "_shape",
    "_capture_time_nsec"
]
LIDAR_FIELD_NAMES = [
    "lidar_point_coord_velodyne",
    "lidar_point_reflectivity",
    "lidar_start_capture_time_nsec",
    "lidar_end_capture_time_nsec"
]


def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since the midnight of that day
    :param time_string: The string to be converted into nanoseconds
    :return: The number of nanoseconds since midnight
    """
    total = np.int64(0)
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total


def get_nsec_time(sample_path, idx):
    """
    Given the file path to the scene and the frame number within that scene, returns a NumPy array containing the
    time (nanoseconds) of the left frame, right frame, start, and end of the idx'th LiDAR scan, in that order.
    :param sample_path: A file_path to a scene within the dataset
    :param idx: The frame number within the scene
    :return: NumPy array shape (4,) containing the time of the idx'th events (see above)
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
    :param path_name: A file path to a scene within the dataset
    :param camera_name: A camera name as defined in CAMERAS
    :param idx: The frame number in the scene
    :return: A dictionary containing the image (in a NumPy array) and the shape of that array
    """
    img_arr = np.asarray(Image.open(os.path.join(path_name, CAMERAS[camera_name] + f"/data/{idx:010}.png")))
    return {
        camera_name + CAMERA_FIELD_NAMES[0]: img_arr,
        camera_name + CAMERA_FIELD_NAMES[1]: img_arr.shape,
        camera_name + CAMERA_FIELD_NAMES[2]: get_nsec_time(os.path.join(path_name, CAMERAS[camera_name] + KITTI_TIMESTAMPS[0]), idx)
    }


def get_lidar_data(path_name, idx):
    lidar_points = load_velodyne_points(os.path.join(path_name, f"velodyne_points/data/{idx:010}.bin"))
    return {
        LIDAR_FIELD_NAMES[0]: lidar_points[:, :3],
        LIDAR_FIELD_NAMES[1]: lidar_points[:, 3],
        LIDAR_FIELD_NAMES[2]: get_nsec_time(os.path.join(path_name, KITTI_TIMESTAMPS[1]), idx),
        LIDAR_FIELD_NAMES[3]: get_nsec_time(os.path.join(path_name, KITTI_TIMESTAMPS[2]), idx)
    }


def generate_split(root_dir, split=0.7, seed=0):
    """
    Generates the dataloader_tests.txt and validate.txt by using random for every frame and deciding whether to put it in
    dataloader_tests.txt or validate.txt.
    :param root_dir: The root directory of the dataset
    :param split: The chance of a given frame being put into train
    :param seed: The seed of the RNG, if None, then it is random (default seed is 0)
    """
    random.seed(seed)

    with open(os.path.join(root_dir, SPLIT_NAMES[0]), "w") as train:
        with open(os.path.join(root_dir, SPLIT_NAMES[1]), "w") as val:

            for direc in glob(root_dir + "/*/"):
                # iterating through all date folders
                for sub_dir in glob(direc + "/*/"):
                    # iterating through all date_drive folders
                    with open(os.path.join(sub_dir, "velodyne_points/timestamps.txt")) as file:
                        subtotal = 0
                        for _ in file:
                            line = sub_dir + " {}\n".format(subtotal)
                            if random.random() < split:
                                train.write(line)
                            else:
                                val.write(line)
                            subtotal += 1
