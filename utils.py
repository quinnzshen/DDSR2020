import numpy as np
from PIL import Image

import random
import os
from glob import glob

CAMERAS = {"stereo_left": "image_02", "stereo_right": "image_03"}
KITTI_TIMESTAMPS = [CAMERAS["stereo_left"] + "/timestamps.txt", CAMERAS["stereo_right"] + "/timestamps.txt",
                    "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]
SPLIT_NAMES = ["train.txt", "validate.txt"]


def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since the midnight of that day
    :param time_string: The string to be converted into nanoseconds
    :return: The number of nanoseconds since midnight
    """
    total = 0
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total


def get_nsec_times(sample_path, idx):
    """
    Given the file path to the scene and the frame number within that scene, returns a NumPy array containing the
    time (nanoseconds) of the left frame, right frame, start, and end of the idx'th LiDAR scan, in that order.
    :param sample_path: A file_path to a scene within the dataset
    :param idx: The frame number within the scene
    :return: NumPy array shape (4,) containing the time of the idx'th events (see above)
    """
    times = np.empty(len(KITTI_TIMESTAMPS), dtype=np.int64)
    for i in range(len(KITTI_TIMESTAMPS)):
        with open(os.path.join(sample_path, KITTI_TIMESTAMPS[i])) as f:
            count = 0
            for line in f:
                if count == idx:
                    times[i] = iso_string_to_nanoseconds(line)
                    break
                count += 1

    return times


def get_camera(path_name, camera_name, idx):
    """
    Gets the basic camera information given the path name to the scene, the camera name, and the frame number within
    that scene.
    :param path_name: A file path to a scene within the dataset
    :param camera_name: A camera name as defined in CAMERAS
    :param idx: The frame number in the scene
    :return: A dictionary containing the image (in a NumPy array) and the shape of that array
    """
    img_arr = np.asarray(Image.open(os.path.join(path_name, CAMERAS[camera_name] + f"/data/{idx:010}.png")))
    return {camera_name + "_image": img_arr, camera_name + "_shape": img_arr.shape}


def generate_split(root_dir, split=0.7, seed=0):
    """
    Generates the train.txt and validate.txt by using random for every frame and deciding whether to put it in
    train.txt or validate.txt.
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
