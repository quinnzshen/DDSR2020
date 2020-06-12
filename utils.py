import numpy as np
from PIL import Image

import os

CAMERAS = {"stereo_left": "image_02", "stereo_right": "image_03"}
KITTI_TIMESTAMPS = [CAMERAS["stereo_left"] + "/timestamps.txt", CAMERAS["stereo_right"] + "/timestamps.txt",
                    "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]


def bin_search(arr, target, init_search):
    """
    Conducts a binary search on a given array (used to determine which directory the nth frame is located at with lists
    date_divisions and drive_divisions)
    Assumes the array is sorted and returns the least index, i, where arr[i] >= target.
    :param arr: The sorted 1D array/list to be searched
    :param target: The target value to be compared to (see above condition)
    :param init_search: The initial guess that the binary search starts at
    :return: The lowest index, i, where arr[i] >= target
    """
    index = init_search
    lower_index = 0
    upper_index = len(arr) - 1
    prev_index = -1
    while 1:
        if arr[index] > target:
            upper_index = index
        elif arr[index] < target:
            lower_index = index
        else:
            return index

        prev_index = index
        index = (upper_index + lower_index) // 2

        if prev_index == index:
            if index == len(arr) - 1:
                return index

            if arr[index + 1] < target:
                index += 1
            return index


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
