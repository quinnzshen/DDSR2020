import numpy as np

import os

EARTH_RADIUS = 6378137  # meters


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


def calc_lon_dist(lat1, lat2, lon1, lon2):
    """
    Calculates the East-West distance (in meters) between two longitudes at given latitudes.
    Utilizes a simplified form of the Haversine formula by pretending that delta latitude is 0, to calculate
    East-West distance.
    :param lat1: The latitude of the original point (radians)
    :param lat2: The latitude of the new point (radians)
    :param lon1: The longitude of the original point (radians)
    :param lon2: The longitude of the new point (radians)
    :return: The East-West distance between the two points on Earth's surface
    """
    avg_lat = (lat2 + lat1) / 2
    delta_lon_two = (lon2 - lon1) / 2
    return 2 * EARTH_RADIUS * np.arctan2(
        np.cos(avg_lat) * np.sin(delta_lon_two),
        np.sqrt(1 - np.sin(avg_lat) * np.sin(avg_lat) * np.cos(delta_lon_two) * np.cos(delta_lon_two))
    )


def time_to_nano(time_string):
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
    with open(os.path.join(sample_path, "image_02/timestamps.txt")) as l_time:
        with open(os.path.join(sample_path, "image_03/timestamps.txt")) as r_time:
            with open(os.path.join(sample_path, "velodyne_points/timestamps_start.txt")) as start_time:
                with open(os.path.join(sample_path, "velodyne_points/timestamps_end.txt")) as end_time:
                    count = 0
                    for l_line, r_line, start_line, end_line in zip(l_time, r_time, start_time, end_time):
                        if count == idx:
                            return np.array(list(map(time_to_nano, [l_line, r_line, start_line, end_line])), dtype=np.int64)
                        count += 1


def calc_transformation_mat(sample_path, idx):
    """
    Given the file path to the scene and the frame number within that scene, returns a 4x4 NumPy array containing the
    transformation matrix to convert the LiDAR point coordinates (relative to the sensor) into global coordinates
    (relative to the starting point), where +x is East, +y is North, and +z is up.
    :param sample_path: A file_path to a scene within the dataset
    :param idx: The frame number within the scene
    :return: 4x4 homogenous transformation matrix to convert relative coordinates into continuous coordinates
    """
    with open(os.path.join(sample_path, "oxts/data/") + f"{0:010}.txt") as f:
        line = f.readline().split()
        orig_coords = np.array(line[:3], dtype=np.float64)
        if idx == 0:
            new_coords = np.array(line[:6], dtype=np.float64)
        else:
            with open(os.path.join(sample_path, "oxts/data/") + f"{idx:010}.txt") as fi:
                new_coords = np.array(fi.readline().split(), dtype=np.float64)

    latlon_orig = np.deg2rad(orig_coords[:2])
    latlon_new = np.deg2rad(new_coords[:2])
    sin_rpy = np.sin(new_coords[3:])
    cos_rpy = np.cos(new_coords[3:])

    # transformation matrix
    return np.array([
        [
            cos_rpy[2] * cos_rpy[1],
            cos_rpy[2] * sin_rpy[1] * sin_rpy[0] - sin_rpy[2] * cos_rpy[0],
            cos_rpy[2] * sin_rpy[1] * cos_rpy[0] + sin_rpy[2] * sin_rpy[0],
            calc_lon_dist(latlon_orig[0], latlon_new[0], latlon_orig[1], latlon_new[1])
        ],
        [
            sin_rpy[2] * cos_rpy[1],
            sin_rpy[2] * sin_rpy[1] * sin_rpy[0] + cos_rpy[2] * cos_rpy[0],
            sin_rpy[2] * sin_rpy[1] * sin_rpy[0] - cos_rpy[2] * sin_rpy[0],
            EARTH_RADIUS * (latlon_new[0] - latlon_orig[0])
        ],
        [
            -1 * sin_rpy[1],
            cos_rpy[1] * sin_rpy[0],
            cos_rpy[1] * cos_rpy[0],
            new_coords[2] - orig_coords[2]
        ],
        [0, 0, 0, 1],
    ], dtype=np.float64)
