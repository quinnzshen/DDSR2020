import numpy as np
from PIL import Image

import os

EARTH_RADIUS = 6378137  # meters


# Binary searches through a given array, finds the greatest arr[index] that is less than target
def bin_search(arr, target, init_search):
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
    avg_lat = (lat2 + lat1) / 2
    delta_lon_two = (lon2 - lon1) / 2
    return 2 * EARTH_RADIUS * np.arctan2(
        np.cos(avg_lat) * np.sin(delta_lon_two),
        np.sqrt(1 - np.sin(avg_lat) * np.sin(avg_lat) * np.cos(delta_lon_two) * np.cos(delta_lon_two))
    )


# Converts a line from timestamp.txt into nanoseconds from the start of the date
def time_to_nano(time_string):
    total = 0
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return


def get_nsec_times(sample_path, idx):
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
