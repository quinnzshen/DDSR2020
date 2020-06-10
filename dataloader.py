import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
from glob import glob

from utils import bin_search, get_nsec_times, calc_transformation_mat, get_velo_to_imu


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initializes the Dataset, just given the root directory of the data and sets the original values of the instance
        variables.
        :param root_dir: string containing the path to the root directory
        """
        self.root_dir = root_dir
        self.len = -1
        self.date_divisions = []
        self.drive_divisions = []

    def set_len(self):
        """
        Sets self.len, self.date_divisions, and self.drive_divisions by iterating through the whole dataset, preparing
        for future retrivals.
        Only called once after object initialization, as all the instance variables never change.
        """
        total = 0
        for direc in glob(self.root_dir + "/*/"):
            self.date_divisions.append(total)
            # iterating through all date folders
            subtotal = 0
            sub_drive_divisions = []
            for sub_dir in glob(direc + "/*/"):
                # iterating through all date_drive folders
                sub_drive_divisions.append(subtotal)
                with open(os.path.join(os.path.join(sub_dir, "velodyne_points"), "timestamps.txt")) as file:
                    for _ in file:
                        subtotal += 1

            total += subtotal
            self.drive_divisions.append(sub_drive_divisions)

        self.len = total

    def __len__(self):
        """
        Returns the total frame count in the dataset. If the length has not been calculated yet, runs set_len, else
        it simply returns the previously calculated length.
        :return: The frame count in the dataset
        """
        if self.len < 0:
            self.set_len()
        return self.len

    def __getitem__(self, item):
        """
        Given a specific index, returns a dictionary containing information about the sample (frame) at that index in
        the dataset.
        (If the index is n, returns the nth frame of the dataset and its information.)
        The dictionary's fields are specified in Dataset Fields (Google Sheet file).

        :param item: An int representing the index of the sample to be retrieved
        :return: A dictionary containing fields about the retrieved sample
        """
        if torch.is_tensor(item):
            item = item.tolist()

        if self.len < 0:
            self.set_len()

        if item >= self.len or item < 0:
            raise IndexError("Dataset index out of range. (Less than 0 or greater than or equal to length)")

        # Searching which date folder it's in
        da_index = bin_search(
            self.date_divisions,
            item,
            int(len(self.date_divisions) * (item / self.len))
        )
        item -= self.date_divisions[da_index]

        # Searching which drive folder it's in
        dr_index = bin_search(
            self.drive_divisions[da_index],
            item,
            len(self.drive_divisions[da_index]) // 2
        )
        item -= self.drive_divisions[da_index][dr_index]

        # Path of date folder
        date_name = glob(self.root_dir + "/*/")[da_index]
        path_name = glob(date_name + "/*/")[dr_index]
                
        sample = {}

        # Taking information from the directory and putting it into the sample dictionary
        img_arr = np.asarray(Image.open(os.path.join(path_name, "image_02/data/") + f"{item:010}.png"))
        sample["stereo_left_image"] = img_arr
        sample["stereo_left_shape"] = img_arr.shape

        img_arr = np.asarray(Image.open(os.path.join(path_name, "image_03/data/") + f"{item:010}.png"))
        sample["stereo_right_image"] = img_arr
        sample["stereo_right_shape"] = img_arr.shape

        nsec_times = get_nsec_times(path_name, item)
        sample["stereo_left_capture_time_nsec"] = nsec_times[0]
        sample["stereo_right_capture_time_nsec"] = nsec_times[1]
        sample["lidar_start_capture_timestamp_nsec"] = nsec_times[2]
        sample["lidar_end_capture_timestamp_nsec"] = nsec_times[3]

        # Getting the LiDAR coordinates
        lidar_points = np.fromfile(os.path.join(path_name, "velodyne_points/data/") + f"{item:010}" + ".bin", dtype=np.float32).reshape((-1, 4))
        sample["lidar_point_sensor"] = lidar_points[:, :3]
        sample["lidar_point_reflectivity"] = lidar_points[:, 3]

        # Velodyne to continuous 4x4 transformation matrix

        sample["transformation"] = np.matmul(calc_transformation_mat(path_name, item), get_velo_to_imu(date_name))

        return sample
