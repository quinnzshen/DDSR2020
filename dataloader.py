import numpy as np
from torch.utils.data import Dataset

import os
from glob import glob
import random

from utils import get_nsec_times, get_camera

SPLIT_NAMES = ["train.txt", "validate.txt"]


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initializes the Dataset, just given the root directory of the data and sets the original values of the instance
        variables.
        :param root_dir: string containing the path to the root directory
        """
        self.root_dir = root_dir
        self.len = -1
        self.train_dir = []

    def generate_split(self, split=0.7, seed=None):
        """
        Generates the train.txt and validate.txt by using random for every frame and deciding whether to put it in
        train.txt or validate.txt.
        :param split: The chance of a given frame being put into train
        :param seed: The seed of the RNG, if None, then it is random
        """
        random.seed(seed)

        with open(os.path.join(self.root_dir, SPLIT_NAMES[0]), "w") as train:
            with open(os.path.join(self.root_dir, SPLIT_NAMES[1]), "w") as val:

                for direc in glob(self.root_dir + "/*/"):
                    # iterating through all date folders
                    for sub_dir in glob(direc + "/*/"):
                        # iterating through all date_drive folders
                        with open(os.path.join(sub_dir, "velodyne_points/timestamps.txt")) as file:
                            subtotal = 0
                            for _ in file:
                                line = sub_dir + " {}\n".format(subtotal)
                                if random.random() < split:
                                    train.write(line)
                                    self.train_dir.append([sub_dir, subtotal])
                                else:
                                    val.write(line)
                                subtotal += 1

        self.len = len(self.train_dir)

    def set_up(self):
        """
        Sets up the self.train_dir containing all the directory once called. (Only called once per instance.)
        """
        with open(os.path.join(self.root_dir, SPLIT_NAMES[0]), "r") as train:
            for line in train:
                self.train_dir.append(line.split())
        self.len = len(self.train_dir)

    def __len__(self):
        """
        Returns the total frame count in the dataset. If the length has not been calculated yet, runs set_len, else
        it simply returns the previously calculated length.
        :return: The frame count in the dataset
        """
        if self.len < 0:
            self.set_up()
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
        if self.len < 0:
            self.set_up()

        if item >= self.len or item < 0:
            raise IndexError("Dataset index out of range. (Less than 0 or greater than or equal to length)")

        path_name = self.train_dir[item][0]
        date_name = os.path.dirname(path_name)
        item = int(self.train_dir[item][1])

        # Taking information from the directory and putting it into the sample dictionary
        sample = {**get_camera(path_name, "stereo_left", item), **get_camera(path_name, "stereo_right", item)}

        nsec_times = get_nsec_times(path_name, item)
        sample["stereo_left_capture_time_nsec"] = nsec_times[0]
        sample["stereo_right_capture_time_nsec"] = nsec_times[1]
        sample["lidar_start_capture_timestamp_nsec"] = nsec_times[2]
        sample["lidar_end_capture_timestamp_nsec"] = nsec_times[3]

        # Getting the LiDAR coordinates
        lidar_points = np.fromfile(os.path.join(path_name, "velodyne_points/data/") + f"{item:010}" + ".bin", dtype=np.float32).reshape((-1, 4))
        sample["lidar_point_coord_velodyne"] = lidar_points[:, :3]
        sample["lidar_point_reflectivity"] = lidar_points[:, 3]

        return sample
