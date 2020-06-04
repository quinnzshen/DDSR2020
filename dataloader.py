import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import os


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.len = -1
        self.num_frame_sum = []

    def set_len(self):
        total = 0
        dir_list = os.listdir(self.root_dir)

        for direc in dir_list:
            # iterating through all date folders
            level_1_path = os.path.join(self.root_dir, direc)
            if os.path.isdir(level_1_path):
                for sub_dir in os.listdir(level_1_path):
                    # iterating through all date_drive folders
                    level_2_path = os.path.join(level_1_path, sub_dir)
                    if os.path.isdir(level_2_path):
                        for sub_sub_dir in os.listdir(level_2_path):
                            # iterating through all date_drive.zip folders
                            velo_folder = os.path.join(os.path.join(level_2_path, sub_sub_dir), "velodyne_points")

                            if os.path.isdir(velo_folder):
                                with open(os.path.join(velo_folder, "timestamps.txt")) as file:
                                    for i, l in enumerate(file):
                                        pass
                                    total += i + 1
                                    self.num_frame_sum.append(total)
                                break

        self.len = total

    def __len__(self):
        if self.len > 0:
            return self.len
        self.set_len()
        return self.len

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.len < 0:
            self.set_len()
        # assumes item is an integer
        if item >= self.len or item < 0:
            raise IndexError("Dataset index out of range. (Less than 0 or greater than or equal to length)")

        index = self.len // 2
        lower_index = 0
        upper_index = self.len - 1

        while 1:
            if self.num_frame_sum[index] > item:
                pass





