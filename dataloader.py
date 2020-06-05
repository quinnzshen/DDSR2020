import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import os
from glob import glob


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.len = -1
        self.date_divisions = []
        self.drive_divisions = []
        self.imagefiles = []
        self.velodyne = root_dir+'/velodyne_points'
        
        #Store image files
        for filename in os.listdir(root_dir):
            if filename.startswith("image"):
                self.imagefiles.append(filename)



    def set_len(self):
        total = 0
        # numFiles =0
        dir_list = os.listdir(self.root_dir)
        for direc in glob(self.root_dir + "/*/"):
            self.date_divisions.append(total)
            # iterating through all date folders
            subtotal = 0
            sub_drive_divisions = []
            for sub_dir in glob(direc + "/*/"):
                # iterating through all date_drive folders
                with open(os.path.join(os.path.join(sub_dir, "velodyne_points"), "timestamps.txt")) as file:
                    for _ in file:
                        subtotal += 1

                sub_drive_divisions.append(subtotal)

            total += subtotal
            self.drive_divisions.append(sub_drive_divisions)

        self.len = total

    def __len__(self):
        if self.len < 0:
            self.set_len()
        return self.len
        # return total

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
        return 1


dataset = KittiDataset('data/kitti_example')
print(len(dataset))

# print(glob('data/kitti_example/2011_09_26/*/velodyne_points/'))