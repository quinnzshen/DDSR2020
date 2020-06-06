import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
from glob import glob


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


# Converts a line from timestamp.txt into nanoseconds from the start of the date
def time_to_nano(time_string):
    total = 0
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.len = -1
        self.date_divisions = []
        self.drive_divisions = []
        
        # for filename in os.listdir(root_dir):
        #     if filename.startswith("image"):
        #         self.imagefiles.append(filename)

    # Records the length and cumulative number of images of each directory for access later
    def set_len(self):
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

    # Calls set_len if hasn't been called prior, else just returns calculated len
    def __len__(self):
        if self.len < 0:
            self.set_len()
        return self.len
        # return total

    # Gets sample from given index, this is assuming item is an integer
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.len < 0:
            self.set_len()
        # assumes item is an integer
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

        # Path of date_drive folder
        path_name = glob(glob(self.root_dir + "/*/")[da_index] + "/*/")[dr_index]
                
        sample = {}

        # Just taking stuff from the directory and putting it into the sample dictionary
        img_arr = np.asarray(Image.open(os.path.join(path_name, "image_02/data/") + f"{item:010}" + ".png"))
        sample["stereo_left_image"] = img_arr
        sample["stereo_left_shape"] = img_arr.shape
        with open(os.path.join(path_name, "image_02/timestamps.txt")) as f:
            for i, line in enumerate(f):
                if i == item:
                    sample["stereo_left_capture_time_nsec"] = time_to_nano(line)
                    break

        img_arr = np.asarray(Image.open(os.path.join(path_name, "image_03/data/") + f"{item:010}" + ".png"))
        sample["stereo_right_image"] = img_arr
        sample["stereo_right_shape"] = img_arr.shape

        with open(os.path.join(path_name, "image_02/timestamps.txt")) as l_time:
            with open(os.path.join(path_name, "image_03/timestamps.txt")) as r_time:
                with open(os.path.join(path_name, "velodyne_points/timestamps_start.txt")) as start_time:
                    with open(os.path.join(path_name, "velodyne_points/timestamps_end.txt")) as end_time:
                        count = 0
                        for l_line, r_line, start_line, end_line in zip(l_time, r_time, start_time, end_time):
                            if count == item:
                                sample["stereo_left_capture_time_nsec"] = time_to_nano(l_line)
                                sample["stereo_right_capture_time_nsec"] = time_to_nano(r_line)
                                sample["lidar_start_capture_timestamp_nsec"] = time_to_nano(start_line)
                                sample["lidar_end_capture_timestamp_nsec"] = time_to_nano(end_line)
                                break
                            count += 1


        #Getting the LiDAR coordinates
        lidar_points = np.fromfile(os.path.join(path_name, "velodyne_points/data/") + f"{item:010}" + ".bin", dtype=np.float32)
        sample["lidar_point_coord_continuous"] = lidar_points.reshape((-1, 4))

        return sample


# Some testing you can ignore this I guess
if __name__ == "__main__":
    dataset = KittiDataset('data/kitti_example')
    # print(len(dataset))
    print(dataset[2]["stereo_right_shape"])
    print(dataset[2]["lidar_start_capture_timestamp_nsec"])
    print(dataset[2]["lidar_end_capture_timestamp_nsec"])
    print(dataset[2]["lidar_point_coord_continuous"])
    # print(glob('data/kitti_example/2011_09_26/*/velodyne_points/'))
