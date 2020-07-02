import numpy as np
from torch.utils.data import Dataset
import yaml
import pandas as pd

import os

from kitti_utils import get_camera_data, get_lidar_data, get_nearby_frames_data


class KittiDataset(Dataset):
    def __init__(self, root_dir, dataset_index, previous_frames, next_frames):
        """
        Initializes the Dataset, given the root directory of the data and a dataframe of the paths to the dataset.
        :param root_dir [str]: string containing the path to the root directory
        :param dataset_index [pd.DataFrame]: The dataframe containing the paths and indices of the data
        """
        self.root_dir = root_dir
        self.dataset_index = dataset_index
        self.previous_frames = previous_frames
        self.next_frames = next_frames

    @classmethod
    def init_from_config(cls, config_path):
        """
        Creates an instance of the class using a config file. The config file supplies the paths to the text files
        containing the all the paths to the data.
        :param [str] config_path: The path to the config file
        :return [KittiDataset]: The object instance
        """
        with open(config_path, "r") as yml:
            config = yaml.load(yml, Loader=yaml.Loader)
            dataset_index = pd.concat([pd.read_csv(path, sep=" ", header=None) for path in config["dataset_paths"]])
        return cls(root_dir=config["root_directory"], 
                   dataset_index=dataset_index, 
                   previous_frames=config["previous_frames"], 
                   next_frames=config["next_frames"])

    def __len__(self):
        """
        Returns the total frame count in the dataset.
        :return [int]: The frame count in the dataset
        """
        return len(self.dataset_index)

    def __getitem__(self, idx):
        """
        Given a specific index, returns a dictionary containing information about the sample (frame) at that index in
        the dataset.
        (If the index is n, returns the nth frame of the dataset and its information.)
        The dictionary's fields are specified in Dataset Fields (Google Sheet file).

        :param [int] idx: An int representing the index of the sample to be retrieved
        :return [dict]: A dictionary containing fields about the retrieved sample
        """

        if idx >= len(self.dataset_index) or idx < 0:
            raise IndexError(f"Dataset index out of range. Given: {idx} (Less than 0 or greater than or equal to length)")

        path_name = os.path.normpath(self.dataset_index.iloc[idx, 0])
        date_name = os.path.dirname(path_name)
        idx = int(self.dataset_index.iloc[idx, 1])

        nearby_frames_data = get_nearby_frames_data(path_name, idx, self.previous_frames, self.next_frames)
        # Taking information from the directory and putting it into the sample dictionary
        sample = {
            **get_camera_data(path_name, idx),
            **get_lidar_data(path_name, idx),
            **{'nearby_frames': nearby_frames_data},
        }

        return sample
