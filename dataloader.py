import numpy as np
from torch.utils.data import Dataset
import yaml
import pandas as pd

import os

from utils import get_camera_data, get_lidar_data


class KittiDataset(Dataset):
    def __init__(self, root_dir, pathdf):
        """
        Initializes the Dataset, given the root directory of the data and a dataframe of the paths to the dataset.
        :param root_dir: string containing the path to the root directory
        :param pathdf: the dataframe containing the paths and indices of the data
        """
        self.root_dir = root_dir
        self.pathdf = pathdf

    @classmethod
    def init_from_config(cls, config_path):
        """
        Creates an instance of the class using a config file. The config file supplies the paths to the text files
        containing the all the paths to the data.
        :param config_path: The path to the config file
        :return: The object instance
        """
        with open(config_path, "r") as yml:
            config = yaml.load(yml, Loader=yaml.Loader)
            path_df = pd.concat([pd.read_csv(path, sep=" ", header=None) for path in config["dataset_paths"]])
        return cls(config["root_directory"], path_df)

    def __len__(self):
        """
        Returns the total frame count in the dataset.
        :return: The frame count in the dataset
        """
        return len(self.pathdf)

    def __getitem__(self, idx):
        """
        Given a specific index, returns a dictionary containing information about the sample (frame) at that index in
        the dataset.
        (If the index is n, returns the nth frame of the dataset and its information.)
        The dictionary's fields are specified in Dataset Fields (Google Sheet file).

        :param idx: An int representing the index of the sample to be retrieved
        :return: A dictionary containing fields about the retrieved sample
        """

        if idx >= len(self.pathdf) or idx < 0:
            raise IndexError(f"Dataset index out of range. Given: {idx} (Less than 0 or greater than or equal to length)")

        path_name = self.pathdf.iloc[idx, 0]
        date_name = os.path.dirname(path_name)
        idx = int(self.pathdf.iloc[idx, 1])

        # Taking information from the directory and putting it into the sample dictionary
        sample = {
            **get_camera_data(path_name, "stereo_left", idx),
            **get_camera_data(path_name, "stereo_right", idx),
            **get_lidar_data(path_name, idx)
        }

        return sample
