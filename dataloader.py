import numpy as np
from torch.utils.data import Dataset
import yaml
import pandas as pd

import os

from utils import get_camera_data, get_lidar_data, generate_split


class KittiDataset(Dataset):
    def __init__(self, root_dir, pathdf):
        """
        Initializes the Dataset, just given the root directory of the data and sets the original values of the instance
        variables.
        :param root_dir: string containing the path to the root directory
        """
        self.root_dir = root_dir
        self.pathdf = pathdf

    @classmethod
    def init_from_config(cls, config_path):
        with open(config_path, "r") as yml:
            config = yaml.load(yml, Loader=yaml.Loader)
            path_df = pd.concat([pd.read_csv(path, sep=" ", header=None) for path in config["dataset_paths"]])
        return cls(config["root_directory"], path_df)

    def reset_split(self, split=0.7, seed=None):
        """
        Resets the split files in the dataset based on the given split probability and seed. If the seed is not given,
        the seed is randomly chosen.
        :param split: The chance of a given frame being put into train
        :param seed: The seed of the RNG, if None, then it is random
        """
        generate_split(self.root_dir, split, seed)
        self.set_up()

    def __len__(self):
        """
        Returns the total frame count in the dataset. If the length has not been calculated yet, runs set_len, else
        it simply returns the previously calculated length.
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
