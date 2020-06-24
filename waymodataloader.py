import torch
from torch.utils.data import Dataset
import os
import tensorflow as tf
import sys
#If possible, use pip install waymo-open-dataset. If that doesn't work, clone the repo add at it to your path.
sys.path.append("thirdparty/waymo-od")
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import pandas as pd
import yaml
from waymo_utils import get_camera_data, get_lidar_data
  
class WaymoDataset(Dataset):
    def __init__(self,root_dir, dataset_paths):
        """
        Initializes the Dataset, just given the root directory of the data and sets the original values of the instance
        variables. 
        Assumes that the data is in TFRecord format and stored in one folder
        :param root_dir: string containing the path to the root directory
        :param dataset_paths [pd.DataFrame]: the dataframe containing the paths and indicies of the data
        """
        self.root_dir = root_dir
        self.pathdf = dataset_paths
        self.train_dir = []
    
    @classmethod
    def init_from_config(cls, config_path):
        """
        Creates an instance of the class using a config file. The config file supplies the paths to the text files
        containing the all the paths to the data.
        :param [str] config_path: The path to the config file
        :return [WaymoDataset]: The object instance
        """
        with open(config_path,"r") as yml:
            config = yaml.load(yml, Loader=yaml.Loader)
            path_df = pd.concat([pd.read_csv(path, sep = " ", header = None) for path in config["dataset_paths"]])
            return cls(config["root_directory"],path_df)
    
    def __len__(self):
        """
        Returns the total frame count in the dataset
        :return [int]: Frame count in dataset
        """
        return len(self.pathdf)

    def __getitem__(self, idx):
        """
        Given a specific index (starting from 0), returns a dictionary containing information about the sample (frame) at that index in
        the dataset.
        (If the index is n, returns the nth frame of the dataset and its information.)
        The dictionary's fields are specified in Dataset Fields (Google Sheet file).

        :param idx: An int representing the index of the sample to be retrieved
        :return: A dictionary containing fields about the retrieved sample
        """
        
        # Data validation
        if idx >= len(self.pathdf) or idx < 0:
            raise IndexError("Dataset index out of range. (Less than 0 or greater than or equal to length)")
       	
        # Finding filename and index
        path_name = self.pathdf.iloc[idx,0]
        idx = int(self.pathdf.iloc[idx,1])

       	# Gets frame data from TFRecord
        item_data = tf.data.TFRecordDataset(path_name, compression_type='')
        count = 0
        for data in item_data:
            if (count == idx):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                break
            count+=1
        
        # Puts all info into the dictionary
        sample = {
            "frame":frame,
            **get_camera_data(frame),
            **get_lidar_data(frame)
            }

        return sample

data = WaymoDataset.init_from_config("waymoloader_test_config.yml")
print(data[0]['front_left_readout_done_time'])