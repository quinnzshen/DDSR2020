import torch
from torch.utils.data import Dataset
import os
import tensorflow as tf
import sys
import waymo_utils as wu
import matplotlib.pyplot as plt
#If possible, use pip install waymo-open-dataset. If that doesn't work, clone the repo add at it to your path.
sys.path.append("C:/Users/alexj/Documents/GitHub/waymo-od")
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
  
class WaymoDataset(Dataset):
    def __init__(self,root_dir):
        """
        Initializes the Dataset, just given the root directory of the data and sets the original values of the instance
        variables. 
        Assumes that the data is in TFRecord format and stored in one folder
        :param root_dir: string containing the path to the root directory
        """
        self.root_dir = root_dir
        self.len = -1
        self.train_dir = []
    
    def set_up(self):
        """
        Sets up self.len and self.train_dir containing all the directories and indices. (Only called once per instance.)
        """
        with open(os.path.join(self.root_dir, wu.SPLIT_NAMES[0]), "r") as train:
            for line in train:
                self.train_dir.append(line.split())
        self.len = len(self.train_dir)
   
    def reset_split(self, split=0.7, seed=None):
        """
        Resets the split files in the dataset based on the given split probability and seed. If the seed is not given,
        the seed is randomly chosen.
        :param split: The chance of a given frame being put into train
        :param seed: The seed of the RNG, if None, then it is random
        """
        wu.generate_split(self.root_dir, split, seed)
        self.set_up()    
    
    def __len__(self):
        """
        Returns the total frame count in the dataset. If the length has not been calculated yet, runs set_len, else
        it simply returns the previously calculated length.
        :return: The frame count in the dataset
        """
        if self.len < 0:
            print("hi")
            self.set_up()
        return self.len

    def __getitem__(self, item):
        """
        Given a specific index (starting from 0), returns a dictionary containing information about the sample (frame) at that index in
        the dataset.
        (If the index is n, returns the nth frame of the dataset and its information.)
        The dictionary's fields are specified in Dataset Fields (Google Sheet file).

        :param item: An int representing the index of the sample to be retrieved
        :return: A dictionary containing fields about the retrieved sample
        """

        if self.len < 0:
            self.set_up()

        path_name = self.train_dir[item][0]
        item = int(self.train_dir[item][1])
        
        if torch.is_tensor(item):
            item = item.tolist()

        if item >= self.len or item < 0:
            raise IndexError("Dataset index out of range. (Less than 0 or greater than or equal to length)")
       	
       	#Gets relevant data from TFRecord
        item_data = tf.data.TFRecordDataset(path_name, compression_type='')
        count = 0
        for data in item_data:
            if (count == item):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                (range_images, camera_projections,
                range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,range_images, camera_projections, range_image_top_pose)
                break
            count+=1
        
        sample = {}
        
        #Frame data
        sample['frame'] = frame
        
        #Camera Images            
        sample['front'] = frame.images[0].image 
        sample['front_left'] = frame.images[1].image 
        sample['side_left'] = frame.images[2].image 
        sample['front_right'] = frame.images[3].image
        sample['side_right'] = frame.images[4].image
        
        #Camera shapes
        sample['front_shape'] = tf.shape(wu.conv_to_image(frame.images[0].image)).numpy() #0
        sample['front_left_shape'] = tf.shape(wu.conv_to_image(frame.images[1].image)).numpy()
        sample['side_left_shape'] = tf.shape(wu.conv_to_image(frame.images[2].image)).numpy()
        sample['front_right_shape'] = tf.shape(wu.conv_to_image(frame.images[3].image)).numpy()
        sample['side_right_shape'] = tf.shape(wu.conv_to_image(frame.images[4].image)).numpy()
        
        #Camera time (seconds since Unix Epoch)
        sample['front_camera_trigger_time'] = frame.images[0].camera_trigger_time
        sample['front_camera_readout_done_time'] = frame.images[0].camera_readout_done_time
        sample['front_left_camera_trigger_time'] = frame.images[1].camera_trigger_time
        sample['front_left_camera_readout_done_time'] = frame.images[1].camera_readout_done_time
        sample['side_left_camera_trigger_time'] = frame.images[2].camera_trigger_time
        sample['side_left_camera_readout_done_time'] = frame.images[2].camera_readout_done_time
        sample['front_right_camera_trigger_time'] = frame.images[3].camera_trigger_time
        sample['front_right_camera_readout_done_time'] = frame.images[3].camera_readout_done_time 
        sample['side_right_camera_trigger_time'] = frame.images[4].camera_trigger_time
        sample['side_right_camera_readout_done_time'] = frame.images[4].camera_readout_done_time
        
        #Boundary boxes        
        sample['projected_lidar_labels'] = frame.projected_lidar_labels
        
        #Start timestamp of the first top lidar spin (microseconds since unix epoch)
        sample['lidar_start_capture_timestamp'] = frame.timestamp_micros
        
        #Data from all 5 LiDAR sensors, format is an array of five arrays
        sample['lidar_point_coord']= points
        sample['camera_proj_point_coord'] = cp_points
    
        return sample