import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
import tensorflow as tf
import sys

#Couldn't import the Waymo package using pip install so I downloaded it and added it to my path
sys.path.append("/Users/alexj/Documents/GitHub/waymo-od")
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

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
        self.record_divisions = []
   
    def __len__(self):
        """
        Returns the total frame count in the dataset. If the length has not been calculated yet, runs set_len, else
        it simply returns the previously calculated length.
        :return: The frame count in the dataset
        """
        if self.len < 0:
            self.set_len()
        return self.len
    def set_len(self):
        """
        Sets self.len, self.date_divisions, and self.drive_divisions by iterating through the whole dataset, preparing
        for future retrivals.
        Only called once after object initialization, as all the instance variables never change.
        """
        total = 0;
        for filename in os.listdir(self.root_dir):
            subtotal = 0;
            if filename.startswith("segment-"):
                subtotal = sum(1 for _ in tf.data.TFRecordDataset(self.root_dir+'/'+filename))
                total+=subtotal
                self.record_divisions.append(subtotal)
        self.len = total
    def __getitem__(self, item):
        """
        Given a specific index (starting from 0), returns a dictionary containing information about the sample (frame) at that index in
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
        
        #Finds what record file it's in
        record_index = 0
        count = 0
        for num in self.record_divisions:
            if count+num>item:
                break
            count += num
            record_index+=1
        item-= count
        print(item)
        
        #Finds the filename of the TFRecord file it's in
        count = 0
        for file in os.listdir(self.root_dir):
            if file.startswith("segment-"):
                if count == record_index:
                    filename = file
                    break
                count += 1
       	
       	#Displays the targeted frames with labels using mathplot lib
        item_data = tf.data.TFRecordDataset(self.root_dir + '/' + filename, compression_type='')
        count = 0
        for data in item_data:
            if (count == item):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                (range_images, camera_projections,
                range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
                frame)
                plt.figure(figsize=(25, 20))
                break
            count+=1
        for index, image in enumerate(frame.images):
            show_camera_image(image, frame.camera_labels, [3, 3, index+1])
        
        #Still a work in progress            
        sample = {}
        sample['a'] = 1
        return sample

#Testing
test = WaymoDataset(os.getcwd()+"/data")
print(test[198]['a'])    
print(test[201]['a'])    
print(test[500]['a'])  