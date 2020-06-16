import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
import sys
#If possible, use pip install waymo-open-dataset. If that doesn't work, clone the repo add at it to your path.
sys.path.append("C:/Users/alexj/Documents/GitHub/waymo-od")
from waymo_open_dataset import dataset_pb2 as open_dataset

SPLIT_NAMES = ["train.txt", "validate.txt"]

def generate_split(root_dir, split=0.7, seed=0):
    """ 
    Generates train.txt or validate.txt.
    :param root_dir: The root directory of the dataset
    :param split: The chance of a given frame being put into train
    :param seed: The seed of the RNG, if None, then it is random (default seed is 0)
    """ 
    random.seed(seed)
    with open(os.path.join(root_dir, SPLIT_NAMES[0]), "w") as train:
        with open(os.path.join(root_dir, SPLIT_NAMES[1]), "w") as val:
            for filename in os.listdir(root_dir):
                if filename.startswith("segment-"):
                    item_data = tf.data.TFRecordDataset(root_dir + '/' + filename, compression_type='')
                    count = 0
                    for data in item_data:
                        line = root_dir + '/' + filename + " {}\n".format(count)
                        if random.random()<split:
                            train.write(line)
                        else:
                            val.write(line)
                        count+=1

def conv_to_image(camera_image):
    """Converts a TFrecord image file to a standard image file
    Args
        camera_image : The TFrecord image file 
    """
    return tf.image.decode_jpeg(camera_image)

def rgba(r):
  """Generates a color based on range.
  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a camera image.
   Args:
    camera_image: The TFrecord image file
  """
  plt.figure(figsize=(20, 12))
  plt.imshow(conv_to_image(camera_image))
  plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))
    
  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
  
def generate_lidar_point_coord_camera_image(frame, points, cp_points, image_name):
    """Converts LiDAR coordinates to camera frame

    Args:
      frame: TFrecord frame object
      points: LiDAR points
      cp_points: Camera projections points
      image_name: the name of the image (e.g front, front_left, etc.)
    """
    
    image_name = image_name.upper()
    conversions = {
             'FRONT':0,
             'FRONT_LEFT':1,
             'SIDE_LEFT':2,
             'FRONT_RIGHT':3,
             'SIDE_RIGHT':4
         }
    image_number = conversions[image_name]
    points_all = np.concatenate(points, axis=0)
   
    cp_points_all = np.concatenate(cp_points, axis=0)
    images = frame.images
    
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    mask = tf.equal(cp_points_all_tensor[..., 0], images[image_number].name)
    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
    projected_points_all_from_raw_data = tf.concat(
    [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
   
    return projected_points_all_from_raw_data