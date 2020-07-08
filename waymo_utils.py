import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
import sys
sys.path.append('third_party/waymo_open_dataset')
from utils import frame_utils

SPLIT_NAMES = ["train.txt", "validate.txt"]
CAMERA_DICT = {
             'FRONT':0,
             'FRONT_LEFT':1,
             'SIDE_LEFT':2,
             'FRONT_RIGHT':3,
             'SIDE_RIGHT':4
         }
CAMERA_NAMES = ['front', 'front_left', 'side_left', 'front_right', 'side_right']

def get_camera_data(frame):
    """
    Gets the basic camera information given the path name to the scene, the camera name, and the frame number within
    that scene.
    :param [str] frame: A frame from a TFRecord file
    :return [dict]: A dictionary containing the image (in a NumPy array), the shape of that array, timestamps, intrinsics and extrinsics, and pose
    """
    dictionary = {}
    for camera_name in CAMERA_NAMES:
        camera_upper = camera_name.upper()
        dictionary.update(
            {
                f"{camera_name}_image": frame.images[CAMERA_DICT[camera_upper]].image,
                f"{camera_name}_shape": tf.shape(conv_to_image(frame.images[CAMERA_DICT[camera_upper]].image)).numpy(),
                f"{camera_name}_trigger_time": frame.images[CAMERA_DICT[camera_upper]].camera_trigger_time,
                f"{camera_name}_readout_done_time": frame.images[CAMERA_DICT[camera_upper]].camera_readout_done_time,
                f"{camera_name}_intrinsics": np.reshape(frame.context.camera_calibrations[CAMERA_DICT[camera_upper]].intrinsic, (3,3)),
                f"{camera_name}_extrinsics": np.reshape(frame.context.camera_calibrations[CAMERA_DICT[camera_upper]].extrinsic.transform, (4,4)),
                f"{camera_name}_pose": np.reshape(frame.images[0].pose.transform, (4,4))
            }
        )
    return dictionary

def get_lidar_data(frame):
    """
    Gets the basic LiDAR information given the path name to the scene and the frame number within that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing the points, reflectivity, start, and end times of the LiDAR scan.
    """
    (range_images, camera_projections,
    range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,range_images, camera_projections, range_image_top_pose)
    return {
        "lidar_point_coord": points,
        "camera_proj_point_coord": cp_points,
        "lidar_start_capture_timestamp": frame.timestamp_micros,
    }
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

def plot_sparse_image(lidar_point_coord_camera_image, image, shape):
    """
    Plots the pixels of an image with corresponding depth values
    :param:[numpy.array] lidar_point_coord_camera_image: [N,3], lidar coordinates in camera frame
    :param: [numpy.array] image: [H,W,3], contains R, G, B values for every pixel of the image
    :return: none, plots a sparse image containing pixels with corresponding depth values
    """
    image = conv_to_image(image)
    xs = []
    ys = []
    colors = []

    for point in lidar_point_coord_camera_image:
        if (point[0]<=shape[1] and point[1]<=shape[0]):
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colortuple = (image[int(point[1])][int(point[0])][0], image[int(point[1])][int(point[0])][1], image[int(point[1])][int(point[0])][2])
            colors.append('#%02x%02x%02x' % colortuple)
    plt.figure(figsize=(20, 12)).gca().invert_yaxis()
    plt.scatter(xs,ys,c=colors,s=10,marker='s')