import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import pytest
import metrics as m
import kitti_utils as ku
import overlay_lidar_utils as olu

lidar_path = 'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000000.bin'
date_path = 'data/kitti_example/2011_09_26/'

def test_compute_l1_error():
    depth = np.ones((2, 2))
    lidar = np.array([[1, 1, 2, 1]])
    assert np.array_equal(m.compute_l1_error(depth, lidar), np.array([[1., 1., 1.]]))