from __future__ import absolute_import, division, print_function

import os
import numpy as np

def load_velodyne_points(filename):
    """
    Load 3D point cloud from KITTI file format.
    """
    return np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
