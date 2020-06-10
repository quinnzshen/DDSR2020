from __future__ import absolute_import, division, print_function

import os
import numpy as np

def load_velodyne_points(filename):
    """
    Load 3D point cloud from KITTI file format.
    """
    lidar_point_coord_velodyne = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    #Delete bottom row of reflectance value so columns are just [X, Y, Z]
    lidar_point_coord_velodyne = lidar_point_coord_velodyne[:, :3]
    return lidar_point_coord_velodyne

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def compute_image_from_velodyne_matrix(calib_dir, cam):
    """
    This function computes the transformation matrix to project 3D lidar points into the 2D image plane.
    :param [String] calib_dir: Directory to folder containing camera/velodyne calibration files
    :param [int] cam: Camera # that matrix is being computed for (0, 1, 2, or 3)
    :return: numpy.array of shape [4,4] that converts 3D lidar pts to 2D image plane 
    """
    #Based on code from monodepth2 repo.
    
    #load cam_to_cam calib file
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    #load velo_to_cam file
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3,1)))#Adds T vals in 4th column.
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_0' + str(cam)].reshape(3,3)#Fills top left 3x3 with R vals.
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3,4)
    
    camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
    
    return camera_image_from_velodyne
