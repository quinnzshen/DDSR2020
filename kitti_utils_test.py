import pytest
import numpy as np

import os
import math

import kitti_utils as ku


def test_load_lidar_points():
    """
    Tests the return of the load_lidar_points funtion in kitti_utils.py
    """
    SAMPLE_LIDAR_POINTS_PATH = 'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000010.bin'
    lidar_point_coord_velodyne = ku.load_lidar_points(SAMPLE_LIDAR_POINTS_PATH)
    assert lidar_point_coord_velodyne.shape == (116006, 4)
    assert math.isclose(lidar_point_coord_velodyne[0][0], 73.89, rel_tol = .00001)
    assert math.isclose(lidar_point_coord_velodyne[0][1], 7.028, rel_tol = .00001)


def test_read_calibration_file():
    """
    Tests the return of the read_calibration_file funtion in kitti_utils.py
    """
    CALIBRATION_DIR = 'data/kitti_example/2011_09_26'
    cam2cam = ku.read_calibration_file(os.path.join(CALIBRATION_DIR, 'calib_cam_to_cam.txt'))
    velo2cam = ku.read_calibration_file(os.path.join(CALIBRATION_DIR, 'calib_velo_to_cam.txt'))
    imu2cam = ku.read_calibration_file(os.path.join(CALIBRATION_DIR, 'calib_imu_to_velo.txt'))
    assert len(cam2cam) == 34
    assert len(velo2cam) == 5
    assert len(imu2cam) == 3


def test_compute_image_from_velodyne_matrices():
    """
    Tests the return of the compute_image_from_velodyne_matrices funtion in kitti_utils.py
    """
    CALIBRATION_DIR = 'data/kitti_example/2011_09_26'
    camera_image_from_velodyne_dict = ku.compute_image_from_velodyne_matrices(CALIBRATION_DIR)
    assert len(camera_image_from_velodyne_dict) == 4
    camera_image_from_velodyne = camera_image_from_velodyne_dict.get('cam00')
    # Check shape.
    testarr = np.array([[609.695409, -721.421597, -1.25125855, -167.899086], \
                        [180.384202,  7.64479802, -719.651474, -101.233067], \
                        [.999945389,  .000124365378,  .0104513030, -.272132796], \
                        [0., 0., 0., 1.]])
    assert np.allclose(camera_image_from_velodyne, testarr)


def test_iso_string_to_nanoseconds():
    assert ku.iso_string_to_nanoseconds("2011-09-26 14:14:11.435280384") == 1317046451435280384
    assert ku.iso_string_to_nanoseconds("2021-09-16 00:00:00.010000001") == 1631750400010000001


def test_get_timestamp_nsec():
    assert ku.get_timestamp_nsec(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync\image_03\timestamps.txt", 3) == 1317046451221580544
    assert ku.get_timestamp_nsec(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync\image_02\timestamps.txt", 5).dtype == np.int64


def test_get_camera_data():
    cam_data = ku.get_camera_data(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", ["stereo_left", "stereo_right"], 3)
    assert type(cam_data) == dict
    assert cam_data["stereo_left_image"].dtype == np.uint8
    assert cam_data["stereo_left_image"].shape == (375, 1242, 3)
    assert cam_data["stereo_right_capture_time_nsec"] == 1317046451221580544


def test_get_lidar_data():
    lidar_data = ku.get_lidar_data(r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync", 6)
    assert type(lidar_data) == dict
    assert lidar_data["lidar_point_coord_velodyne"].shape == (114395, 3)
    assert lidar_data["lidar_point_reflectivity"].dtype == np.float32
    assert lidar_data["lidar_start_capture_time_nsec"].dtype == np.int64
    assert lidar_data["lidar_end_capture_time_nsec"] == 1317046451573549201
