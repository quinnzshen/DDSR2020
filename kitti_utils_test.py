import os
import math
import numpy as np
import pandas as pd
import pytest

from dataloader import KittiDataset
import kitti_utils as ku


SAMPLE_SCENE_PATH = 'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/'

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
    assert ku.get_timestamp_nsec(r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_03/timestamps.txt", 3) == 1317046451221580544
    assert ku.get_timestamp_nsec(r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/timestamps.txt", 5).dtype == np.int64


def test_get_camera_data():
    cam_data = ku.get_camera_data(r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 3)
    assert type(cam_data) == dict
    assert cam_data["stereo_left_image"].dtype == np.uint8
    assert cam_data["stereo_left_image"].shape == (375, 1242, 3)
    assert cam_data["stereo_right_capture_time_nsec"] == 1317046451221580544


def test_get_lidar_data():
    lidar_data = ku.get_lidar_data(r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 6)
    assert type(lidar_data) == dict
    assert lidar_data["lidar_point_coord_velodyne"].shape == (114395, 3)
    assert lidar_data["lidar_point_reflectivity"].dtype == np.float32
    assert lidar_data["lidar_start_capture_time_nsec"].dtype == np.int64
    assert lidar_data["lidar_end_capture_time_nsec"] == 1317046451573549201

    
def test_get_imu_data():
    imu_data = ku.get_imu_data(SAMPLE_SCENE_PATH, 0)
    expected_imu_data = {
        'lat': '49.030860615858',
        'lon': '8.3397493123379',
        'alt': '114.53007507324',
        'roll': '0.050175',
        'pitch': '0.003312',
        'yaw': '-0.9314506732051',
        'vn': '-5.903827060491',
        've': '4.4003650005689',
        'vf': '7.3633076185673',
        'vl': '-0.0064749380045055',
        'vu': '0.042500049108082',
        'ax': '0.44636518303217',
        'ay': '1.3095214718555',
        'az': '9.5246768612237',
        'af': '0.46047950971548',
        'al': '0.81812852972705',
        'au': '9.5787585828325',
        'wx': '-0.022391715154777',
        'wy': '0.031519786498138',
        'wz': '-0.0059401645007176',
        'wf': '-0.022403169760322',
        'wl': '0.031778651415645',
        'wu': '-0.0042553567211989',
        'pos_accuracy': '0.11751595636338',
        'vel_accuracy': '0.023345235059858',
        'navstat': '4',
        'numsats': '7',
        'posmode': '5',
        'velmode': '5',
        'orimode': '6'
    }

    assert len(imu_data) == 30
    assert imu_data == expected_imu_data


def test_get_imu_dataframe():
    imu_df = ku.get_imu_dataframe(SAMPLE_SCENE_PATH)
    assert imu_df.shape == (22, 30)

@pytest.fixture
def kitti_root_directory():
    return 'data/kitti_example'

@pytest.fixture
def kitti_dataset_index():
    test_data = {'path_name': ['data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/', 
                               'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/',
                               'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/'],
                 'index': ['0', '1', '2']}
    
    return pd.DataFrame(test_data, columns = ['path_name', 'index'])

def test_get_nearby_frames(kitti_root_directory, kitti_dataset_index):
    """
    Tests the return of get_nearby_frames in the kitti_utils.py
    """
    dataset = KittiDataset(root_dir=kitti_root_directory, 
                           dataset_index=kitti_dataset_index, 
                           previous_frames=2,
                           next_frames=2)

    # On index 0, we expect there to be data for the relative index +1 and an empty dictionary for the relative index -1
    expected_fields = ['stereo_left_image', 
                       'stereo_left_shape', 
                       'stereo_left_capture_time_nsec', 
                       'stereo_right_image', 'stereo_right_shape', 
                       'stereo_right_capture_time_nsec']
    data = dataset[0]
    data1 = dataset[1]
    """
    When index is 0 the frames -1 and -2 relative to the index do not exist and should return empty dictionaries
    The keys returned for nearby_frames should be integers in the range(-previous_frames, next_frames + 1) excluding 0
    The value of valid keys within nearby_frames should should be keys for each of the expected_fields
    The value of invalid keys within nearby_frames should should return an empty list
    """
    assert data['nearby_frames'][-1] == {}
    assert data['nearby_frames'][-2] == {}
    assert list(data['nearby_frames'].keys()) == [-2, -1, 1, 2]
    assert list(data['nearby_frames'][1].keys()) == expected_fields
    assert list(data['nearby_frames'][-1].keys()) == []
     """
    When index is 1 the frames -1 relative to the index exists and should not longer return an empty dictionary,
    the frame -2 relative to the index do not exist and should return an empty dictionary
    The keys returned for nearby_frames should be integers in the range(-previous_frames, next_frames + 1) excluding 0
    The value of valid keys within nearby_frames should should be keys for each of the expected_fields
    """
    assert data1['nearby_frames'][-1] != {}
    assert data1['nearby_frames'][-2] == {}
    assert list(data1['nearby_frames'].keys()) == [-2, -1, 1, 2]
    assert list(data1['nearby_frames'][-1].keys()) == expected_fields
    

