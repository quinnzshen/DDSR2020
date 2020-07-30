import os
import math
import numpy as np
import pandas as pd
import pytest
import torch

from dataloader import KittiDataset
import kitti_utils as ku

EXAMPLE_SCENE_PATH = 'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/'
EXAMPLE_CALIBRATION_DIR = 'data/kitti_example/2011_09_26'


def test_load_lidar_points():
    SAMPLE_LIDAR_POINTS_PATH = 'data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000010.bin'
    lidar_point_coord_velodyne = ku.load_lidar_points(SAMPLE_LIDAR_POINTS_PATH)
    assert lidar_point_coord_velodyne.shape == (116006, 4)
    assert math.isclose(lidar_point_coord_velodyne[0][0], 73.89, rel_tol=.00001)
    assert math.isclose(lidar_point_coord_velodyne[0][1], 7.028, rel_tol=.00001)


def test_read_calibration_file():
    cam2cam = ku.read_calibration_file(os.path.join(EXAMPLE_CALIBRATION_DIR, 'calib_cam_to_cam.txt'))
    velo2cam = ku.read_calibration_file(os.path.join(EXAMPLE_CALIBRATION_DIR, 'calib_velo_to_cam.txt'))
    imu2cam = ku.read_calibration_file(os.path.join(EXAMPLE_CALIBRATION_DIR, 'calib_imu_to_velo.txt'))
    assert len(cam2cam) == 34
    assert len(velo2cam) == 5
    assert len(imu2cam) == 3


def test_compute_image_from_velodyne_matrices():
    camera_image_from_velodyne_dict = ku.compute_image_from_velodyne_matrices(EXAMPLE_CALIBRATION_DIR)
    assert len(camera_image_from_velodyne_dict) == 2
    camera_image_from_velodyne = camera_image_from_velodyne_dict.get('stereo_left')
    testarr = np.array([[613.040929, -718.575854, -2.95002805, -124.072003], \
                        [182.759005, 12.2395125, -718.988552, -101.607812], \
                        [.999893357, .00469739411, .0138291498, -.269119537], \
                        [0., 0., 0., 1.]])
    assert np.allclose(camera_image_from_velodyne, testarr)


def test_iso_string_to_nanoseconds():
    assert ku.iso_string_to_nanoseconds("2011-09-26 14:14:11.435280384") == 1317046451435280384
    assert ku.iso_string_to_nanoseconds("2021-09-16 00:00:00.010000001") == 1631750400010000001


def test_get_timestamp_nsec():
    assert ku.get_timestamp_nsec("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_03/timestamps.txt",
                                 3) == 1317046451221580544
    assert ku.get_timestamp_nsec("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/timestamps.txt",
                                 5).dtype == np.int64


def test_get_camera_data():
    cam_data = ku.get_camera_data("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 3)
    assert type(cam_data) == dict
    assert cam_data["stereo_left_image"].dtype == torch.uint8
    assert cam_data["stereo_left_image"].shape == (375, 1242, 3)
    assert cam_data["stereo_right_capture_time_nsec"] == 1317046451221580544


def test_get_lidar_data():
    lidar_data = ku.get_lidar_data("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 6)
    assert type(lidar_data) == dict
    assert lidar_data["lidar_point_coord_velodyne"].shape == (114395, 3)
    assert lidar_data["lidar_point_reflectivity"].dtype == torch.float32
    assert lidar_data["lidar_start_capture_time_nsec"].dtype == np.int64
    assert lidar_data["lidar_end_capture_time_nsec"] == 1317046451573549201


def test_get_imu_data():
    imu_data = ku.get_imu_data(EXAMPLE_SCENE_PATH, 0)
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
    imu_df = ku.get_imu_dataframe(EXAMPLE_SCENE_PATH)
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

    return pd.DataFrame(test_data, columns=['path_name', 'index'])


def test_get_nearby_frames(kitti_root_directory, kitti_dataset_index):
    """
    Tests the return of get_nearby_frames in the kitti_utils.py
    """
    dataset = KittiDataset(root_dir=kitti_root_directory,
                           dataset_index=kitti_dataset_index,
                           previous_frames=2,
                           next_frames=2)

    # On index 0, we expect there to be data for the relative index +1 and an empty dictionary for the relative index -1
    expected_fields = ['camera_data', 'pose']
    expected_camera_data_fields = ['stereo_left_image',
                                   'stereo_left_shape',
                                   'stereo_left_capture_time_nsec',
                                   'stereo_right_image', 'stereo_right_shape',
                                   'stereo_right_capture_time_nsec']

    data = dataset[0]
    # When idx = 0, [nearby_frames] keys: -1 and -2 should return empty dictionaries
    assert data['nearby_frames'][-1]['camera_data'] == {}
    assert data['nearby_frames'][-1]['pose'] == {}
    assert data['nearby_frames'][-2]['camera_data'] == {}
    assert data['nearby_frames'][-2]['pose'] == {}
    # Keys for [nearby_frames] should be int values within range(-previous_frames, next_frames + 1) with exception of 0
    assert list(data['nearby_frames'].keys()) == [-2, -1, 1, 2]
    # Values of valid [nearby_frames] keys should be elements of [expected_fields]
    assert list(data['nearby_frames'][1].keys()) == expected_fields
    assert list(data['nearby_frames'][1]['camera_data'].keys()) == expected_camera_data_fields
    # Values of invalid [nearby_frames] keys should be empty dictionaries
    assert list(data['nearby_frames'][-1]['camera_data'].keys()) == []

    data = dataset[1]
    # When idx = 0, [nearby_frames] keys: -1 should return camera data, while -2 should return an empty dictionary
    assert data['nearby_frames'][-1]['camera_data'] != {}
    assert data['nearby_frames'][-2]['camera_data'] == {}
    # Keys for [nearby_frames] should be int values within range(-previous_frames, next_frames + 1) with exception of 0
    assert list(data['nearby_frames'].keys()) == [-2, -1, 1, 2]
    # Values of valid [nearby_frames] keys should be elements of [expected_fields]
    assert list(data['nearby_frames'][-1].keys()) == expected_fields
    assert list(data['nearby_frames'][1]['camera_data'].keys()) == expected_camera_data_fields


def test_get_camera_intrinsic_dict():
    sample_cam_intrinsic_dict = ku.get_camera_intrinsic_dict(EXAMPLE_CALIBRATION_DIR)
    assert len(sample_cam_intrinsic_dict) == 2
    test_arr = torch.tensor([[959.791, 0., 696.0217], [0., 956.9251, 224.1806], [0., 0., 1.]], dtype=torch.double)
    torch.testing.assert_allclose(sample_cam_intrinsic_dict["stereo_left"], test_arr)


def test_get_relative_rotation_stereo():
    rel_rotation_sample = ku.get_relative_rotation_stereo(EXAMPLE_CALIBRATION_DIR)
    test_arr = torch.tensor(
        [[.9995572, -.02222673, .01978616], [.02225614, .99975152, -.00126738], [-.01975307, .00170718, .99980338]], dtype=torch.double)
    torch.testing.assert_allclose(rel_rotation_sample, test_arr)


def test_get_relative_translation_stereo():
    rel_translation_sample = ku.get_relative_translation_stereo(EXAMPLE_CALIBRATION_DIR)
    test_arr = torch.tensor([-0.53267121, 0.00526146, -0.00782809], dtype=torch.double)
    torch.testing.assert_allclose(rel_translation_sample, test_arr)


def test_get_relative_pose():
    pose1 = ku.get_relative_pose(EXAMPLE_SCENE_PATH, 0, 0)
    torch.testing.assert_allclose(pose1, torch.eye(4))
    pose2 = ku.get_relative_pose(EXAMPLE_SCENE_PATH, 3, 4)
    torch.testing.assert_allclose(pose2, torch.tensor([[1.0000, -0.0072, -0.0025, 0.0045],
                                                       [0.0072, 1.0000, 0.0028, -0.0030],
                                                       [0.0025, -0.0028, 1.0000, -0.8254],
                                                       [0.0000, 0.0000, 0.0000, 1.0000]]), atol=0.01, rtol=0.0001)
