import pytest
import numpy as np
import math
from waymo_utils import generate_split, generate_lidar_point_coord_camera_image, rgba
from waymodataloader import WaymoDataset

def setup_module():
    global dataset
    dataset = WaymoDataset("C:/Users/alexj/Desktop/Waymo/data")
    dataset.reset_split(split=0.7, seed=3)

class TestWaymoDataloader:
    def test_length(self):
        assert len(dataset) == 3316
    def test_get_item(self):
        assert dataset[0]['front_camera_trigger_time'] == 1550083467.369629
        assert dataset[0]['front_camera_readout_done_time'] == 1550083467.423879
        assert dataset[1]['front_left_camera_trigger_time'] == 1550083467.4571795
        assert dataset[1]['front_left_camera_readout_done_time'] == 1550083467.511357
        assert dataset[100]['side_left_camera_trigger_time'] == 1550083481.7473109
        assert dataset[100]['side_left_camera_readout_done_time'] == 1550083481.801485
        assert dataset[200]['front_right_camera_trigger_time'] == 1552440203.9000068
        assert dataset[200]['front_right_camera_readout_done_time'] == 1552440203.950837
        assert dataset[300]['side_right_camera_trigger_time'] == 1510593603.6944396
        assert dataset[300]['side_right_camera_readout_done_time'] == 1510593603.739335
        assert dataset[0]['lidar_start_capture_timestamp'] == 1550083467346370
        assert dataset[0]['front_shape'][0] == 1280
        assert dataset[1]['front_left_shape'][1] == 1920
        assert dataset[123]['side_left_shape'][2] == 3
        assert dataset[200]['front_right_shape'][1] == 1920
        assert dataset[123]['side_right_shape'][0] == 886
        assert math.isclose(dataset[0]['lidar_point_coord'][0][0][0], -19.172607, rel_tol=0.00001)
        assert dataset[0]['camera_proj_point_coord'][1][0][1] == 236
    def test_waymo_utils(self):
        lidar_on_cam = generate_lidar_point_coord_camera_image(dataset[0]['frame'],dataset[0]['lidar_point_coord'],dataset[0]['camera_proj_point_coord'],'front')
        assert math.isclose(lidar_on_cam[0][1],509,rel_tol=0.00001)
        assert math.isclose(rgba(lidar_on_cam[0][2])[1], 0.5671750181554105, rel_tol=0.00001)