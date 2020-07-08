import pytest
import numpy as np
import math
from waymo_utils import generate_split, generate_lidar_point_coord_camera_image, rgba
from waymodataloader import WaymoDataset

TEST_CONFIG_PATH = "waymoloader_test_config.yml"

def setup_module():
    global dataset
    dataset = WaymoDataset.init_from_config("waymoloader_test_config.yml")

class TestWaymoDataloader:
    def test_length(self):
        assert len(dataset) == 2
    def test_get_item(self):
        assert dataset[0]['front_trigger_time'] == 1553640277.2346969
        assert dataset[0]['front_readout_done_time'] == 1553640277.279122
        assert dataset[1]['front_left_trigger_time'] == 1553640277.3225036
        assert dataset[1]['front_left_readout_done_time'] == 1553640277.367252
        assert dataset[0]['lidar_start_capture_timestamp'] == 1553640277206678
        assert dataset[0]['front_shape'][0] == 1280
        assert dataset[1]['front_left_shape'][1] == 1920
        assert dataset[0]['side_left_shape'][2] == 3
        assert dataset[0]['front_right_shape'][1] == 1920
        assert dataset[1]['side_right_shape'][0] == 886
        assert math.isclose(dataset[0]['lidar_point_coord'][0][0][0], -51.13867, rel_tol=0.00001)
        assert dataset[0]['camera_proj_point_coord'][1][0][1] == 849
    def test_waymo_utils(self):
        lidar_on_cam = generate_lidar_point_coord_camera_image(dataset[0]['frame'],dataset[0]['lidar_point_coord'],dataset[0]['camera_proj_point_coord'],'front')
        assert math.isclose(lidar_on_cam[0][1],530,rel_tol=0.00001)
        assert math.isclose(rgba(lidar_on_cam[0][2])[1], 0.48823529411764705, rel_tol=0.00001)