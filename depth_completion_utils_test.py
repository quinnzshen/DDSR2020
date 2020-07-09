import depth_completion_utils as dcu
import numpy as np
import pytest

def test_create_depth_map_from_nearest_lidar_point():
    sample_lidar_points = np.array([[1., 1., 30., 1.], [1., 0., 50., 1.]])
    sample_dense_depth_map = dcu.create_depth_map_from_nearest_lidar_point(sample_lidar_points, 2, 2)
    assert np.allclose(sample_dense_depth_map, np.array([[1., 1.], [0., 0.]]))

def test_create_depth_map_from_lidar_smoothing():
    sample_lidar_points = np.array([[1., 1., 30., 1.], [1., 0., 50., 1.], [2., 2., 20., 1.]])
    sample_dense_depth_map = dcu.create_depth_map_from_lidar_smoothing(sample_lidar_points, 3, 3, 3)
    test_arr = np.eye(3)
    test_arr.fill(.666666666667)
    assert np.allclose(test_arr, sample_dense_depth_map)
test_create_depth_map_from_lidar_smoothing()
test_create_depth_map_from_nearest_lidar_point()