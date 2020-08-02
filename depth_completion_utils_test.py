import depth_completion_utils as dcu
import numpy as np
import pytest

def test_create_depth_map_from_nearest_lidar_point():
    sample_lidar_points = np.array([[1., 1., 30., 1.], [1., 0., 50., 1.]])
    sample_dense_depth_map = dcu.create_depth_map_from_nearest_lidar_point(sample_lidar_points, 2, 2)
    assert np.allclose(sample_dense_depth_map, np.array([[1., 1.], [0., 0.]]))

def test_create_depth_map_from_lidar_smoothing():
    sample_lidar_points = np.array([[1., 1., 30., 1.], [1., 0., 50., 1.]])
    sample_dense_depth_map = dcu.create_depth_map_from_lidar_smoothing(sample_lidar_points, 2, 2, 2)
    test_arr = np.array([0., 1.])
    assert np.allclose(test_arr, sample_dense_depth_map[0, :])
