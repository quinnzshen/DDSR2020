import naive_depth_completion as ndc
import numpy as np
import pytest

def test_get_dense_depth_map_from_lidar():
    sample_lidar_points = np.array([[1., 1., 30., 1.], [1., 0., 50., 1.]])
    sample_dense_depth_map = ndc.get_dense_depth_map_from_lidar(sample_lidar_points, 2, 2)
    assert np.allclose(sample_dense_depth_map, np.array([[1., 1.], [0., 0.]]))
