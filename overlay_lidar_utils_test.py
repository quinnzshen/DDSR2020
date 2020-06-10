import unittest
import overlay_lidar_utils as olu
import numpy as np

class TestOverlayLidarUitls(unittest.TestCase):
    def test_generate_lidar_point_coord_camera_image(self):
        """
        Tests the return of the generate_lidar_point_coord_camera_image_test function in overlay_lidar_utils.py
        """
        lidar_point_coord_velodyne = np.array([[10, 10, 10], [10, 10, 10]])
        camera_image_from_velodyne = np.eye(4)
        lidar_point_coord_camera_image = olu.generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne, 100, 100)
        self.assertTrue(np.array_equal(lidar_point_coord_camera_image, np.array([[0., 0., 10., 1.], [0., 0., 10., 1.]])))

    def test_normalize_depth(self):
        """
        Tests the return of the normalize_depth function in overlay_lidar_utils.py
        """
        lidar_point_depth = np.array([[-1], [5], [9]])
        norm_depth = olu.normalize_depth(lidar_point_depth)
        self.assertTrue(np.array_equal(norm_depth, np.array([[0.], [0.6], [1.]])))
