import kitti_utils as ku
import os
import unittest
import numpy as np

class TestKittiUtils(unittest.TestCase):
    def test_load_velodyne_points(self):
        """
        Tests the return of the load_velodyne_points funtion in kitti_utils.py
        """
        lidar_point_coord_velodyne = ku.load_velodyne_points('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000010.bin')
        self.assertEqual(lidar_point_coord_velodyne.shape, (116006, 3))
        self.assertAlmostEqual(lidar_point_coord_velodyne[0][0], 73.89, 4)
        self.assertAlmostEqual(lidar_point_coord_velodyne[0][1], 7.028, 4)
        

    def test_read_calib_file(self):
        """
        Tests the return of the read_calib_file funtion in kitti_utils.py
        """
        calib_dir = 'data/kitti_example/2011_09_26'
        cam2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        imu2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
        self.assertEqual(len(cam2cam), 34)
        self.assertEqual(len(velo2cam), 5)
        self.assertEqual(len(imu2cam), 3)

    def test_compute_image_from_velodyne_matrices(self):
        """
        Tests the return of the compute_image_from_velodyne_matrices funtion in kitti_utils.py
        """
        calib_dir = 'data/kitti_example/2011_09_26'
        camera_image_from_velodyne_dict = ku.compute_image_from_velodyne_matrices(calib_dir)
        self.assertEqual(len(camera_image_from_velodyne_dict), 4)
        camera_image_from_velodyne = camera_image_from_velodyne_dict.get('cam0')
        #check shape
        testarr = np.array([[609.695409, -721.421597, -1.25125855, -167.899086],\
                            [180.384202,  7.64479802, -719.651474, -101.233067],\
                            [.999945389,  .000124365378,  .0104513030, -.272132796],\
                            [0., 0., 0., 1.]])
        self.assertTrue(np.allclose(camera_image_from_velodyne, testarr))
        
