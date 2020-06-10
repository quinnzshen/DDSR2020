import pytest
import overlay_lidar_utils as olu
import kitti_utils as ku

def generate_lidar_point_coord_camera_image_test():
    """
    Tests the return of the generate_lidar_point_coord_camera_image_test funtion in overlay_lidar_utils.py
    """
    #load velodyne points
    lidar_point_coord_velodyne = ku.load_velodyne_points('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000010.bin')
    calib_dir = 'data/kitti_example/2011_09_26'
    camera_image_from_velodyne = ku.compute_image_from_velodyne_matrix(calib_dir, cam=0)
    lidar_point_coord_camera_image = olu.generate_lidar_point_coord_camera_image(lidar_point_coord_velodyne, camera_image_from_velodyne, 1242, 375)
    assert lidar_point_coord_camera_image.shape == (19887, 4)
    assert lidar_point_coord_camera_image[0][0] == 539
    assert lidar_point_coord_camera_image[0][1] == 153
    