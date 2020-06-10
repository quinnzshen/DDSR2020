import kitti_utils as ku
import os

def load_velodyne_points_test():
    """
    Tests the return of the load_velodyne_points funtion in kitti_utils.py
    """
    lidar_point_coord_velodyne = ku.load_velodyne_points('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/0000000010.bin')
    assert lidar_point_coord_velodyne.shape == (116006, 3)
    assert int(lidar_point_coord_velodyne[0][0]) == 73
    assert int(lidar_point_coord_velodyne[0][1]) == 7

def read_calib_file_test():
    """
    Tests the return of the read_calib_file funtion in kitti_utils.py
    """
    calib_dir = 'data/kitti_example/2011_09_26'
    cam2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    imu2cam = ku.read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
    assert len(cam2cam) == 34
    assert len(velo2cam) == 5
    assert len(imu2cam) == 3

def compute_image_from_velodyne_matrix_test():
    """
    Tests the return of the compute_image_from_velodyne_matrix funtion in kitti_utils.py
    """
    calib_dir = 'data/kitti_example/2011_09_26'
    camera_image_from_velodyne = ku.compute_image_from_velodyne_matrix(calib_dir, cam=0)
    assert camera_image_from_velodyne.shape == (4,4)
    assert int(camera_image_from_velodyne[0][0]) == 609
    assert int(camera_image_from_velodyne[0][1]) == -721

    