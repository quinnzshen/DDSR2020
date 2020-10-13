from __future__ import absolute_import, division, print_function

import numpy as np
from PIL import Image
import torch
from collections import Counter

import os
import pandas as pd
from enum import Enum

from compute_photometric_error_utils import calc_transformation_matrix


class KITTICameraNames(str, Enum):
    stereo_left = "stereo_left"
    stereo_right = "stereo_right"


CAMERA_NAME_TO_PATH_MAPPING = {
    KITTICameraNames.stereo_left: "image_02",
    KITTICameraNames.stereo_right: "image_03"
}

KITTI_TIMESTAMPS = ["/timestamps.txt", "velodyne_points/timestamps_start.txt", "velodyne_points/timestamps_end.txt"]
EPOCH = np.datetime64("1970-01-01")
VELO_INDICES = np.array([7, 6, 10])

def load_lidar_points(filename):
    """
    This function loads 3D point cloud from KITTI file format.
    :param [string] filename: File path for 3d point cloud data.
    :return [np.array]: [N, 4] N lidar points represented as (X, Y, Z, reflectivity) points.
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calibration_file(path):
    """
    This function reads the KITTI calibration file.
    (from https://github.com/hunse/kitti)
    :param [string] path: File path for KITTI calbration file.
    :return [dictionary] data: Dictionary containing the camera intrinsic and extrinsic matrices.
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # Try to cast to float array.
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # Casting error: data[key] already eq. value, so pass.
                    pass

    return data


def compute_image_from_velodyne_matrices(calibration_dir):
    """
    This function computes the translation matrix to project 3D lidar points into the 2D image plane.
    :param [String] calibration_dir: Directory to folder containing camera/lidar calibration files
    :return:  dictionary of numpy.arrays of shape [4, 4] that converts 3D lidar points to 2D image plane for each camera
    (keys: stereo_left, stereo_right)
    """
    # Based on code from monodepth2 repo.

    # Load cam_to_cam calib file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    # Load velo_to_cam file.
    velo2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))

    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3, 1)))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    camera_image_from_velodyne_dict = {}

    for camera_name in KITTICameraNames:
        # Get camera number by slicing last 2 characters off of camera_name string.
        camera_path = CAMERA_NAME_TO_PATH_MAPPING[camera_name]

        cam_num = camera_path[-2:]
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam[f"R_rect_{cam_num}"].reshape(3, 3)
        P_rect = cam2cam[f"P_rect_{cam_num}"].reshape(3, 4)
        camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
        camera_image_from_velodyne = np.vstack((camera_image_from_velodyne, np.array([[0, 0, 0, 1.0]])))
        camera_image_from_velodyne_dict.update({KITTICameraNames(camera_name).name: camera_image_from_velodyne})

    return camera_image_from_velodyne_dict

def generate_depth_map(calibration_dir, velo, cam):
    """
    Generate a depth map from velodyne data
    Adapted from Monodepth2
    """
    # Based on code from monodepth2 repo.

    # Load cam_to_cam calib file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    
    # Load velo_to_cam file.
    velo2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    camera_image_from_velodyne = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    
    velo = velo[velo[:, 0] >= 0, :]
    
    velo_pts_im = np.dot(camera_image_from_velodyne, velo.T).T       
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    
    velo_pts_im[:, 2] = velo[:, 0]
    
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]
    
    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    
    return depth
def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1
def iso_string_to_nanoseconds(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since EPOCH
    :param [str] time_string: The string to be converted into nanoseconds
    :return [np.int64]: The number of nanoseconds since epoch (Jan 1st 1970 UTC)
    """
    return (np.datetime64(time_string) - EPOCH).astype(np.int64)


def get_timestamp_nsec(sample_path, idx):
    """
    Given the file path to the scene and the frame number within that scene, returns an integer containing the
    time (nanoseconds) retrieved from the idx'th line in the path given.
    :param [str] sample_path: A file_path to a timestamps file
    :param [int] idx: The frame number within the scene
    :return [np.int64]: Integer containing the nanosecond taken of the given frame
    """
    with open(sample_path) as f:
        count = 0
        for line in f:
            if count == idx:
                return iso_string_to_nanoseconds(line)
            count += 1


def get_nearby_frames_data(path_name, idx, previous_frames, next_frames, is_jpeg):
    """
    Given a specific index, return a dictionary containing information about the frame n frames before and after the target index
    in the dataset.
    :param dataset_index [pd.DataFrame]: The dataframe containing the paths and indices of the data
    :param [int] previous_frames: Number of frames before the target frame that will be retrieved.
    :param [int] next_frames: Number of frames after the target frame that will be retrieved.
    :param [bool] is_jpeg: Whether or not the read images are jpegs
    :return [dict]: Dictionary containing camera data and pose of nearby frames, the key is the relative index and the value is the data (e.g. -1 would be the previous image, 2 would be the next-next image).
    """
    nearby_frames = {}
    for relative_idx in range(-previous_frames, next_frames + 1):
        # We do not want to include the current frame in the nearby frames data.
        if relative_idx == 0:
            continue
        try:
            nearby_frames[relative_idx] = {'camera_data': get_camera_data(path_name, idx + relative_idx, is_jpeg),
                                           'pose': get_relative_pose_between_consecutive_frames(path_name, idx, idx+relative_idx)}
        except FileNotFoundError:
            nearby_frames[relative_idx] = {"camera_data": {},
                                           "pose": {}}
    return nearby_frames


def get_camera_data(path_name, idx, is_jpeg=True):
    """
    Gets the basic camera information given the path name to the scene and the frame number within
    that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :param [bool] is_jpeg: Whether or not the read images are jpegs
    :return [dict]: A dictionary containing camera data. If the camera data cannot be found, return an empty dictionary.
    """
    camera_data = dict()

    for camera_name in KITTICameraNames:
        camera_path = CAMERA_NAME_TO_PATH_MAPPING[camera_name]
        # Check if required paths exist.
        # The f-string is following the format of KITTI, padding the frame number with 10 zeros.
        if is_jpeg:
            camera_image_path = os.path.join(path_name, f"{camera_path}/data/{idx:010}.jpg")
        else:
            camera_image_path = os.path.join(path_name, f"{camera_path}/data/{idx:010}.png")

        timestamp_path = os.path.join(path_name, f"{camera_path}/timestamps.txt")
        camera_image = torch.from_numpy(np.array(Image.open(camera_image_path))).float() / 255.0
        timestamp = get_timestamp_nsec(timestamp_path, idx)
        camera_data[f"{camera_name}_image"] = camera_image
        camera_data[f"{camera_name}_shape"] = camera_image.shape
        camera_data[f"{camera_name}_capture_time_nsec"] = timestamp

    return camera_data


def get_lidar_data(path_name, idx):
    """
    Gets the basic LiDAR information given the path name to the scene and the frame number within that scene.
    :param [str] path_name: A file path to a scene within the dataset
    :param [int] idx: The frame number in the scene
    :return [dict]: A dictionary containing the points, reflectivity, start, and end times of the LiDAR scan.
    """
    lidar_points = torch.from_numpy(load_lidar_points(os.path.join(path_name, f"velodyne_points/data/{idx:010}.bin")))
    start_time = get_timestamp_nsec(os.path.join(path_name, "velodyne_points/timestamps_start.txt"), idx)
    end_time = get_timestamp_nsec(os.path.join(path_name, "velodyne_points/timestamps_end.txt"), idx)
    return {
        "lidar_point_coord_velodyne": lidar_points[:, :3],
        "lidar_point_reflectivity": lidar_points[:, 3],
        "lidar_start_capture_time_nsec": start_time,
        "lidar_end_capture_time_nsec": end_time
    }


def get_imu_data(scene_path, idx):
    """
    Get Intertial Measurement Unit (IMU) data. 
    :param [string] scene_path: A file path to a scene within the KITTI dataset.
    :param [int] idx: The frame number in the scene. 
    :return [dict]: Return a dictionary of imu data (key: field name, value: field value).
    """
    imu_data_path = os.path.join(scene_path, f"oxts/data/{idx:010}.txt")
    imu_format_path = os.path.join(scene_path, "oxts/dataformat.txt")

    with open(imu_format_path) as f:
        # The data is formatted as "name: description". We only care about the name here.
        imu_keys = [line.split(':')[0] for line in f.readlines()]

    with open(imu_data_path) as f:
        imu_values = f.read().split()

    return dict(zip(imu_keys, imu_values))


def get_imu_dataframe(scene_path):
    """
    Get Intertial Measurement Unit (IMU) data for an entire scene.
    :param [string] scene_path: A file path to a scene within the KITTI dataset.
    :return [pd.DataFrame]: A dataframe with the entire scenes IMU data.
    """
    num_frames = len(os.listdir(os.path.join(scene_path, 'oxts/data')))

    imu_values = []
    for idx in range(num_frames):
        imu_data = get_imu_data(scene_path, idx)
        imu_values.append(list(imu_data.values()))

    return pd.DataFrame(imu_values, columns=list(imu_data.keys()))


def get_camera_intrinsic_dict(calibration_dir):
    """
    This function gets the intrinsic matrix for each camera from the KITTI calibration file
    :param: [string] calibration_dir: directory where the KITTI calbration files are located
    :return: dictionary of length 2 containing the 3x3 intrinsic matrix for each camera (keys: stereo_left, stereo_right)
    """
    # Load cam_to_cam calib file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))

    camera_intrinsic_dict = {}
    for camera_name in KITTICameraNames:
        camera_path = CAMERA_NAME_TO_PATH_MAPPING[camera_name]

        # Get camera number by slicing last 2 characters off of camera_name string.
        cam_num = camera_path[-2:]
        intrinsic_matrix = torch.from_numpy(cam2cam[f"P_rect_{cam_num}"].reshape(3, 4)[:,:3]).float()
        camera_intrinsic_dict.update({KITTICameraNames(camera_name).name: intrinsic_matrix})
    return camera_intrinsic_dict


def get_relative_rotation_stereo(calibration_dir):
    """
    This function computes the relative rotation matrix between stereo cameras for KITTI.
    :param: [string] calibration dir: directory where KITTI calibration files are located.
    :return: torch.Tensor of shape [3, 3], matrix representing the relative rotation between the camera that captured
    the source image and the camera that captured the target image.
    """
    # Read calibration file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    # Compute relative rotation matrix.
    rotation_target = cam2cam['R_02'].reshape(3, 3)
    rotation_source = cam2cam['R_03'].reshape(3, 3)
    rotation_source_to_target = np.linalg.inv(rotation_source) @ rotation_target
    return torch.from_numpy(rotation_source_to_target).float()


def get_relative_translation_stereo(calibration_dir):
    """
    This function computes the relative translation vector between stereo cameras for KITTI.
    :param: [string] calibration dir: directory where KITTI calibration files are located.
    :return: torch.Tensor of shape [3, 1], vector representing the relative translation between the camera that captured
    the source image and the camera that captured the target image.
    """
    # Read calibration file.
    cam2cam = read_calibration_file(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    # Compute relative translation vector.
    translation_target = cam2cam['T_02']
    translation_source = cam2cam['T_03']
    translation_source_to_target = translation_source - translation_target
    return torch.from_numpy(translation_source_to_target).float()


def get_relative_pose_between_consecutive_frames(scene_path, target, source):
    """
    Computes relative pose matrix [4x4] between the 2 given frames in a scene (frames must be consecutive).
    By multiplying, transforms target coordinates into source coordinates.
    :param [str] scene_path: Path name to the scene folder
    :param [int] target : The target frame number
    :param [int] source: The source frame number
    :return [torch.tensor]: Shape of (4, 4) containing the values to transform between target frame to source frame
    """
    if target == source:
        return np.eye(4, dtype=np.float32)

    with open(os.path.join(scene_path, f"oxts/data/{target:010}.txt")) as ft:
        datat = np.array(ft.readline().split(), dtype=np.float)
        with open(os.path.join(scene_path, f"oxts/data/{source:010}.txt")) as fs:
            datas = np.array(fs.readline().split(), dtype=np.float)

            # Calculates relative rotation and velocity
            rot = np.array(datat[3:6], dtype=np.float) - np.array(datas[3:6], dtype=np.float)
            velo = (datat[VELO_INDICES] + datas[VELO_INDICES]) / 2
            yaw = -datat[5]
            yaw_rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            velo[:2] = velo[:2] @ yaw_rot_mat.T

    # Determines the relative time passed between the 2 frames, as target - source
    with open(os.path.join(scene_path, "oxts/timestamps.txt")) as time:
        i = target_time = source_time = 0
        for line in time:
            if i == target:
                target_time = iso_string_to_nanoseconds(line)
                if source_time:
                    break
            elif i == source:
                source_time = iso_string_to_nanoseconds(line)
                if target_time:
                    break
            i += 1
        delta_time_nsec = target_time - source_time

    # Determines displacement by multiplying velocity by time
    pos = velo * delta_time_nsec / 1E9
    # Convert trnasformation from IMU frame to camera frame.
    pos_cam = np.array([-pos[1], -pos[2], pos[0]])
    rot_cam = np.array([-rot[1], -rot[2], rot[0]])
    rel_pose = calc_transformation_matrix(rot_cam, pos_cam)
    return torch.from_numpy(rel_pose)


def get_pose(scene_path, frame):
    """
    This function gets the pose matrix with respect to the frame at index 0 for a given frame.
    :param [str] scene_path: Path name to the scene folder
    :param [int] frame: the index of the frame that pose is being calculated for.
    :return: tensor of shape [4, 4] containing the pose of the image at index frame with respect to the the image at index 0.
    """
    pose = get_relative_pose_between_consecutive_frames(scene_path, frame, 0)
    rel_translation = torch.zeros(3)
    for idx in range(0, frame - 1):
        rel_translation += get_relative_pose_between_consecutive_frames(scene_path, idx, idx + 1)[:3, 3]
    pose[:3, 3] = rel_translation

    return pose

def get_stereo_pose():
    stereo_T = np.eye(4, dtype=np.float32)
    baseline_sign = 1
    side_sign = -1
    stereo_T[0, 3] = side_sign * baseline_sign * 0.1
    return torch.from_numpy(stereo_T)