import numpy as np

import os


VELO_INDICES = np.array([7, 6, 10])



def string_to_nano(time_string):
    """
    Converts a line in the format provided by timestamps.txt to the number of nanoseconds since the midnight of that day
    :param time_string: The string to be converted into nanoseconds
    :return: The number of nanoseconds since midnight
    """
    total = 0
    total += int(time_string[11:13]) * 3600 * 1000000000
    total += int(time_string[14:16]) * 60 * 1000000000
    total += int(time_string[17:19]) * 1000000000
    total += int(time_string[20:])
    return total

def calc_transformation_matrix(rotation, translation):
    """
    Calculates the homogeneous transformation matrix given relative rotation and translation
    :param [np.ndarray] rotation: Shape of [3] containing the relative roll, pitch, and yaw (in radians)
    :param [np.ndarray] translation: Shape of [3] containing the relative XYZ displacement
    :return [np.ndarray]: 4x4 matrix that transforms given the relative rotation and translation
    """
    sin_rot = np.sin(rotation)
    cos_rot = np.cos(rotation)
    return np.array([
        [
            cos_rot[2] * cos_rot[1],
            cos_rot[2] * sin_rot[1] * sin_rot[0] - sin_rot[2] * cos_rot[0],
            cos_rot[2] * sin_rot[1] * cos_rot[0] + sin_rot[2] * sin_rot[0],
            translation[0]
        ],
        [
            sin_rot[2] * cos_rot[1],
            sin_rot[2] * sin_rot[1] * sin_rot[0] + cos_rot[2] * cos_rot[0],
            sin_rot[2] * sin_rot[1] * sin_rot[0] - cos_rot[2] * sin_rot[0],
            translation[1]
        ],
        [
            -1 * sin_rot[1],
            cos_rot[1] * sin_rot[0],
            cos_rot[1] * cos_rot[0],
            translation[2]
        ],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def get_relative_pose(scene_path, target, source):
    """
    Computes relative pose matrix [4x4] between the 2 given frames in a scene.
    By multiplying, transforms target coordinates into source coordinates.
    :param [str] scene_path: Path name to the scene folder
    :param [int] target : The target frame number
    :param [int] source: The source frame number
    :return [np.ndarray]: Shape of (4, 4) containing the values to transform between target frame to source frame
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
            yaw_rot_mat = np.array(
                [[np.cos(-datat[5]), -np.sin(-datat[5])], [np.sin(-datat[5]), np.cos(-datat[5])]])
            velo[:2] = velo[:2] @ yaw_rot_mat.T

    # Determines the relative time passed between the 2 frames, as target - source
    with open(os.path.join(scene_path, "oxts/timestamps.txt")) as time:
        i = 0
        target_time = 0
        source_time = 0
        for line in time:
            if i == target:
                target_time = string_to_nano(line)
                if source_time:
                    break
            elif i == source:
                source_time = string_to_nano(line)
                if target_time:
                    break
            i += 1
        delta_time_nsec = target_time - source_time

    # Determines displacement by multiplying velocity by time
    pos = velo * delta_time_nsec / 1E9

    newrot = np.array([-rot[1], -rot[2], rot[0]])
    newpos = np.array([-pos[1], -pos[2], pos[0]])
    # return calc_transformation_matrix(rot, pos)
    return calc_transformation_matrix(newrot, newpos)

