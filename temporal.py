import numpy as np

import os


def iso_string_to_nanoseconds(time_string):
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


def get_relative_pose(scene_path, target, source):
    # target coord to source coord
    with open(os.path.join(scene_path, f"oxts/data/{target:010}.txt")) as ft:
        # start index 8
        # end index 10
        datat = ft.readline().split()
        with open(os.path.join(scene_path, f"oxts/data/{source:010}.txt")) as fs:
            datas = fs.readline().split()
            rot = np.array(datas[3:6]) - np.array(datat[3:6])
            velo = (np.array(datat[8:11]) + np.array(datas[8:11])) / 2
    with open(os.path.join(scene_path, "oxts/timestamps.txt")) as time:
        i = 0
        target_time = 0
        source_time = 0
        for line in time:
            if i == target:
                target_time = iso_string_to_nanoseconds(line)
                if source_time:
                    break
            elif i == source:
                source_time = iso_string_to_nanoseconds(line)
                if target_time:
                    break
        delta_time = source_time - target_time

    pos = velo * delta_time / 10E9
    return calc_transformation_matrix(rot, pos)


def calc_transformation_matrix(rotation, translation):
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
    ], dtype=np.float64)


if __name__ == "__main__":
    path = r"data\kitti_example\2011_09_26\2011_09_26_drive_0048_sync"

    rot = np.deg2rad([45, 0, 0])
    print(rot)
    t_mat = calc_transformation_matrix(rot, [0, 3, 0])
    print(t_mat)
    inarr = np.array([
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1]
    ])
    inarr = np.transpose(inarr)
    outarr = t_mat @ inarr
    print(outarr)
    # print(np.array([0, 1, 0, 1]) @ np.transpose(t_mat))
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly_utils

    fig = go.Figure(layout=go.Layout(
        scene=dict(camera=dict(eye=dict(x=1.14, y=1.14, z=1.14)),  # the default values are 1.25, 1.25, 1.25
                   xaxis=dict(),
                   yaxis=dict(),
                   zaxis=dict(),
                   aspectmode="cube",  # this string can be 'data', 'cube', 'auto', 'manual'
                   # a custom aspectratio is defined as follows:
                   aspectratio=dict(x=1, y=1, z=1)
                   )    ))
    fig.add_trace(go.Scatter3d(x=inarr[0],
                               y=inarr[1],
                               z=inarr[2],
                               mode='markers',
                               marker=dict(size=3, color=1, colorscale='Viridis'),
                               name='lidar')
                  )
    fig.add_trace(go.Scatter3d(x=outarr[0],
                               y=outarr[1],
                               z=outarr[2],
                               mode='markers',
                               marker=dict(size=3, color=1, colorscale='Viridis'),
                               name='lidar')
                  )

    # fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
    #                     shared_xaxes=True, shared_yaxes=True)
    # fig.add_trace(go.Scatter3d(x=inarr[0],
    #                            y=inarr[1],
    #                            z=inarr[2],
    #                            mode='markers',
    #                            marker=dict(size=1, color=1, colorscale='Viridis'),
    #                            name='lidar'), row=1, col=1
    #               )
    # fig.add_trace(go.Scatter3d(x=outarr[0],
    #                     y=outarr[1],
    #                     z=outarr[2],
    #                     mode='markers',
    #                     marker=dict(size=1, color=1, colorscale='Viridis'),
    #                     name='lidar'), row=1, col=2
    #               )




    # plotly_utils.setup_layout(fig)
    fig.show()

    print("hi")


# def calc_transformation_mat(sample_path, idx):
#     """
#     Given the file path to the scene and the frame number within that scene, returns a 4x4 NumPy array containing the
#     translation matrix to convert the LiDAR point coordinates (relative to the sensor) into global coordinates
#     (relative to the starting point), where +x is East, +y is North, and +z is up.
#     :param sample_path: A file_path to a scene within the dataset
#     :param idx: The frame number within the scene
#     :return: 4x4 homogenous translation matrix to convert relative coordinates into continuous coordinates
#     """
#     with open(os.path.join(sample_path, "oxts/data/") + f"{0:010}.txt") as f:
#         line = f.readline().split()
#         orig_coords = np.array(line[:3], dtype=np.float64)
#         if idx == 0:
#             new_coords = np.array(line[:6], dtype=np.float64)
#         else:
#             with open(os.path.join(sample_path, "oxts/data/") + f"{idx:010}.txt") as fi:
#                 new_coords = np.array(fi.readline().split(), dtype=np.float64)
#
#     latlon_orig = np.deg2rad(orig_coords[:2])
#     latlon_new = np.deg2rad(new_coords[:2])
#     sin_rpy = np.sin(new_coords[3:])
#     cos_rpy = np.cos(new_coords[3:])
#
#     # translation matrix
#     return np.array([
#         [
#             cos_rpy[2] * cos_rpy[1],
#             cos_rpy[2] * sin_rpy[1] * sin_rpy[0] - sin_rpy[2] * cos_rpy[0],
#             cos_rpy[2] * sin_rpy[1] * cos_rpy[0] + sin_rpy[2] * sin_rpy[0],
#             calc_lon_dist(latlon_orig[0], latlon_new[0], latlon_orig[1], latlon_new[1])
#         ],
#         [
#             sin_rpy[2] * cos_rpy[1],
#             sin_rpy[2] * sin_rpy[1] * sin_rpy[0] + cos_rpy[2] * cos_rpy[0],
#             sin_rpy[2] * sin_rpy[1] * sin_rpy[0] - cos_rpy[2] * sin_rpy[0],
#             EARTH_RADIUS * (latlon_new[0] - latlon_orig[0])
#         ],
#         [
#             -1 * sin_rpy[1],
#             cos_rpy[1] * sin_rpy[0],
#             cos_rpy[1] * cos_rpy[0],
#             new_coords[2] - orig_coords[2]
#         ],
#         [0, 0, 0, 1],
#     ], dtype=np.float64)