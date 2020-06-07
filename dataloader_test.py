import numpy as np
import plotly.graph_objects as go
import plotly_utils

from dataloader import KittiDataset

# Some testing you can ignore this I guess
if __name__ == "__main__":
    def add_1_column(arr):
        shape = arr.shape
        new_arr = np.ones((shape[0], shape[1] + 1))
        new_arr[:, :-1] = arr
        return new_arr

    dataset = KittiDataset('data/kitti_example')
    # d = dataset[5]
    # print(dataset[0]["lidar_start_capture_timestamp_nsec"] / 1000000000)
    # print(d["lidar_start_capture_timestamp_nsec"] / 1000000000)
    # print(d["transformation"])
    # print(d["lidar_point_sensor"])
    # print(glob('data/kitti_example/2011_09_26/*/velodyne_points/'))



    fig = go.Figure()
    # data = go.Scatter3d(x=lidar_data[:, 0],
    #                    y=lidar_data[:, 1],
    #                    z=lidar_data[:, 2],
    #                    mode='markers',
    #                    marker=dict(size=1, color=lidar_data[:, 3], colorscale='Viridis'),
    #                    name='lidar')

    orig_frame = dataset[0]

    velo_points = np.matmul(add_1_column(orig_frame["lidar_point_sensor"]), orig_frame["transformation"].transpose())
    colors = orig_frame["lidar_point_reflectivity"]

    for i in range(1, 3):
        frame = dataset[i]
        print(frame["lidar_point_sensor"].shape)
        velo_points = np.concatenate(
            (velo_points, np.matmul(add_1_column(frame["lidar_point_sensor"]), frame["transformation"].transpose())))
        colors = np.concatenate((colors, frame["lidar_point_reflectivity"]))

    data = go.Scatter3d(x=velo_points[:, 0], y=velo_points[:, 1], z=velo_points[:, 2],
                        mode="markers", marker=dict(size=1, color=colors, colorscale="Viridis"), name="lidar")

    fig.add_traces(data)

    plotly_utils.setup_layout(fig)
    fig.show()