import pytest
import numpy as np

from dataloader import KittiDataset

TEST_CONFIG_PATH = "configs/kitti_dataset.yml"


@pytest.fixture
def dataset():
    return KittiDataset.init_from_config(TEST_CONFIG_PATH)


def test_dataset_length(dataset):
    assert len(dataset) == 17


def test_get_item(dataset):
    assert dataset[0]["stereo_left_capture_time_nsec"] == iso_string_to_nanoseconds("2011-09-26 14:14:10.911916288")
    assert dataset[5]["stereo_right_capture_time_nsec"] == iso_string_to_nanoseconds("2011-09-26 14:14:11.428480512")
    assert dataset[6]["stereo_left_image"].shape == (375, 1242, 3)
    assert dataset[16]["lidar_point_coord_velodyne"].shape[1] == 3
    assert dataset[2]["lidar_start_capture_time_nsec"].dtype == np.int64
    assert dataset[3]["lidar_end_capture_time_nsec"].dtype == np.int64
    assert dataset[15]["lidar_point_reflectivity"].dtype == np.float32


def test_out_of_bounds(dataset):
    with pytest.raises(IndexError):
        dataset[-3]
        dataset[300999933]
