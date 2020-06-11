import pytest
import numpy as np

from dataloader import KittiDataset
from utils import iso_string_to_nanoseconds, bin_search


def test_time_to_nano():
    assert iso_string_to_nanoseconds("2011-09-26 14:14:11.435280384") == int(5.1251E13 + 435280384)
    assert iso_string_to_nanoseconds("2021-09-36 00:00:00.010000001") == 10000001


def test_bin_search():
    test_arr1 = [0, 5, 234, 346, 7645, 3987584395, 48735875834758374875387]
    assert bin_search(test_arr1, 234, 4) == 2
    assert bin_search(test_arr1, 3, 4) == 0

    test_arr2 = [0, 3]
    assert bin_search(test_arr2, 500, 0) == 1


class TestKittiDataset:
    def test_dataset_length(self):
        dataset = KittiDataset("data/kitti_example")
        assert len(dataset) == 22

    def test_get_item(self):
        dataset = KittiDataset("data/kitti_example")
        assert dataset[0]["stereo_left_capture_time_nsec"] == iso_string_to_nanoseconds("2011-09-26 14:14:10.911916288")
        assert dataset[5]["stereo_right_capture_time_nsec"] == iso_string_to_nanoseconds("2011-09-26 14:14:11.428480512")
        assert dataset[6]["stereo_left_image"].shape == (375, 1242, 3)
        assert dataset[21]["lidar_point_sensor"].shape[1] == 3
        assert dataset[2]["lidar_start_capture_timestamp_nsec"].dtype == np.int64
        assert dataset[3]["lidar_end_capture_timestamp_nsec"].dtype == np.int64
        assert dataset[15]["lidar_point_reflectivity"].dtype == np.float32

        with pytest.raises(IndexError):
            dataset[-3]
            dataset[300999933]