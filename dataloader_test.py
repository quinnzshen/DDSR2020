import pytest

from dataloader import KittiDataset
from utils import time_to_nano, bin_search


def test_time_to_nano():
    assert time_to_nano("2011-09-26 14:14:11.435280384") == int(5.1251E13 + 435280384)
    assert time_to_nano("2021-09-36 00:00:00.010000001") == 10000001


def test_bin_search():
    test_arr1 = [0, 5, 234, 346, 7645, 3987584395, 48735875834758374875387]
    assert bin_search(test_arr1, 234, 4) == 2
    assert bin_search(test_arr1, 3, 4) == 0

    test_arr2 = [0, 3]
    assert bin_search(test_arr2, 500, 0) == 1


class TestKittiDataset:
    def test_len(self):
        test1 = KittiDataset("data/kitti_example")
        assert len(test1) == 22

    def test_get_item(self):
        test1 = KittiDataset("data/kitti_example")
        assert test1[0]["stereo_left_capture_time_nsec"] == time_to_nano("2011-09-26 14:14:10.911916288")
        assert test1[5]["stereo_right_capture_time_nsec"] == time_to_nano("2011-09-26 14:14:11.428480512")
        assert test1[21]["lidar_point_sensor"].shape[1] == 3

        with pytest.raises(IndexError):
            test1[-3]
            test1[300999933]
