import compute_photometric_error_utils as cpeu
import pytest
import numpy as np
import torch


def test_compute_relative_pose_matrix():
    sample_rotation = torch.eye(3)
    sample_translation = torch.zeros(3)
    rel_pose = cpeu.compute_relative_pose_matrix(sample_translation, sample_rotation)
    torch.testing.assert_allclose(torch.eye(4), rel_pose)


def test_reproject_source_to_target():
    sample_tgt_intrinsic = sample_src_intrinsic = np.eye(3)
    sample_lidar = np.array([[10., 10., 10.]])
    sample_rel_pose = np.eye(4)
    proj_coords = cpeu.reproject_source_to_target(sample_tgt_intrinsic, sample_src_intrinsic, sample_lidar,
                                                  sample_rel_pose)
    assert np.allclose(proj_coords[0], np.array([[10, 10, 1]]))
    assert np.allclose(proj_coords[1], np.array([[10, 10]]))


def test_calc_transformation_matrix():
    rot1 = np.deg2rad([0, 0, 0])
    dis1 = np.array([0, 0, 1])
    np.testing.assert_allclose(cpeu.calc_transformation_matrix(rot1, dis1), np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]))

    rot2 = np.deg2rad([45, 0, 90])
    dis2 = np.array([0, 2, 0])
    np.testing.assert_allclose(cpeu.calc_transformation_matrix(rot2, dis2), np.array([
        [6.123234e-17, -7.071068e-01, 7.071068e-01, 0],
        [1, 4.329780e-17, -4.329780e-17, 2.000000e+00],
        [0, 7.071068e-01, 7.071068e-01, 0],
        [0, 0, 0, 1]
    ]))


def test_calc_photo_error():
    color_points = np.array([
        [0, 0, 3, 1, 255, 255, 255],
        [0, 1, 2, 1, 0, 0, 255]
    ])
    img = np.array([
        [[0, 0, 0], [3, 5, 100]],
        [[0, 0, 0], [255, 32, 200]]
    ])

    np.testing.assert_allclose(cpeu.calc_photo_error(img, color_points), np.array([
        [np.sqrt(3) * 255, 0],
        [255, 0]
    ]))


def test_calc_photo_error_velo():
    color_points = np.array([
        [0, 0, 3, 1, 255, 255, 255],
        [0, 1, 2, 1, 0, 0, 255]
    ])
    img = np.array([
        [[0, 0, 0], [3, 5, 100]],
        [[0, 0, 0], [255, 32, 200]]
    ])
    np.testing.assert_allclose(cpeu.calc_photo_error_velo(img, color_points), np.array([
        np.sqrt(3) * 255, 255
    ]))
