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


def test_project_points_on_image():
    sample1 = np.array([
        [0, 3, 5, 1, 2, 3],
        [-3, 4, 3, 1, 4, 3]
    ])
    mat1 = np.eye(4)
    np.testing.assert_allclose(cpeu.project_points_on_image(sample1, mat1), sample1)
    mat2 = np.array([
        [3, 2, 1, 0],
        [0, 0, 0, 1],
        [-3, 5, 0, 0],
        [1, 1, 1, 2]
    ])
    np.testing.assert_allclose(cpeu.project_points_on_image(sample1, mat2), np.array([
        [11, 1, 15, 10, 2, 3],
        [2, 1, 29, 6, 4, 3]
    ]))


def test_get_associated_colors():
    img1 = np.array([
        [[3, 2, 8], [5, 3, 9]],
        [[2, 1, 7], [1, 1, 4]],
    ])
    sample_points1 = np.array([[0, 1, 5, 1, 2]])
    np.testing.assert_allclose(cpeu.get_associated_colors(sample_points1, img1), np.array([
        [2, 1, 7, 2]
    ]))


def test_color_image():
    points = np.array([
        [1, 0, 3, 1, 2, 3, 4]
    ])
    shape = (2, 2, 3)
    np.testing.assert_allclose(cpeu.color_image(points, shape), np.array([
        [[255, 255, 255], [2, 3, 4]],
        [[255, 255, 255], [255, 255, 255]]
    ]))


def test_filter_to_plane():
    points = np.array([
        [55, 10, 5, 1, 3],
        [2, 3, -1, 1, 5],
        [1, 2, 1, 1, 2]
    ])
    np.testing.assert_allclose(cpeu.filter_to_plane(points), np.array([
        [11, 2, 5, 1, 3],
        [1, 2, 1, 1, 2]
    ]))


def test_filter_to_fov():
    points = np.array([
        [300, 2, 3, 1, 3],
        [2, 2, 4, 1, 2],
        [-1, -2, 60, 1, 6]
    ])
    shape = (301, 300, 3)
    np.testing.assert_allclose(cpeu.filter_to_fov(points, shape), np.array([
        [2, 2, 4, 1, 2]
    ]))


def test_color_target_points_with_source():
    pose = np.eye(4)
    coord2img = np.eye(4)
    points = np.array([
        [0, 0, 1, 1],
        [0, 0, -3, 1],
        [2, 2, 2, 1],
        [500, 300, 5, 1]
    ])
    img = np.array([
        [[3, 3, 4], [2, 7, 5]],
        [[1, 2, 5], [8, 6, 4]]
    ])
    color_out, tgt_points = cpeu.color_target_points_with_source(points, img, coord2img, pose)
    np.testing.assert_allclose(color_out, np.array([
        [0, 0, 1, 1, 3, 3, 4],
        [1, 1, 2, 1, 8, 6, 4]
    ]))
    np.testing.assert_allclose(tgt_points, np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 1],
        [100, 60, 5, 1]
    ]))


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
