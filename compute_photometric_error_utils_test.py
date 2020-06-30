import compute_photometric_error_utils as cpeu
import pytest
import numpy as np


def test_compute_relative_pose_matrix():
    sample_rotation = np.eye(3)
    sample_translation = np.array([[0], [0], [0]])
    rel_pose = cpeu.compute_relative_pose_matrix(sample_translation, sample_rotation)
    assert np.allclose(np.eye(4), rel_pose)


def test_reproject_source_to_target():
    sample_tgt_intrinsic = sample_src_intrinsic = np.eye(3)
    sample_lidar = np.array([[10., 10., 10.]])
    sample_rel_pose = np.eye(4)
    proj_coords = cpeu.reproject_source_to_target(sample_tgt_intrinsic, sample_src_intrinsic, sample_lidar,
                                                  sample_rel_pose)
    assert np.allclose(proj_coords[0], np.array([[10, 10, 1]]))
    assert np.allclose(proj_coords[1], np.array([[10, 10]]))
