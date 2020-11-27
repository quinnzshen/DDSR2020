from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os
import PIL.Image as pil

from kitti_utils import load_lidar_points, generate_depth_map


def export_gt_depths(split_path, gt_depth_dir, output_dir, use_lidar):
    split_folder = split_path

    print("Exporting ground truth depths for {}".format(split_path))

    with open(split_folder, 'r') as f:
        lines = f.read().splitlines()

    gt_depths = []

    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        if use_lidar:
            calib_dir = gt_depth_dir
            velo_filename = os.path.join(calib_dir, folder,
                                         "velodyne_points/data", f"{frame_id:010}.bin")
            velo = load_lidar_points(velo_filename)
            gt_depth = generate_depth_map(os.path.join(gt_depth_dir, os.path.split(folder)[0]), velo, 2)

        else:
            folder = "/".join(folder.strip("/").split('/')[1:])
            gt_depth_path = os.path.join(gt_depth_dir, folder, "proj_depth", "groundtruth", "image_02",
                                         f"{frame_id:010}.png")
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        gt_depths.append(gt_depth.astype(np.float32))

    if use_lidar:
        output_path = os.path.join(output_dir, "gt_lidar.npz")
    else:
        output_path = os.path.join(output_dir, "gt_depthmaps.npz")

    print("Saving to {}".format(output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ddsr options")
    parser.add_argument("--split_path",
                        type=str,
                        help="path to split file",
                        default="splits/eigen_zhou/test.txt")
    parser.add_argument("--gt_depth_dir",
                        type=str,
                        help="path to directory containing all ground truth depth maps",
                        default="data/kitti_gt/depth_maps")
    parser.add_argument("--output_dir",
                        type=str,
                        help="path to directory containing exported ground truth depth maps",
                        default="data/kitti_gt")
    parser.add_argument("--use_lidar",
                        action='store_true',
                        help="Activating his flag uses lidar instead of gt kitti depth maps")
    opt = parser.parse_args()
    export_gt_depths(opt.split_path, opt.gt_depth_dir, opt.output_dir, opt.use_lidar)
