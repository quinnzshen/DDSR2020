from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os
import PIL.Image as pil

from overlay_lidar_utils import project_points_on_image, filter_to_plane, filter_to_fov
from kitti_utils import load_lidar_points, compute_image_from_velodyne_matrices

def export_gt_depths_eigen_benchmark(split_path, gt_depth_dir, output_dir, use_eigen):

    split_folder = split_path

    print("Exporting ground truth depths for {}".format(split_path))
 
    with open(split_folder, 'r') as f:
        lines = f.read().splitlines()
    
    gt_depths = []
    
    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        if use_eigen:
            calib_dir = gt_depth_dir
            velo_filename = os.path.join(calib_dir, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            velo = load_lidar_points(velo_filename)
            cam_from_velo = compute_image_from_velodyne_matrices(os.path.join(gt_depth_dir, os.path.split(os.path.split(folder)[0])[0]))["stereo_left"]
            reproj = filter_to_fov(filter_to_plane(project_points_on_image(velo, cam_from_velo)), (375, 1242, 3))
            gt_depth = np.zeros((375, 1242))
            gt_depth[reproj[:,1], reproj[:,0]] = reproj[:,2]
            print(gt_depth.shape)
        else:
            folder = "/".join(folder.strip("/").split('/')[1:])
            gt_depth_path = os.path.join(gt_depth_dir, folder, "proj_depth",
                                             "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        gt_depths.append(gt_depth.astype(np.float32))
    
    if use_eigen:
        output_path = os.path.join(output_dir, "gt_eigen_lidar.npz")
    else:
        output_path = os.path.join(output_dir, "gt_depths.npz")

    print("Saving to {}".format(output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ddsr options")
    parser.add_argument("--split_path",
                             type = str,
                             help = "path to split file",
                             default = "splits/eigen_zhou/test.txt")
    parser.add_argument("--gt_depth_dir",
                             type = str,
                             help = "path to directory containing all ground truth depth maps",
                             default = "data/kitti_gt/depth_maps")
    parser.add_argument("--output_dir",
                             type = str,
                             help = "path to directory containing exported ground truth depth maps",
                             default = "data/kitti_gt")
    parser.add_argument("--use_eigen",
                             type = bool,
                             help = "path to directory containing exported ground truth depth maps",
                             default = False)
    opt = parser.parse_args()
    export_gt_depths_eigen_benchmark(opt.split_path, opt.gt_depth_dir, opt.output_dir, opt.use_eigen)