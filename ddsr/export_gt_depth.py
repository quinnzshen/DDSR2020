# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil

def export_gt_depths_kitti(split_path, gt_depth_dir, output_dir):

    split_folder = split_path

    print("Exporting ground truth depths for {}".format(split_path))
 
    with open(split_folder, 'r') as f:
        lines = f.read().splitlines()
    
    gt_depths = []

    for line in lines:
        folder, frame_id, _ = line.split()
        folder = "/".join(folder.strip("/").split('/')[1:])
        
        frame_id = int(frame_id)

        gt_depth_path = os.path.join(gt_depth_dir, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
        gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(output_dir, "gt_depths.npz")

    print("Saving to {}".format(output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))

if __name__ == "__main__":
    export_gt_depths_kitti("splits/eigen_zhou/test.txt", "data/kitti_gt", "data/kitti_gt")