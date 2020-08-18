import layers
import kitti_utils as ku
from inference import run_inference
import torch
import os
import datasets


class Metrics:
    def __init__(self, options):
        self.opt = options
        self.model_path = self.opt.load_weights_folder
        split_path = os.path.join('splits', self.opt.eval_split, 'test_files.txt')
        filenames = open(split_path).readlines()
        dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext='.jpg')
        dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=self.opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        self.predicted_disps = []
        self.lidar_depth_maps = []
        idx=0
        print('-> Loading predicted depth maps and lidar depth maps')
        for data in dataloader:
            file_info = filenames[idx].split()
            input_color = data[("color", 0, 0)].cuda()
            lidar_depth_map = ku.generate_depth_map(os.path.join('kitti_data', file_info[0][:10]), os.path.join('kitti_data', file_info[0], f'velodyne_points/data/{int(file_info[1]):010}.bin'))
            self.lidar_depth_maps.append(lidar_depth_map)
            with torch.no_grad():
                predicted_disp = run_inference(self.model_path, input_color, use_fpn = self.opt.use_fpn)
            self.predicted_disps.append(torch.nn.functional.interpolate(predicted_disp, (375, 1242), mode="bilinear", align_corners=False))
            idx+=1

    def compute_l1_error_on_split(self):
        """
        This function computes the average L1 error over a given split of images.
        :return [float]: average L1 error across all images in a given split.
        """
        total_error = 0
        print(f'-> Computing avg L1 error over {len(self.predicted_disps)} images\n')
        for idx in range(len(self.predicted_disps)):
            avg_error = self.compute_l1_error_single_frame(self.predicted_disps[idx], self.lidar_depth_maps[idx])
            total_error+=avg_error
        return total_error/len(self.predicted_disps)

    def compute_l1_error_single_frame(self, predicted_disp_map, lidar_depth_map):
        """
        This function computes the L1 error (average difference between lidar depth and predicted depth) for a single image.
        :param [torch.Tensor] predicted_disp_map: [1, 1, H, W] predicted disparity map for image.
        :param [numpy.array] lidar_depth_map: [H, W] ground truth lidar points projected into the image frame.
        :return [float] avg_error: average L1 error across the entire image.
        """
        predicted_depth_map = layers.disp_to_depth(predicted_disp_map, .1, 100)[1]
        predicted_depth_map = predicted_depth_map.squeeze(0).squeeze(0) * 31.257
        
        total_diff=0
        counter=0
        for h in range(lidar_depth_map.shape[0]):
            for w in range(lidar_depth_map.shape[1]):
                if(lidar_depth_map[h,w] != 0):
                    total_diff+=abs(lidar_depth_map[h,w] - predicted_depth_map[h,w])
                    counter+=1
        avg_error = total_diff/counter
        return avg_error
