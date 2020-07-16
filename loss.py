import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from temporal import t_transform_n


class SSIM(nn.Module):
    def __init__(self):
        """
        Sets up the layers/pooling to run the forward method which actually does the computation
        """
        super(SSIM, self).__init__()
        self.padding_reflect = nn.ReflectionPad2d(1)

        self.mu_pred = nn.AvgPool2d(3, 1)
        self.mu_targ = nn.AvgPool2d(3, 1)
        self.sigma_pred = nn.AvgPool2d(3, 1)
        self.sigma_targ = nn.AvgPool2d(3, 1)
        self.sigma_pred_targ = nn.AvgPool2d(3, 1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, targ):
        """
        Computes the SSIM between the two given images by running them through layers
        :param [torch.tensor] pred: The predicted image, formatted as [batch_size, 3, H, W]
        :param [torch.tensor] targ: The target image, formatted as [batch_size, 3, H, W]
        :return [torch.tensor]: A tensor representing how similar the two images are, on a pixel basis in the format
        [batch_size, 3, H, W]
        """
        pred = self.padding_reflect(pred)
        targ = self.padding_reflect(targ)

        mu_p = self.mu_pred(pred)
        mu_t = self.mu_targ(targ)
        sigma_p = self.sigma_pred(pred ** 2) - mu_p ** 2
        sigma_t = self.sigma_targ(targ ** 2) - mu_t ** 2
        sigma_pt = self.sigma_pred_targ(pred * targ) - mu_p * mu_t

        SSIM_n = (2 * mu_p * mu_t + self.C1) * (2 * sigma_pt + self.C2)
        SSIM_d = (mu_p ** 2 + mu_t ** 2 + self.C1) * (sigma_p + sigma_t + self.C2)

        return SSIM_n / SSIM_d


def calc_pe(predict, target, alpha=0.85):
    """
    Calculates the photometric error between two images using SSIM and L1Loss
    :param [torch.tensor] predict: The predicted images in format [batch_size, 3, H, W]
    :param [torch.tensor] target: The target images in format [batch_size, 3, H, W]
    :param [float] alpha: Constant that determines how much the SSIM value and L1loss are weighted in the error
    :return [torch.tensor]: The numerical loss for each pixel in format [batch_size, 1, H, W]
    """
    ssim = SSIM()
    ssim_val = torch.mean(torch.clamp((1 - ssim(predict, target)) / 2, 0, 1), 1, True)
    l1 = torch.mean(torch.abs(predict - target), 1, True)

    return alpha * ssim_val + (1-alpha) * l1


def calc_smooth_loss(disp, image):
    """
    Calculates the edge-aware smoothness of the given depth map with relation to the target image. Returns a higher
    loss if the depth map fluctates a lot in depth where it should be smooth.
    :param [torch.tensor] disp: The depth map, formatted as [batch_size, 1, H, W]
    :param [torch.tensor] image: The target image, formatted as [batch_size, 3, H, W]
    :return [torch.float]: A 0 dimensional tensor containing a numerical loss punishing for a rough depth map
    """
    d_disp_x = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    d_disp_y = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])

    d_color_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, True)
    d_color_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, True)

    d_disp_x *= torch.exp(-d_color_x)
    d_disp_y *= torch.exp(-d_color_y)

    return d_disp_x.mean() + d_disp_x.mean()


def get_mask(targets, sources, min_reproject_errors):
    """
    Calculates the auto-masking for each pixel in the images. If a given pixel's photometric error between the source
    images and the target image is less than the photometric error between the reprojected images and the target image,
    then the auto-masking feature will be 0 for that point, eliminating its contribution to the loss.
    :param [torch.tensor] targets: The target images, in format [batch_size, 3, H, W]
    :param [torch.tensor] sources: The source images, in format [num_source_imgs, batch_size, 3, H, W]
    :param [torch.tensor] min_reproject_errors: The calculated photometric errors between the reprojected images and
    the target image, formatted as [batch_size, 1, H, W]
    :return [torch.tensor]: A binary mask containing either a 1 or 0 which allows a given pixel to be represented or
    to be ignored, respectively. Formatted as [batch_size, 1, H, W]
    """
    source_error = []
    for source in sources:
        source_error.append(calc_pe(source, targets))

    source_error = torch.cat(source_error, dim=1)
    min_source_errors, _ = torch.min(source_error, dim=1)

    return min_reproject_errors < min_source_errors


def calc_loss(inputs, outputs, smooth_term=0.001):
    """
    Takes in the inputs and outputs from the neural network to calulate a numeric loss value based on the Monodepth2
    paper.
    :param [dict] inputs: Contains the keys "targets" and "sources" which are tensors [batch_size, 3, H, W] and
    [num_src_imgs, batch_size, 3, H, W] respectively
    :param [dict] outputs: Contains the keys "reproj" and "depth" which are tensors
    [num_reprojected_imgs, batch_size, 3, H, W] and [batch_size, H, W] respectively
    :param [float] smooth_term: Constant that controls how much the smoothing term is considered in the loss
    :return [torch.float]: A 0 dimensional tensor representing the loss calculated
    """
    targets = inputs["targets"]
    sources = inputs["sources"]
    reprojections = outputs["reproj"]
    batch_size = len(targets)
    loss = 0

    reproj_errors = []
    for reproj in reprojections:
        # print(targets, "BLELBELBE\n")
        # print(reproj, "EWOIFJWEOI\n")
        reproj_errors.append(calc_pe(reproj, targets))
    # Could do something like reproj_errors[:, 1] = calc_pe(...) or smth like that
    reproj_errors = torch.cat(reproj_errors, dim=1)

    min_errors, _ = torch.min(reproj_errors, dim=1)

    # Masking
    reproj_errors *= get_mask(targets, sources, min_errors)

    depth = outputs["depth"]
    normalized_depth = depth / depth.mean(2, True).mean(3, True)
    loss += min_errors.mean() + smooth_term * calc_smooth_loss(normalized_depth, targets)

    return loss


def process_depth(src_images, depths, poses, tgt_intr, src_intr):
    """
    Reprojects a batch of source images into the target frame, using the target depth map, relative pose between the
    two frames, and the target and source intrinsic matrices.
    :param [list] src_images: List of dictionaries, with each dictionary representing one source frame (like t-1, t+1,
    stereo), and keys "stereo" which indicates whether it is a stereo or temporal source image and "images" which is a
    tensor containing the actual image data, in format [batch_size, 3, H, W]
    :param [torch.tensor] depths: Tensor containing the depth maps as determined from the target images, in the format
    [batch_size, 1, H, W]
    :param [torch.tensor] poses: Tensor containing the relative poses for each given source image to the target frame,
    in format [num_source_imgs, batch_size, 4, 4]
    :param [np.ndarray] tgt_intr: The intrinsic matrix for the target camera, as a 3x3 NumPy array
    :param [np.ndarray] src_intr: The intrinsic matrix for the source (stereo) camera, as a 3x3 NumPy array
    :return [torch.tensor]: Returns a tensor containing the reprojected images, in format [num_source_imgs, batch_size,
    3, H, W]
    """
    img_shape = src_images[0]["images"][0, 0].shape

    reprojected = torch.zeros((len(src_images), len(depths), 3, img_shape[0], img_shape[1]), dtype=torch.float)

    # Creates an array of all image coordinates: [0, 0], [1, 0], [2, 0], etc.
    img_indices = torch.ones((img_shape[0] * img_shape[1], 3))
    img_coords = torch.from_numpy(np.indices(img_shape).ravel().reshape(-1, 2, order="F"))
    img_indices[:, 1] = img_coords[:, 0]
    img_indices[:, 0] = img_coords[:, 1]

    # Converts intrinsic matrices into torch tensors, also inverting those that need to be inverted
    tgt_intr_torch_T = torch.from_numpy(tgt_intr.T).float()
    src_intr_torch_T = torch.from_numpy(src_intr.T).float()
    tgt_intr_inv_torch_T = torch.inverse(tgt_intr_torch_T)

    t_poses = poses.transpose(2, 3)

    # Iterates through all source image types (t+1, t-1, etc.)
    for i in range(len(src_images)):
        if src_images[i]["stereo"]:
            src_intr_T = src_intr_torch_T
        else:
            src_intr_T = tgt_intr_torch_T

        # Iterates through all images in batch
        for j in range(len(depths)):
            world_coords = torch.ones(img_indices.shape[0], 4)

            world_coords[:, :3] = img_indices @ tgt_intr_inv_torch_T * depths[j, 0].view(-1, 1)

            src_coords = torch.empty(img_indices.shape[0], 5)
            src_coords[:, 3:] = img_indices[:, :2]

            src_coords[:, :3] = (world_coords @ t_poses[i, j])[:, :3] @ src_intr_T

            src_coords = src_coords[src_coords[:, 2] > 0]
            src_coords[:, :2] = src_coords[:, :2] / src_coords[:, 2].reshape(-1, 1)

            src_coords = src_coords[
                (src_coords[:, 1] >= 0) & (src_coords[:, 1] <= img_shape[0] - 1) & (src_coords[:, 0] >= 0) & (
                            src_coords[:, 0] <= img_shape[1] - 1)]

            # Put nan here in case a pixel isn't filled
            reproj_image = torch.from_numpy(np.full((3, img_shape[0], img_shape[1]), np.nan, dtype=np.float32))

            # Bilinear sampling
            x = src_coords[:, 0]
            y = src_coords[:, 1]
            x12 = (torch.floor(x).long(), torch.ceil(x).long())
            y12 = (torch.floor(y).long(), torch.ceil(y).long())
            xdiff = (x - x12[0], x12[1] - x)
            ydiff = (y - y12[0], y12[1] - y)
            src_img = src_images[i]["images"][j]
            reproj_image[:, src_coords[:, 4].long(), src_coords[:, 3].long()] = \
                src_img[:, y12[0], x12[0]] * xdiff[1] * ydiff[1] + \
                src_img[:, y12[0], x12[1]] * xdiff[0] * ydiff[1] + \
                src_img[:, y12[1], x12[0]] * xdiff[1] * ydiff[0] + \
                src_img[:, y12[1], x12[1]] * xdiff[0] * ydiff[0]

            int_coords = (x12[0] == x12[1]) | (y12[0] == y12[1])
            if int_coords.any():
                rounded_coords = src_coords[int_coords].round().long()
                reproj_image[:, rounded_coords[:, 4], rounded_coords[:, 3]] = src_img[:, rounded_coords[:, 1], rounded_coords[:, 0]].float()

            reprojected[i, j] = reproj_image

    return reprojected


if __name__ == "__main__":

    from addon import get_relative_pose, calc_transformation_matrix

    ssim = SSIM()
    test_t = torch.arange(36, dtype=torch.float).reshape(2, 3, 2, 3)
    test_r = torch.rand((2, 3, 2, 3), dtype=torch.float)
    # test_s = torch.arange(6).reshape(3, 2)

    target = torch.arange(108, dtype=torch.float).reshape(2, 3, 3, 6)
    sources = (torch.zeros(target.shape), target)
    reprojs = (target, target)
    depth = torch.arange(36, dtype=torch.float).reshape(2, 1, 3, 6)

    inputs = {"targets": target, "sources": sources}
    outputs = {"reproj": reprojs, "depth": depth}
    l = calc_loss(inputs, outputs, 1)
    print(l)

    print(test_r)
    print(ssim(test_t, test_r))
    out = ssim(test_t, test_r).mean(1, True)
    print(out)
    print(out.shape)

    pe = calc_pe(test_t, test_r)
    print(pe)
    print(pe.shape)

    print("hi")
    # open("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0")

    from dataloader import KittiDataset
    from kitti_utils import get_camera_intrinsic_dict, get_relative_rotation_stereo, get_relative_translation_stereo
    from compute_photometric_error_utils import compute_relative_pose_matrix

    calibration_dir = 'data/kitti_example/2011_09_26'
    SCENE_PATH = r"data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync"

    TEST_CONFIG_PATH = "configs/kitti_dataset.yml"
    d = KittiDataset.init_from_config(TEST_CONFIG_PATH)


    bruh = np.load("data/0000000000_disp.npy")
    bruh = torch.from_numpy(bruh)
    print(bruh)
    print(bruh.shape)
    # bruhF = F.interpolate(bruh, [375, 1242], mode="bilinear", align_corners=False).reshape(1, 375, 1242)
    bruh = F.interpolate(bruh, [375, 1242], mode="bilinear", align_corners=False)
    # print(bruhF.shape)

    # bruh.fill_(1)
    img_shape = bruh.shape[2:]

    target = np.copy(d[0]["stereo_left_image"])
    source = np.copy(d[1]["stereo_left_image"])
    # source = np.copy(d[0]["stereo_right_image"])
    rel_pose = get_relative_pose(SCENE_PATH, 0, 1)
    # v_forward = rel_pose[0][3]
    # v_leftward = rel_pose[1][3]
    # v_upward = rel_pose[2][3]
    # new_translation = np.array([-1 * v_leftward, -1 * v_upward, v_forward, 1.])
    # rel_pose[:, 3] = new_translation

    rot = np.deg2rad([0, 0, 0])
    tran = np.array([0, 0, -1])
    # print(rot)

    # rel_pose = calc_transformation_matrix(rot, tran)
    # stereo pose
    relative_rotation = get_relative_rotation_stereo(calibration_dir)
    relative_translation = get_relative_translation_stereo(calibration_dir)
    # rel_pose = compute_relative_pose_matrix(relative_translation, relative_rotation)


    # min_disp = 1 / 100
    # max_disp = 1 / 0.1
    # bruh = 1 / (max_disp * bruh + min_disp)
    # bruh = bruh * 5.4
    # bruh = 1 / (min_disp + (max_disp - min_disp) * bruh)

    # bruh = 54 * 721 / (1242 * bruh)

    bruh = 31.257 / bruh


    # bruh = 5.4 / bruh

    tgt_intrinsic = get_camera_intrinsic_dict(calibration_dir).get('stereo_left')
    src_intrinsic = get_camera_intrinsic_dict(calibration_dir).get('stereo_right')

    tgt_intrinsic[0] = tgt_intrinsic[0] * (img_shape[1] / 1242)
    tgt_intrinsic[1] = tgt_intrinsic[1] * (img_shape[0] / 375)

    target = F.interpolate(torch.from_numpy(target).permute(2, 0, 1).reshape(1, 3, 375, 1242).float(), (img_shape[0], img_shape[1]), mode="bilinear", align_corners=False)
    source = F.interpolate(torch.from_numpy(source).permute(2, 0, 1).reshape(1, 3, 375, 1242).float(), (img_shape[0], img_shape[1]), mode="bilinear", align_corners=False)

    plt.imshow(bruh[0, 0])
    plt.show()

    # plt.imshow(target[0].permute(1, 2, 0)/255)
    # plt.show()

    rel_pose = torch.from_numpy(rel_pose).reshape(1, 1, 4, 4).float()

    source_dict = [{"stereo": False, "images": source}]


    out_img = process_depth(source_dict, bruh, rel_pose, tgt_intrinsic, src_intrinsic)

    plt.imshow(out_img[0, 0].permute(1, 2, 0) / 255)
    plt.show()

    out_img = np.array(out_img[0, 0].permute(1, 2, 0))


    def disp_to_depth(disp, min_depth, max_depth):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth






    # for i in range(batch_size):
    #     target = outputs["targets"][i]
    #     reprojections = outputs["reproj"][i]
    #     reproj_errors = []
    #     for reproj in range(len(reprojections)):
    #         reproj_errors.append(calc_pe(target, reproj))
    #
    #     reproj_errors = torch.cat(reproj_errors, dim=1)
    #
    #     # Masking
    #     # reproj_errors *= get_mask()
    #
    #     min_errors, _ = torch.min(reproj_errors, dim=1)
    #
    #     depth = outputs["depth"]
    #     normalized_depth = depth / depth.mean(2, True).mean(3, True)
    #     loss += min_errors.mean() + LAMBDA * calc_smooth_loss(normalized_depth, target)
