import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SSIM(nn.Module):
    """
    Based off SSIM in Monodepth2 repo
    """

    def __init__(self):
        """
        Sets up the layers/pooling to run the forward method which actually does the SSIM computation, measuring how
        "structurally similar" the images are compared to each other.
        """
        super(SSIM, self).__init__()
        # Pads image with reflection of its pixels
        self.padding_reflect = nn.ReflectionPad2d(1)

        # Goes across the image and averages with a 3x3 kernel
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
    ssim_val = torch.mean(torch.clamp((1 - ssim(predict, target)) / 2, 0, 1), dim=1, keepdim=True)
    l1 = torch.mean(torch.abs(predict - target), dim=1, keepdim=True)

    return alpha * ssim_val + (1 - alpha) * l1


def calc_smooth_loss(disp, image):
    """
    Calculates the edge-aware smoothness of the given disparity map with relation to the target image. Returns a higher
    loss if the disparity map fluctates a lot in disparity where it should be smooth.
    :param [torch.tensor] disp: The disparity map, formatted as [batch_size, 1, H, W]
    :param [torch.tensor] image: The target image, formatted as [batch_size, 3, H, W]
    :return [torch.float]: A 0 dimensional tensor containing a numerical loss punishing for a rough depth map
    """
    # Based on Monodepth2 repo
    # Takes the derivative of the disparity map by subtracting a pixel with the pixel value to the left and above
    disp_dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    disp_dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])

    # Essentially same logic as above, but needs to be averaged because of the 3 separate color channels
    image_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, True)
    image_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, True)

    disp_dx *= torch.exp(-image_dx)
    disp_dy *= torch.exp(-image_dy)

    return disp_dx.mean() + disp_dy.mean()


def get_mask(targets, sources, min_reproject_errors):
    """
    Calculates the auto-masking for each pixel in the images. If a given pixel's photometric error between the source
    images and the target image is less than the photometric error between the reprojected images and the target image,
    then the auto-masking feature will be 0 for that point, eliminating its contribution to the loss.
    :param [torch.tensor] targets: The target images, in format [batch_size, 3, H, W]
    :param [torch.tensor] sources: The source images, in format [num_source_imgs, batch_size, 3, H, W]
    :param [torch.tensor] min_reproject_errors: The calculated photometric errors between the reprojected images and
    the target image, formatted as [batch_size, 1, H, W]
    :return [torch.tensor]: A binary mask containing either True or False which allows a given pixel to be represented
    or to be ignored, respectively. Formatted as [batch_size, 1, H, W]
    """
    source_error = []
    for source in sources:
        source_error.append(calc_pe(source, targets))

    source_error = torch.cat(source_error, dim=1)
    min_source_errors, _ = torch.min(source_error, dim=1)
    return min_reproject_errors < min_source_errors


def calc_loss(inputs, outputs, scale=0, smooth_term=0.001):
    """
    Takes in the inputs and outputs from the neural network to calulate a numeric loss value based on the Monodepth2
    paper.
    :param [dict] inputs: Contains the keys "targets" and "sources" which are tensors [batch_size, 3, H, W] and
    [num_src_imgs, batch_size, 3, H, W] respectively
    :param [dict] outputs: Contains the keys "reproj", "disparities", and "initial_masks" which are tensors
    [num_reprojected_imgs, batch_size, 3, H, W], [batch_size, 1, H, W], and [num_src_imgs, batch_size, 1, H, W]
    (dtype=torch.bool) respectively
    :param [float] smooth_term: Constant that controls how much the smoothing term is considered in the loss
    :return [tuple]: Returns a 3 element tuple containing: a float representing the calculated loss, a torch.Tensor
    with dimensions [batch_size, H, W] representing the auto-mask, and a torch.Tensor of dimensions [batch_size, H,
    W] representing the minimum photometric error calculated
    """
    targets = inputs["targets"]
    sources = inputs["sources"]
    reprojections = outputs["reproj"]

    loss = 0

    reproj_errors = torch.stack([calc_pe(reprojections[i], targets).squeeze(1) for i in range(len(reprojections))])
    min_errors_reproj, _ = torch.min(reproj_errors, dim=0)
    mask = get_mask(targets, sources, min_errors_reproj)

    # Source errors
    source_errors = torch.stack([calc_pe(sources[i], targets).squeeze(1) for i in range(len(sources))])
    combined_errors = torch.cat((source_errors, reproj_errors), dim=0)
    
    min_errors, _ = torch.min(combined_errors, dim=0)
    min_error_vis = min_errors.detach().clone()
    

    disp = outputs["disparities"]
    normalized_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)

    loss = loss + min_errors.mean()
    loss = loss + smooth_term * calc_smooth_loss(normalized_disp, targets) / (2 ** scale)

    return loss, mask, min_error_vis


def process_depth(src_images, depths, poses, tgt_intr, src_intr, img_shape):
    """
    Reprojects a batch of source images into the target frame, using the target depth map, relative pose between the
    two frames, and the target and source intrinsic matrices.
    :param [torch.tensor] src_images: Tensor of source images, where dimension 0 separates the different type of source
    images. In format [num_source_images, batch_size, 3, H, W]
    :param [torch.tensor] depths: Tensor containing the depth maps as determined from the target images, in the format
    [batch_size, 1, H, W]
    :param [torch.tensor] poses: Tensor containing the relative poses for each given source image to the target frame,
    in format [num_source_imgs, batch_size, 4, 4]
    :param [torch.tensor] tgt_intr: The intrinsic matrices for the target camera, in format [batch_size, 3, 3]
    :param [torch.tensor] src_intr: The intrinsic matrix for the source camera, in format
    [num_source_imgs, batch_size, 3, 3]
    :param [tuple] img_shape: An integer, indexable data type where the 0th and 1st index represent the height and width
    of the images respectively.
    :return [tuple]: Returns a tuple containing 2 tensors, the first containing the reprojected images, in format
    [num_source_imgs, batch_size, 3, H, W], and the second containing binary masks recording which pixels were able to
    be reprojected back onto target, in format [num_source_imgs, 1, batch_size, H, W]
    """
    reprojected = torch.full((len(src_images), len(depths), 3, img_shape[0], img_shape[1]), 127, dtype=torch.float,
                             device=poses.device)

    # Creates an array of all image coordinates: [0, 0], [1, 0], [2, 0], etc.
    img_ones = torch.ones((img_shape[0] * img_shape[1], 1), device=poses.device)
    img_coords = torch.meshgrid([
        torch.arange(img_shape[0], dtype=torch.float, device=poses.device),
        torch.arange(img_shape[1], dtype=torch.float, device=poses.device)
    ])
    img_indices = torch.cat((img_coords[1].reshape(-1, 1), img_coords[0].reshape(-1, 1), img_ones), dim=1)

    # Transposes intrinsic matrices, also inverting those that need to be inverted
    tgt_intr_torch_T = tgt_intr.transpose(1, 2)
    src_intr_torch_T = src_intr.transpose(2, 3)
    tgt_intr_inv_torch_T = tgt_intr_torch_T.inverse()

    t_poses = poses.transpose(2, 3)

    # Iterates through all source image types (t+1, t-1, etc.)
    for i in range(len(src_images)):
        # Iterates through all images in batch
        for j in range(len(depths)):

            world_coords = torch.cat((img_indices @ tgt_intr_inv_torch_T[j] * depths[j, 0].view(-1, 1),
                                      torch.ones(img_indices.shape[0], 1, device=poses.device)), dim=1)

            src_coords = torch.cat(((world_coords @ t_poses[i, j])[:, :3] @ src_intr_torch_T[i, j], img_indices[:, :2]),
                                   dim=1)
            src_coords = src_coords[src_coords[:, 2] > 0]

            src_coords = torch.cat((src_coords[:, :2] / src_coords[:, 2].reshape(-1, 1), src_coords[:, 2:]), dim=1)

            src_coords = src_coords[
                (src_coords[:, 1] >= 0) & (src_coords[:, 1] <= img_shape[0] - 1) & (src_coords[:, 0] >= 0) & (
                        src_coords[:, 0] <= img_shape[1] - 1)]
            # Bilinear sampling
            x = src_coords[:, 0]
            y = src_coords[:, 1]
            x12 = (torch.floor(x).long(), torch.ceil(x).long())
            y12 = (torch.floor(y).long(), torch.ceil(y).long())
            xdiff = (x - x12[0], x12[1] - x)
            ydiff = (y - y12[0], y12[1] - y)
            src_img = src_images[i, j]
            reprojected[i, j, :, src_coords[:, 4].long(), src_coords[:, 3].long()] = \
                src_img[:, y12[0], x12[0]] * xdiff[1] * ydiff[1] + \
                src_img[:, y12[0], x12[1]] * xdiff[0] * ydiff[1] + \
                src_img[:, y12[1], x12[0]] * xdiff[1] * ydiff[0] + \
                src_img[:, y12[1], x12[1]] * xdiff[0] * ydiff[0]

            int_coords = (x12[0] == x12[1]) | (y12[0] == y12[1])
            if int_coords.any():
                rounded_coords = src_coords[int_coords].round().long()
                reprojected[i, j, :, rounded_coords[:, 4], rounded_coords[:, 3]] = src_img[:, rounded_coords[:, 1], rounded_coords[:, 0]].float()

            # Using F.grid_sample
            # pic_coords = torch.empty((1, img_shape[0], img_shape[1], 2), device=poses.device)
            # pic_coords[0, src_coords[:, 4].long(), src_coords[:, 3].long()] = 2 * src_coords[:, :2] / torch.tensor([img_shape[1], img_shape[0]], device=poses.device) - 1
            # reprojected[i, j] = F.grid_sample(src_images[i, j].unsqueeze(0), pic_coords, padding_mode="border", align_corners=False)

    return reprojected
