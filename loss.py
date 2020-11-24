import torch
import torch.nn as nn
import torch.nn.functional as F

from color_utils import color_difference


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

        return SSIM_n / (SSIM_d + 1e-7)


def calc_pe(predict, target, alpha=0.85, color="RGB"):
    """
    Calculates the photometric error between two images using SSIM and L1Loss
    :param [torch.tensor] predict: The predicted images in format [batch_size, 3, H, W]
    :param [torch.tensor] target: The target images in format [batch_size, 3, H, W]
    :param [float] alpha: Constant that determines how much the SSIM value and L1loss are weighted in the error
    :param [str] color: The color model to use for calculations
    :return [torch.tensor]: The numerical loss for each pixel in format [batch_size, 1, H, W]
    """
    ssim = SSIM()
    ssim_val = torch.mean(torch.clamp((1 - ssim(predict, target)) / 2, 0, 1), dim=1, keepdim=True)
    diff = color_difference(predict, target, color=color).unsqueeze(1)
    return alpha * ssim_val + (1 - alpha) * diff


def calc_smooth_loss(disp, image, color="RGB"):
    """
    Calculates the edge-aware smoothness of the given disparity map with relation to the target image. Returns a higher
    loss if the disparity map fluctates a lot in disparity where it should be smooth.
    :param [torch.tensor] disp: The disparity map, formatted as [batch_size, 1, H, W]
    :param [torch.tensor] image: The target image, formatted as [batch_size, 3, H, W]
    :param [str] color: The color model to use for calculations
    :return [torch.float]: A 0 dimensional tensor containing a numerical loss punishing for a rough depth map
    """
    # Based on Monodepth2 repo
    # Takes the derivative of the disparity map by subtracting a pixel with the pixel value to the left and above
    disp_dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    disp_dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])

    # Essentially same logic as above, but needs to be averaged because of the separate color channels
    image_dx = color_difference(image[:, :, :, 1:], image[:, :, :, :-1], color=color).unsqueeze(1)
    image_dy = color_difference(image[:, :, 1:, :], image[:, :, :-1, :], color=color).unsqueeze(1)

    disp_dx *= torch.exp(-image_dx)
    disp_dy *= torch.exp(-image_dy)

    return disp_dx.mean() + disp_dy.mean()


def calc_loss(inputs, outputs, scale=0, smooth_term=0.001, color="RGB"):
    """
    Takes in the inputs and outputs from the neural network to calulate a numeric loss value based on the Monodepth2
    paper.
    :param [dict] inputs: Contains the keys "targets" and "sources" which are tensors [batch_size, 3, H, W] and
    [num_src_imgs, batch_size, 3, H, W] respectively
    :param [dict] outputs: Contains the keys "reproj", "disparities", and "initial_masks" which are tensors
    [num_reprojected_imgs, batch_size, 3, H, W], [batch_size, 1, H, W], and [num_src_imgs, batch_size, 1, H, W]
    (dtype=torch.bool) respectively
    :param [int] scale: The scale number, applied to the smoothness term calculation
    :param [float] smooth_term: Constant that controls how much the smoothing term is considered in the loss
    :param [str] color: The color model to use for calculations
    :return [tuple]: Returns a 3 element tuple containing: a float representing the calculated loss, a torch.Tensor
    with dimensions [batch_size, H, W] representing the auto-mask, and a torch.Tensor of dimensions [batch_size, H,
    W] representing the minimum photometric error calculated
    """
    targets = inputs["targets"]
    sources = inputs["sources"]
    reprojections = outputs["reproj"]

    loss = 0

    reproj_errors = torch.stack([calc_pe(reprojections[i], targets, color=color).squeeze(1) for i in range(len(reprojections))])

    # Source errors
    source_errors = torch.stack([calc_pe(sources[i], targets, color=color).squeeze(1) for i in range(len(sources))])
    combined_errors = torch.cat((source_errors, reproj_errors), dim=0)
    
    min_errors, _ = torch.min(combined_errors, dim=0)
    min_error_vis = min_errors.detach().clone()
    
    disp = outputs["disparities"]
    normalized_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)

    loss = loss + min_errors.mean()
    loss = loss + smooth_term * calc_smooth_loss(normalized_disp, targets, color) / (2 ** scale)

    return loss, min_error_vis


class GenerateReprojections(nn.Module):
    def __init__(self, height, width, default_batch_size):
        super(GenerateReprojections, self).__init__()

        self.h = height
        self.w = width
        self.batch_size = default_batch_size

        meshgrid = torch.meshgrid([
            torch.arange(height, dtype=torch.float,),
            torch.arange(width, dtype=torch.float,)
        ])
        img_coords = torch.stack((meshgrid[1].reshape(-1), meshgrid[0].reshape(-1)), dim=0).unsqueeze(0).repeat(
            default_batch_size, 1, 1)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, width * height), requires_grad=False)
        self.img_indices = nn.Parameter(torch.cat([img_coords, self.ones], 1), requires_grad=False)

    def forward(self, src_images, depths, poses, tgt_intr, src_intr, local_batch_size):
        reprojected = []
        tgt_intr_inv = tgt_intr.inverse()
        for i in range(len(poses)):
            world_coords = depths.view(local_batch_size, 1, -1) * (tgt_intr_inv[:, :3, :3] @ self.img_indices[:local_batch_size])
            world_coords = torch.cat([world_coords, self.ones[:local_batch_size]], dim=1)
            src_coords = (src_intr[i] @ poses[i][:, :3]) @ world_coords

            src_coords = src_coords[:, :2] / (src_coords[:, 2].unsqueeze(1) + 1e-7)
            src_coords = src_coords.view(local_batch_size, 2, self.h, self.w)
            src_coords = src_coords.permute(0, 2, 3, 1)
            src_coords[..., 0] /= self.w - 1
            src_coords[..., 1] /= self.h - 1
            src_coords = (src_coords - 0.5) * 2

            reprojected.append(F.grid_sample(src_images[i], src_coords, padding_mode="border", align_corners=False))
        return reprojected
