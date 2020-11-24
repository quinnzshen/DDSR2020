import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """
    Adpated from https://github.com/nianticlabs/monodepth2/blob/master/layers.py
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
        :return [torch.tensor]: A tensor representing how similar the two images are, on a pixel basis in the format [batch_size, 3, H, W]
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
    Calculates the photometric error between two images using SSIM and L1Loss. 
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
    Calculates the edge-aware smoothness of the given disparity map with relation to the target image. Returns a higher loss if the disparity map 
    fluctates a lot in disparity where it should be smooth. Adapted from https://github.com/nianticlabs/monodepth2/blob/master/layers.py
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
    then the auto-masking feature will be 0 for that point.
    :param [torch.tensor] targets: The target images, in format [batch_size, 3, H, W]
    :param [torch.tensor] sources: The source images, in format [num_source_imgs, batch_size, 3, H, W]
    :param [torch.tensor] min_reproject_errors: The calculated photometric errors between the reprojected images and
    the target image, formatted as [batch_size, 1, H, W]
    :return [torch.tensor]: A binary mask containing either True or False. Formatted as [batch_size, 1, H, W]
    """
    source_error = []
    for source in sources:
        source_error.append(calc_pe(source, targets))

    source_error = torch.cat(source_error, dim=1)
    min_source_errors, _ = torch.min(source_error, dim=1)
    return min_reproject_errors < min_source_errors


def calc_loss(inputs, outputs, scale=0, smooth_term=0.001):
    """
    Takes in the inputs and outputs from the neural network to calulate a numeric loss value based on the Monodepth2 paper.
    :param [dict] inputs: Contains the keys "targets" and "sources" which are tensors [batch_size, 3, H, W] and
    [num_src_imgs, batch_size, 3, H, W] respectively
    :param [dict] outputs: Contains the keys "reproj", "disparities", and "initial_masks" which are tensors
    [num_reprojected_imgs, batch_size, 3, H, W], [batch_size, 1, H, W], and [num_src_imgs, batch_size, 1, H, W]
    (dtype=torch.bool) respectively
    :param [int] scale: The scale number, applied to the smoothness term calculation
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


class GenerateReprojections(nn.Module):
    """
    Reprojects pixels from a source image onto a target frame.
    """
    def __init__(self, height, width, default_batch_size):
        """
        Sets up the layers used in the forward method.
        :param [int] height: height of input image
        :param [int] width: width of input image
        :param [int] default_batch_size: number of images in a batch
        """
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
        """
        Computes the reprojected images (source onto target)
        :param [torch.tensor] src_images: source images
        :param [torch.tensor] depths: depth maps
        :param [torch.tensor] poses: relative pose matrices
        :param [torch.tensor] tgt_intr: target intrinsics matrices
        :param [torch.tensor] src_intr: source intrinsics matrices
        :param [int] local_batch_size: size of batch
        :return [torch.tensor] reprojected: Tensor of reprojected images
        """
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