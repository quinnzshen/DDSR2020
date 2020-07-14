import torch
import torch.nn as nn


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
    :param [torch.tensor] disp: The disparity map, formatted as [batch_size, 1, H, W]
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
    :return [torch.tensor]: A binary mask containing either True or False which allows a given pixel to be represented
    or to be ignored, respectively. Formatted as [batch_size, 1, H, W]
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
    :return [torch.float]: A float representing the calculated loss
    """
    targets = inputs["targets"]
    sources = inputs["sources"]
    reprojections = outputs["reproj"]
    loss = 0

    shape = list(targets.shape)
    shape[1] = reprojections.shape[0]
    reproj_errors = torch.empty(shape, dtype=torch.float)
    for i in range(len(reprojections)):
        reproj_errors[:, i] = calc_pe(reprojections[i], targets).squeeze(1)

    min_errors, _ = torch.min(reproj_errors, dim=1)

    # Masking
    reproj_errors *= get_mask(targets, sources, min_errors)

    depth = outputs["depth"]
    normalized_depth = depth / depth.mean(2, True).mean(3, True)
    loss += min_errors.mean() + smooth_term * calc_smooth_loss(normalized_depth, targets)

    return loss
