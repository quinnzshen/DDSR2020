import torch
import torch.nn as nn


ALPHA = 0.85
LAMBDA = 1


class SSIM(nn.Module):
    """
    Based off SSIM in Monodepth2 repo
    """
    def __init__(self):
        """
        Sets up the layers/pooling to run the forward method which actually does the computation
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


def calc_pe(predict, target):
    """
    Calculates the photometric error between two images using SSIM and L1Loss
    :param [torch.tensor] predict: The predicted images in format [batch_size, 3, H, W]
    :param [torch.tensor] target: The target images in format [batch_size, 3, H, W]
    :return [torch.tensor]: The numerical loss for each pixel in format [batch_size, 1, H, W]
    """
    ssim = SSIM()
    ssim_val = torch.mean(torch.clamp((1 - ssim(predict, target)) / 2, 0, 1), dim=1, keepdim=True)
    l1 = torch.mean(torch.abs(predict - target), dim=1, keepdim=True)

    return ALPHA * ssim_val + (1-ALPHA) * l1
