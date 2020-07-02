import torch
import torch.nn as nn
import numpy as np


class SSIM(nn.Module):
    def __init__(self):
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
        pred = self.padding_reflect(pred)
        targ = self.padding_reflect(targ)

        mu_p = self.mu_pred(pred)
        mu_t = self.mu_targ(targ)
        sigma_p = self.sigma_pred(pred ** 2) - mu_p ** 2
        sigma_t = self.sigma_targ(targ ** 2) - mu_t ** 2
        sigma_pt = self.sigma_pred_targ(pred * targ) - pred * targ

        SSIM_n = (2 * mu_p * mu_t + self.C1) * (2 * sigma_pt + self.C2)
        SSIM_d = (mu_p ** 2 + mu_t ** 2 + self.C1) * (sigma_p + sigma_t + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
