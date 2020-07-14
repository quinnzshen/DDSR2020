import torch
import numpy as np
import pytest

from loss import SSIM, calc_pe


def test_SSIM():
    ssim = SSIM()
    img1 = torch.rand(2, 3, 5, 4)
    img2 = img1.clone()
    torch.testing.assert_allclose(ssim(img1, img2), torch.ones(2, 3, 5, 4))

    img3 = torch.arange(18).float().reshape(1, 3, 3, 2) / 100
    img4 = torch.zeros(1, 3, 3, 2)
    ans = torch.tensor([[[[0.1780, 0.2356],
                          [0.0933, 0.1175],
                          [0.0735, 0.0890]],

                         [[0.0137, 0.0149],
                          [0.0099, 0.0107],
                          [0.0101, 0.0109]],

                         [[0.0045, 0.0047],
                          [0.0035, 0.0037],
                          [0.0038, 0.0039]]]])

    torch.testing.assert_allclose(ssim(img3, img4), ans, atol=0.01, rtol=0.0001)
    torch.testing.assert_allclose(ssim(img3, img4), ssim(img4, img3))


def test_calc_pe():
    img1 = torch.rand(2, 3, 4, 5)
    img2 = img1.clone()
    torch.testing.assert_allclose(calc_pe(img1, img2), torch.zeros(2, 1, 4, 5))

    img3 = torch.arange(18).float().reshape(1, 3, 2, 3)
    img4 = torch.zeros(1, 3, 2, 3)

    ans = torch.tensor([[[[1.3250, 1.4750, 1.6250],
                          [1.7750, 1.9250, 2.0750]]]])
    torch.testing.assert_allclose(calc_pe(img3, img4), ans, atol=0.01, rtol=0.0001)
    torch.testing.assert_allclose(calc_pe(img3, img4), calc_pe(img4, img3))
