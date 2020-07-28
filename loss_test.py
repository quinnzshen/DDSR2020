import torch
import pytest

from loss import SSIM, calc_pe, calc_smooth_loss, get_mask, calc_loss, process_depth


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
    torch.testing.assert_allclose(calc_pe(img3, img4, 0.85), ans, atol=0.01, rtol=0.0001)
    torch.testing.assert_allclose(calc_pe(img3, img4), calc_pe(img4, img3))


def test_calc_smooth_loss():
    img1 = torch.ones(1, 3, 4, 5)
    disp1 = torch.zeros(1, 1, 4, 5)
    torch.testing.assert_allclose(calc_smooth_loss(disp1, img1), torch.tensor(0).float())

    img2 = torch.zeros(1, 3, 2, 3)
    disp2 = torch.ones(1, 3, 2, 3)
    disp2[:, :, :, 1] = 16
    torch.testing.assert_allclose(calc_smooth_loss(disp2, img2), torch.tensor(15).float())

    img2[:, :, :, 1] = 255
    torch.testing.assert_allclose(calc_smooth_loss(disp2, img2), torch.tensor(0).float())


def test_get_mask():
    target1 = torch.ones(3, 3, 4, 5)
    source1 = torch.zeros(3, 3, 3, 4, 5)
    reproj_err1 = torch.zeros(3, 1, 4, 5)
    torch.testing.assert_allclose(get_mask(target1, source1, reproj_err1).float(), torch.ones(3, 1, 4, 5))

    reproj_err2 = torch.full((3, 1, 4, 5), 1, dtype=torch.float)
    torch.testing.assert_allclose(get_mask(target1, source1, reproj_err2).float(), torch.zeros(3, 1, 4, 5))

    target1[:, :, :2] = 0
    ans = torch.ones(3, 1, 4, 5)
    ans[:, :, :2] = 0
    reproj_err3 = torch.full((3, 1, 4, 5), 0.5)
    torch.testing.assert_allclose(get_mask(target1, source1, reproj_err3).float(), ans)


def test_calc_loss():
    input1 = {
        "targets": torch.zeros(3, 3, 4, 5),
        "sources": torch.ones(3, 3, 3, 4, 5)
    }
    output1 = {
        "reproj": torch.zeros(3, 3, 3, 4, 5),
        "disparities": torch.ones(3, 1, 4, 5),
        "initial_masks": torch.ones(3, 3, 1, 4, 5, dtype=torch.bool)
    }
    torch.testing.assert_allclose(calc_loss(input1, output1), torch.tensor(0).float())

    input2 = {
        "targets": torch.arange(108, dtype=torch.float).reshape(2, 3, 3, 6),
        "sources": torch.zeros(2, 2, 3, 3, 6)
    }
    output2 = {
        "reproj": torch.full((2, 2, 3, 3, 6), 5, dtype=torch.float),
        "disparities": torch.eye(6, dtype=torch.float).reshape(2, 1, 3, 6) * 5,
        "initial_masks": torch.arange(72).reshape(2, 2, 1, 3, 6) % 2 == 0
    }
    torch.testing.assert_allclose(calc_loss(input2, output2, 0.001), torch.tensor(7.6757))


def test_process_depth():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src_img1 = torch.ones(3, 2, 3, device=device)
    src_img1[:, 0, 0] = 5

    source_imgs1 = src_img1.repeat(2, 1, 1, 1, 1)
    depths1 = torch.ones(1, 1, 2, 3, device=device)
    poses1 = torch.eye(4, device=device).repeat(2, 1, 1, 1)
    tgt_intrs1 = torch.eye(3, dtype=torch.float, device=device).reshape(1, 3, 3)
    src_intrs1 = torch.cat((tgt_intrs1[0], torch.zeros(3, 3, device=device)), dim=0).reshape(2, 1, 3, 3)

    reproj1, mask1 = process_depth(source_imgs1, depths1, poses1, tgt_intrs1, src_intrs1,
                                   (src_img1.shape[1], src_img1.shape[2]))

    torch.testing.assert_allclose(reproj1[0], src_img1)
    torch.testing.assert_allclose(reproj1[1], torch.full((1, 3, 2, 3), 127, dtype=torch.float, device=device))
    torch.testing.assert_allclose(mask1[0].float(), torch.ones(1, 1, 2, 3, device=device))
    torch.testing.assert_allclose(mask1[1].float(), torch.zeros(1, 1, 2, 3, device=device))

    source_imgs2 = torch.arange(120, dtype=torch.float, device=device).reshape(1, 2, 3, 4, 5)

    tgt_intrs2 = torch.eye(3, dtype=torch.float, device=device).repeat(2, 1, 1)
    tgt_intrs2[1, :2, 2] = 5
    src_intrs2 = tgt_intrs2.reshape(1, 2, 3, 3)
    depths2 = torch.arange(40, device=device).reshape(2, 1, 4, 5)
    poses2 = torch.eye(4, device=device).repeat(1, 2, 1, 1)
    poses2[0, :, 0, 1] = 0.4
    poses2[0, 0, 1, 3] = 5

    reproj2, mask2 = process_depth(source_imgs2, depths2, poses2, tgt_intrs2, src_intrs2, (4, 5))
    exp_ans2 = torch.tensor([[[[[127.0000, 127.0000, 12.0000, 13.0000, 9.0000],
                                [10.0000, 10.5667, 10.9714, 11.5250, 127.0000],
                                [13.3000, 14.0727, 14.8833, 15.7231, 127.0000],
                                [127.0000, 127.0000, 127.0000, 127.0000, 127.0000]],

                               [[127.0000, 127.0000, 32.0000, 33.0000, 29.0000],
                                [30.0000, 30.5667, 30.9714, 31.5250, 127.0000],
                                [33.3000, 34.0727, 34.8833, 35.7231, 127.0000],
                                [127.0000, 127.0000, 127.0000, 127.0000, 127.0000]],

                               [[127.0000, 127.0000, 52.0000, 53.0000, 49.0000],
                                [50.0000, 50.5667, 50.9714, 51.5250, 127.0000],
                                [53.3000, 54.0727, 54.8833, 55.7231, 127.0000],
                                [127.0000, 127.0000, 127.0000, 127.0000, 127.0000]]],

                              [[[127.0000, 127.0000, 60.0000, 61.0000, 62.0000],
                                [127.0000, 127.0000, 65.0000, 66.0000, 67.0000],
                                [127.0000, 127.0000, 71.0000, 72.0000, 73.0000],
                                [127.0000, 75.0000, 76.0000, 77.0000, 78.0000]],

                               [[127.0000, 127.0000, 80.0000, 81.0000, 82.0000],
                                [127.0000, 127.0000, 85.0000, 86.0000, 87.0000],
                                [127.0000, 127.0000, 91.0000, 92.0000, 93.0000],
                                [127.0000, 95.0000, 96.0000, 97.0000, 98.0000]],

                               [[127.0000, 127.0000, 100.0000, 101.0000, 102.0000],
                                [127.0000, 127.0000, 105.0000, 106.0000, 107.0000],
                                [127.0000, 127.0000, 111.0000, 112.0000, 113.0000],
                                [127.0000, 115.0000, 116.0000, 117.0000, 118.0000]]]]], device=device)

    torch.testing.assert_allclose(reproj2, exp_ans2)
    exp_mask2 = torch.tensor([[[[[False, False, True, True, True],
                                 [True, True, True, True, False],
                                 [True, True, True, True, False],
                                 [False, False, False, False, False]]],

                               [[[False, False, True, True, True],
                                 [False, False, True, True, True],
                                 [False, False, True, True, True],
                                 [False, True, True, True, True]]]]], device=device)
    torch.testing.assert_allclose(mask2.float(), exp_mask2.float())