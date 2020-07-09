import torch
import torch.nn as nn
import numpy as np


ALPHA = 0.85
LAMBDA = 1


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
        sigma_pt = self.sigma_pred_targ(pred * targ) - mu_p * mu_t

        SSIM_n = (2 * mu_p * mu_t + self.C1) * (2 * sigma_pt + self.C2)
        SSIM_d = (mu_p ** 2 + mu_t ** 2 + self.C1) * (sigma_p + sigma_t + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def calc_pe(predict, target):
    """
    Calculates the photometric error between two images using SSIM and L1Loss
    :param [torch.tensor] predict: The predicted images in format [batch_size, 3, H, W]
    :param [torch.tensor] target: The target images in format [batch_size, 3, H, W]
    :return [torch.tensor]: The numerical loss for each pixel in format [batch_size, 1, H, W]
    """
    ssim = SSIM()
    ssim_val = torch.mean(torch.abs(predict - target), 1, True)
    l1 = torch.mean(ssim(predict, target), 1, True)
    # print(ssim_val)
    return ALPHA * ssim_val + (1-ALPHA) * l1


def calc_smooth_loss(disp, image):
    """
    TB continued
    :param disp:
    :param image:
    :return:
    """
    d_disp_x = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    d_disp_y = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])

    d_color_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, True)
    d_color_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, True)

    d_disp_x *= torch.exp(d_color_x)
    d_disp_y *= torch.exp(d_color_y)

    return d_disp_x.mean() + d_disp_x.mean()


def get_mask(targets, sources, min_reproject_errors):
    source_error = []
    for source in sources:
        source_error.append(calc_pe(source, targets))
        print(calc_pe(source, targets).shape)
    source_error = torch.cat(source_error, dim=1)
    min_source_errors, _ = torch.min(source_error, dim=1)
    return min_reproject_errors < min_source_errors


def process_depth(tgt_images, src_images, depths, poses, tgt_intr, src_intr):
    img_shape = tgt_images[0, 0].shape
    reprojected = torch.zeros((len(tgt_images), len(src_images[0]), 3, img_shape[0], img_shape[1]), dtype=torch.uint8)
    img_indices = torch.ones((img_shape[0] * img_shape[1], 3))
    img_indices[:, :2] = torch.from_numpy(np.indices(img_shape).ravel().reshape(-1, 2, order="F"))

    tgt_intr_torch_T = torch.from_numpy(tgt_intr.T)
    tgt_intr_inv_torch_T = torch.inverse(tgt_intr.T)
    src_intr_torch_T = torch.from_numpy(src_intr.T)

    for i in range(len(tgt_images)):
        for j in range(len(src_images[i])):
            if src_images[i]["stereo"]:
                src_intr_T = src_intr_torch_T
            else:
                src_intr_T = tgt_intr_torch_T

            world_coords = torch.ones(img_indices.shape[0], 4)
            world_coords[:, :3] = img_indices @ tgt_intr_inv_torch_T * depths[i].view(-1, 1)

            src_coords = torch.empty(img_indices.shape[0], 5)
            src_coords[:, 3:] = img_indices[:, :2]
            src_coords[:, :3] = (world_coords @ torch.t(poses[i, j]))[:, :3] @ src_intr_T

            src_coords = src_coords[src_coords[:, 2] > 0]
            src_coords[:, :2] = src_coords[:, :2] / src_coords[:, 2].view(-1, 1)

            # Potential bug here; 0th column is the height, while 1st column is width, might have to be switched
            src_coords = src_coords[
                (src_coords[:, 1] >= 0) & (src_coords[:, 1] <= img_shape[0] - 1) & (src_coords[:, 0] >= 0) & (
                            src_coords[:, 0] <= img_shape[1] - 1)]

            # Put nan here in case a pixel isn't filled
            reproj_image = torch.from_numpy(np.empty((3, img_shape[0], img_shape[1])).fill(np.nan))

            # Bilinear sampling
            x = src_coords[:, 1]
            y = src_coords[:, 0]
            x12 = (torch.floor(x), torch.ceil(x))
            y12 = (torch.floor(y), torch.ceil(y))
            src_img = src_images[i, j]
            reproj_image[:, src_coords[:, 3], src_coords[:, 4]] = \
                1 / (x12[1]-x12[0]) / (y12[1]-y12[0]) * \
                torch.tensor([x12[1] - x, x - x12[0]]) @ \
                torch.tensor([
                    [src_img[:, y12[0], x12[0]], src_img[:, y12[1], x12[0]]],
                    [src_img[:, y12[0], x12[1]], src_img[:, y12[1], x12[1]]]
                ]) @ \
                torch.tensor([[y12[1] - y], [y - y12[0]]])

            reprojected[i, j] = reproj_image

    return reprojected


def calc_loss(inputs, outputs):
    """
    Takes in the inputs and outputs from the neural network to calulate a numeric loss value based on the Monodepth2
    paper.
    :param [dict] inputs: Contains the keys "targets" and "sources" which are tensors [batch_size, 3, H, W] and
    [num_src_imgs, batch_size, 3, H, W] respectively
    :param [dict] outputs: Contains the keys "reproj" and "depth" which are tensors
    [num_reprojected_imgs, batch_size, 3, H, W] and [batch_size, H, W] respectively
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
    print(reproj_errors.shape)

    min_errors, _ = torch.min(reproj_errors, dim=1)
    print(min_errors.shape)

    # Masking
    reproj_errors *= get_mask(targets, sources, min_errors)

    depth = outputs["depth"]
    normalized_depth = depth / depth.mean(2, True).mean(3, True)
    loss += min_errors.mean() + LAMBDA * calc_smooth_loss(normalized_depth, targets)

    # Might not need to be dividing over batch size
    return loss / batch_size


if __name__ == "__main__":
    ssim = SSIM()
    test_t = torch.arange(36, dtype=torch.float).reshape(2, 3, 2, 3)
    test_r = torch.rand((2, 3, 2, 3), dtype=torch.float)
    # test_s = torch.arange(6).reshape(3, 2)

    target = torch.arange(108, dtype=torch.float).reshape(2, 3, 3, 6)
    sources = (torch.zeros(target.shape), target)
    reprojs = (target, target)
    depth = torch.arange(36, dtype=torch.float).reshape(2, 1, 3, 6)
    LAMBDA = 1
    inputs = {"targets": target, "sources": sources}
    outputs = {"reproj": reprojs, "depth": depth}
    l = calc_loss(inputs, outputs)
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
