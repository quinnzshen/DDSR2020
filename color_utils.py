import torch
import cv2
import numpy as np

from math import pi


RGB_XYZ_D65 = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

JZAZBZ_MATRICES = (
    torch.tensor([
        [0.41478972, 0.579999, 0.0146480],
        [-0.2015100, 1.120649, 0.0531008],
        [-0.0166008, 0.264800, 0.6684799]
    ]),
    torch.tensor([
        [0.5, 0.5, 0],
        [3.524, -4.066708, 0.542708],
        [0.199076, 1.096799, -1.295875]
    ])
)

b = 1.15
g = 0.66
c1 = 3424 / 2 ** 12
c2 = 2413 / 2 ** 7
c3 = 2392 / 2 ** 7
n = 2610 / 2 ** 14
p = 1.7 * 2523 / 2 ** 5
d = -0.56
d0 = 1.6295499532821566 / 10 ** 11


def convert_rgb(rgb_images, color="RGB"):
    if color == "HSV":
        hsv_images = rgb_to_hsv(rgb_images)
        hsv_images[:, 0] *= pi / 180
        hsv_images = torch.cat((torch.cos(hsv_images[:, 0].unsqueeze(1)), hsv_images), dim=1)
        hsv_images[:, 1] = torch.sin(hsv_images[:, 0])
        return hsv_images
    if color == "jzazbz":
        return xyz_to_jzazbz(rgb_to_xyz(rgb_images)).contiguous()
    return rgb_images


def rgb_to_hsv(rgb_images):
    out = torch.empty_like(rgb_images)
    if len(rgb_images.shape) == 4:
        v = rgb_images.max(dim=1)
        mins = rgb_images.min(dim=1)
        diff = 60 / (v[0] - mins[0])

        out[:, 2] = v[0]

        out[:, 1] = 1 - mins[0] / v[0]
        out[:, 1][torch.isnan(out[:, 1])] = 0

        rgb_max_mask = (v[1] == 0, v[1] == 1, v[1] == 2)
        out[:, 0][rgb_max_mask[0]] = rgb_images[:, 1][rgb_max_mask[0]] - rgb_images[:, 2][rgb_max_mask[0]]
        out[:, 0][rgb_max_mask[1]] = rgb_images[:, 2][rgb_max_mask[1]] - rgb_images[:, 0][rgb_max_mask[1]]
        out[:, 0][rgb_max_mask[2]] = rgb_images[:, 0][rgb_max_mask[2]] - rgb_images[:, 1][rgb_max_mask[2]]
        out[:, 0] = out[:, 0] * diff + 120 * v[1]
        out[:, 0][torch.isnan(out[:, 0])] = 0
        out[:, 0][out[:, 0] < 0] += 360

    return out


def rgb_to_xyz(rgb_images):
    return batch_channel_matmul(RGB_XYZ_D65, rgb_images)


def xyz_to_jzazbz(xyz_images):
    # https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-13-15131&id=368272
    lms = torch.empty_like(xyz_images)
    lms[:, 2] = xyz_images[:, 2]
    lms[:, 0] = b * xyz_images[:, 0] - (b - 1) * xyz_images[:, 2]
    lms[:, 1] = g * xyz_images[:, 1] - (g - 1) * xyz_images[:, 0]
    lms = batch_channel_matmul(JZAZBZ_MATRICES[0], lms)
    lms = ((c1 + c2 * (lms / 10000) ** n) / (1 + c3 * (lms / 10000) ** n)) ** p
    lms = batch_channel_matmul(JZAZBZ_MATRICES[1], lms)
    lms[:, 0] = (1 + d) * lms[:, 0] / (1 + d * lms[:, 0]) - d0
    return lms


def color_difference(image1, image2, color="RGB"):
    if color == "HSV":
        return (
            torch.abs(image1[:, 0] * image1[:, 2] - image2[:, 0] * image2[:, 2]) +
            torch.abs(image1[:, 1] * image1[:, 2] - image2[:, 1] * image2[:, 2]) +
            torch.abs(image1[:, 3] - image2[:, 3])
        ) / 3
    if color == "jzazbz":
        cz1 = torch.sqrt(image1[:, 1] ** 2 + image1[:, 2] ** 2)
        cz2 = torch.sqrt(image2[:, 1] ** 2 + image2[:, 2] ** 2)
        delta_hue = torch.atan2(image1[:, 2], image1[:, 1]) - torch.atan2(image2[:, 2], image2[:, 1])

        delta_hz = 2 * torch.sqrt(cz1 * cz2 + 1e-7) * torch.sin(delta_hue / 2)
        return torch.sqrt((image1[:, 0] - image2[:, 0]) ** 2 + (cz1 - cz2) ** 2 + delta_hz ** 2 + 1e-7)

    return torch.mean(torch.abs(image1 - image2), 1)


def batch_channel_matmul(mat, images):
    return (images.permute(0, 2, 3, 1) @ mat.to(images.device).T).permute(0, 3, 1, 2)
