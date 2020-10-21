import torch
import cv2
import numpy as np


def rgb_to_hsv(rgb_images):
    out = torch.empty_like(rgb_images)
    if len(rgb_images.shape) == 4:
        v = rgb_images.max(dim=1)
        mins = rgb_images.min(dim=1)
        diff = 60 / (v[0] - mins[0])

        out[:, 2] = v[0]

        out[:, 1] = 1 - mins[0] / v[0]
        out[:, 1, torch.isnan(out[:, 1]).squeeze(0)] = 0

        rgb_max_mask = ((v[1] == 0).squeeze(0), (v[1] == 1).squeeze(0), (v[1] == 2).squeeze(0))
        out[:, 0, rgb_max_mask[0]] = rgb_images[:, 1, rgb_max_mask[0]] - rgb_images[:, 2, rgb_max_mask[0]]
        out[:, 0, rgb_max_mask[1]] = rgb_images[:, 2, rgb_max_mask[1]] - rgb_images[:, 0, rgb_max_mask[1]]
        out[:, 0, rgb_max_mask[2]] = rgb_images[:, 0, rgb_max_mask[2]] - rgb_images[:, 1, rgb_max_mask[2]]
        out[:, 0] = out[:, 0] * diff + v[1] * 120
        out[:, 0, torch.isnan(out[:, 0]).squeeze(0)] = 0
        out[:, 0, (out[:, 0] < 0).squeeze(0)] += 360

    return out

#
# ble = torch.tensor([
#     [1, 2, 3],
#     [0, 0, 0],
#     [1, 1, 1]
# ])
# ble = ble.permute(1, 0).unsqueeze(1).unsqueeze(0) / 255.
#
# bcv = np.array(ble[0].permute(1, 2, 0))
#
# print(ble)
# nice = rgb_to_hsv(ble)
# nice2 = torch.from_numpy(cv2.cvtColor(bcv, cv2.COLOR_RGB2HSV)).permute(2, 0, 1).unsqueeze(0)
# print(nice)
# print(nice2)
# print(nice.shape)
# print(nice[0, :, 0, 2])
#






